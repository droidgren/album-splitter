"""CLI entry point: python -m albumsplit <command> ..."""

from __future__ import annotations

import argparse
import fnmatch
import sys
import time
from pathlib import Path

from .detect import Params
from .imageio import find_images
from .llm import DEFAULT_MODEL
from .model import assign_page_ids
from .pipeline import load_input_dir, process_page, save_input_dir


def cmd_detect(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).resolve()
    work_dir = Path(args.out).resolve()
    images = find_images(input_dir)
    if args.only:
        images = [p for p in images if fnmatch.fnmatch(p.name, args.only)]
    if not images:
        print(f"No images found in {input_dir}", file=sys.stderr)
        return 1

    # Ids are assigned over every image in the input dir (not just --only matches) so a
    # page keeps the same id however the run is filtered.
    try:
        ids = assign_page_ids([p.relative_to(input_dir) for p in find_images(input_dir)])
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1

    save_input_dir(work_dir, input_dir)
    params = Params(stamps=args.stamps)
    llm = queue = None
    if args.llm or args.llm_batch:
        llm = _make_linker(args.llm_model)
        if llm is None:
            return 1
    if args.llm_batch:
        from .batch import BatchQueue
        queue = BatchQueue(llm, work_dir, input_dir)
    failures = 0
    for i, path in enumerate(images, 1):
        t0 = time.perf_counter()
        try:
            res = process_page(input_dir, path, work_dir, force=args.force, debug=args.debug,
                               params=params, page_id=ids[path.relative_to(input_dir)], llm=llm,
                               llm_queue=queue)
        except Exception as e:  # keep going on a bad scan
            failures += 1
            print(f"[{i}/{len(images)}] {path.name}: ERROR {e}", file=sys.stderr)
            continue
        if res.skipped:
            print(f"[{i}/{len(images)}] {res.page_id}: {res.skip_reason} in review, skipped"
                  f" (--force re-detects)")
            continue
        pg = res.page
        review = sum(p.needs_review for p in pg.photos)
        unlinked = len({t.id for t in pg.texts} - {lk.text_id for lk in pg.links})
        stamps = [p.stamp for p in pg.photos if p.stamp]
        dated = sum(1 for st in stamps if st.date)
        llm_note = ""
        if res.llm == "failed":
            llm_note = ", llm failed (heuristic kept)"
        elif res.llm == "queued":
            llm_note = ", llm queued"
        elif res.llm != "off":
            read = sum(1 for t in pg.texts if t.transcription)
            llm_note = f", llm{' (cached)' if res.llm == 'cached' else ''}: {read} transcribed"
            if pg.issues:
                llm_note += f", {len(pg.issues)} issues"
        print(f"[{i}/{len(images)}] {res.page_id}: {len(pg.photos)} photos"
              f"{f' ({review} need review)' if review else ''}, {len(pg.texts)} texts, "
              f"{len(pg.links)} links{f', {unlinked} unlinked' if unlinked else ''}"
              f"{f', {len(stamps)} date stamps ({dated} read)' if stamps else ''}"
              f"{', page outline' if pg.page_quad else ''}{llm_note}  ({time.perf_counter() - t0:.1f}s)")
        for issue in pg.issues:
            where = f" (P{', P'.join(map(str, issue.photo_ids))})" if issue.photo_ids else ""
            print(f"    ! {issue.kind}{where}: {issue.note}")
    print(f"Wrote {work_dir / 'pages'}" + (f" and {work_dir / 'debug'}" if args.debug else ""))
    if queue is not None:
        queue.flush()
        if queue.submitted and not _wait_and_collect(llm, work_dir, queue.submitted, args.debug):
            return 1
    return 1 if failures else 0


def _make_linker(model: str):
    from .llm import LlmLinker, LlmUnavailable
    try:
        return LlmLinker(model)
    except LlmUnavailable as e:
        print(f"--llm: {e}", file=sys.stderr)
        return None


def _wait_and_collect(llm, work_dir: Path, batch_ids: list[str], debug: bool) -> bool:
    from .batch import collect, wait
    print(f"Waiting for {len(batch_ids)} batch(es); usually under an hour, at most 24 h. "
          f"Ctrl+C stops waiting, the batches keep running.")
    try:
        wait(llm, batch_ids)
    except KeyboardInterrupt:
        print(f"\nStopped waiting. Collect later with: python -m albumsplit llm-collect {work_dir}")
        return False
    updated, failed = collect(llm, work_dir, debug=debug)
    print(f"LLM batch: {updated} pages updated" + (f", {failed} kept heuristic links" if failed else ""))
    return True


def cmd_llm_collect(args: argparse.Namespace) -> int:
    from .batch import collect, pending_batches, wait
    work_dir = Path(args.work_dir).resolve()
    pending = pending_batches(work_dir)
    if not pending:
        print("No uncollected batches.")
        return 0
    for model in sorted({b["model"] for b in pending}):
        llm = _make_linker(model)
        if llm is None:
            return 1
        if args.wait:
            try:
                wait(llm, [b["id"] for b in pending if b["model"] == model])
            except KeyboardInterrupt:
                print("\nStopped waiting.")
                return 1
        updated, failed = collect(llm, work_dir, debug=args.debug)
        print(f"{model}: {updated} pages updated" + (f", {failed} kept heuristic links" if failed else ""))
    left = len(pending_batches(work_dir))
    if left:
        print(f"{left} batch(es) still running; run llm-collect again later or add --wait.")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    try:
        import uvicorn

        from .review import create_app
    except ImportError as e:
        print(f"review needs fastapi and uvicorn (pip install -r requirements.txt): {e}",
              file=sys.stderr)
        return 1
    work_dir = input_dir = None
    if args.work_dir:
        work_dir = Path(args.work_dir).resolve()
        input_dir = Path(args.input).resolve() if args.input else load_input_dir(work_dir)
        if input_dir is not None and not input_dir.is_dir():
            print(f"Scan folder {input_dir} not found; pass --input", file=sys.stderr)
            return 1
    app = create_app(work_dir, input_dir, llm_model=args.llm_model)
    base = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
    # Straight into review when the project is known, else the start page to choose one.
    url = base + ("/review" if work_dir and input_dir and (work_dir / "pages").is_dir() else "/")
    where = f"{work_dir} (scans in {input_dir})" if work_dir and input_dir else "start page"
    print(f"albumsplit: {where} at {base}/  - Ctrl+C stops")
    if args.open if args.open is not None else args.host != "0.0.0.0":
        import threading
        import webbrowser
        threading.Timer(1.0, webbrowser.open, [url]).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="albumsplit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="detect photos and text blocks, link them heuristically")
    d.add_argument("input_dir")
    d.add_argument("--out", required=True, help="work directory")
    d.add_argument("--only", help="only process files whose name matches this glob")
    d.add_argument("--force", action="store_true",
                   help="re-detect pages that were approved or edited in review")
    d.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True,
                   help="write debug images to <work_dir>/debug (default: on)")
    d.add_argument("--stamps", action=argparse.BooleanOptionalAction, default=True,
                   help="find and read camera date stamps inside the photos (default: on)")
    d.add_argument("--llm", action="store_true",
                   help="link and transcribe captions with Claude (needs ANTHROPIC_API_KEY)")
    d.add_argument("--llm-model", default=DEFAULT_MODEL,
                   help=f"Claude model for --llm (default: {DEFAULT_MODEL})")
    d.add_argument("--llm-batch", action="store_true",
                   help="like --llm, but through the Batch API: half price, results within 24 h")
    d.set_defaults(func=cmd_detect)

    c = sub.add_parser("llm-collect", help="fetch finished --llm-batch results into the pages")
    c.add_argument("work_dir")
    c.add_argument("--wait", action="store_true", help="wait until all batches have ended")
    c.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True,
                   help="re-render debug images of updated pages (default: on)")
    c.set_defaults(func=cmd_llm_collect)

    r = sub.add_parser("review", help="start page and page review in the browser")
    r.add_argument("work_dir", nargs="?", help="open this work folder (default: choose on the start page)")
    r.add_argument("--input", help="scan directory (default: the one detect used)")
    r.add_argument("--llm-model", default=DEFAULT_MODEL,
                   help=f"Claude model for re-reading pages and stamps (default: {DEFAULT_MODEL})")
    r.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to reach it from other machines")
    r.add_argument("--port", type=int, default=8000)
    r.add_argument("--open", action=argparse.BooleanOptionalAction, default=None,
                   help="open a browser (default: on unless --host 0.0.0.0)")
    r.set_defaults(func=cmd_review)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
