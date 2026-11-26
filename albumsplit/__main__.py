"""CLI entry point: python -m albumsplit <command> ..."""

from __future__ import annotations

import argparse
import fnmatch
import sys
import time
from pathlib import Path

from .detect import Params
from .imageio import find_images
from .llm import DEFAULT_MODEL, EFFORTS
from .model import assign_page_ids
from .pipeline import (load_guess_dates, load_input_dir, load_llm_effort, load_llm_model,
                       process_page, save_guess_dates, save_input_dir)


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
    if args.guess_dates and not (args.llm or args.llm_batch):
        print("--guess-dates needs --llm or --llm-batch", file=sys.stderr)
        return 1
    if args.guess_dates:
        save_guess_dates(work_dir, True)  # so llm-collect and the start page know
    if args.llm or args.llm_batch:
        llm = _make_linker(args.llm_model, args.llm_effort)
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
    if args.guess_dates and not _guess(llm, work_dir, input_dir):
        failures += 1
    return 1 if failures else 0


def _guess(llm, work_dir: Path, input_dir: Path, force: bool = False,
           cache_dir: Path | None = None) -> bool:
    from .dates import guess_dates
    print("Guessing dates from the album order...")
    rep = guess_dates(llm, work_dir, input_dir, force=force, log=lambda m: print("  " + m),
                      cache_dir=cache_dir)
    print(f"Dates: {rep.guessed} photos have a guessed date, {rep.undated} still undated, "
          f"{rep.suspects} dates out of order; {rep.pages_updated} pages updated "
          f"({rep.called} Claude calls, {rep.cached} cached"
          + (f", {rep.failed} failed" if rep.failed else "") + ")")
    return not rep.failed


def cmd_guess_dates(args: argparse.Namespace) -> int:
    work_dir = Path(args.work_dir).resolve()
    if not (work_dir / "pages").is_dir():
        print(f"No pages in {work_dir}; run detect first", file=sys.stderr)
        return 1
    input_dir = Path(args.input).resolve() if args.input else load_input_dir(work_dir)
    if input_dir is None or not input_dir.is_dir():
        print(f"Scan folder {input_dir} not found; pass --input", file=sys.stderr)
        return 1
    llm = _make_linker(args.llm_model or load_llm_model(work_dir) or DEFAULT_MODEL,
                       args.llm_effort or load_llm_effort(work_dir))
    if llm is None:
        return 1
    return 0 if _guess(llm, work_dir, input_dir, force=args.force) else 1


def cmd_claude_code(args: argparse.Namespace) -> int:
    import threading
    from concurrent.futures import ThreadPoolExecutor

    from .claude_code import ClaudeCodeLinker, llm_cache_dir, requests_dir
    from .llm import LlmUnavailable
    from .model import Page
    from .pipeline import claude_page
    work_dir = Path(args.work_dir).resolve()
    if not (work_dir / "pages").is_dir():
        print(f"No pages in {work_dir}; run detect first", file=sys.stderr)
        return 1
    input_dir = Path(args.input).resolve() if args.input else load_input_dir(work_dir)
    if input_dir is None or not input_dir.is_dir():
        print(f"Scan folder {input_dir} not found; pass --input", file=sys.stderr)
        return 1
    try:
        llm = ClaudeCodeLinker(work_dir, args.model, effort=args.effort, claude=args.claude,
                               mode="export" if args.export else "run", timeout=args.timeout)
    except LlmUnavailable as e:
        print(f"claude-code: {e}", file=sys.stderr)
        return 1
    if llm.claude:
        print(f"Claude Code: {llm.claude} (model {args.model})")
    cache = llm_cache_dir(work_dir)
    ok = True
    if not args.dates_only:
        paths = []
        for path in sorted((work_dir / "pages").glob("*.json")):
            pg = Page.load(path)
            if pg.status == "approved":
                continue
            if args.only and not fnmatch.fnmatch(pg.page_id, args.only):
                continue
            if args.flagged and not (pg.issues or any(ph.needs_review for ph in pg.photos)):
                continue
            paths.append(path)
        print(f"{len(paths)} pages to read")
        counts: dict[str, int] = {}
        lock = threading.Lock()

        def count(status: str) -> None:
            with lock:
                counts[status] = counts.get(status, 0) + 1

        def one(path: Path) -> None:
            t0 = time.perf_counter()
            try:
                res = claude_page(llm, work_dir, input_dir, path, debug=args.debug,
                                  cache_dir=cache)
            except Exception as e:  # keep going on a bad page
                count("failed")
                print(f"{path.stem}: ERROR {e}")
                return
            status = "waiting" if res.status == "failed" and args.export else res.status
            count(status)
            pg = res.page
            note = {"called": "read by Claude Code", "cached": "cached",
                    "failed": "no answer, page unchanged", "waiting": "request written",
                    "skipped": "changed meanwhile, not updated"}[status]
            if res.status in ("called", "cached"):
                read = sum(1 for t in pg.texts if t.transcription)
                note = f"{read} of {len(pg.texts)} texts read, {note}"
                if pg.issues:
                    note += f", {len(pg.issues)} issues"
            print(f"{pg.page_id}: {note}  ({time.perf_counter() - t0:.1f}s)")

        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            list(pool.map(one, paths))
        print("Pages: " + ", ".join(f"{v} {k}" for k, v in sorted(counts.items())) if counts
              else "Pages: nothing to do")
        if args.export and llm.pending:
            print(f"{llm.pending} requests wait for an answer in {requests_dir(work_dir)} "
                  "(see README.md there); run this command again afterwards.")
        ok = not counts.get("failed")
    if args.dates or args.dates_only:
        if args.export and llm.pending and not args.dates_only:
            print("Dates are guessed once the page answers are in: run with --dates-only then.")
        else:
            pending = llm.pending
            ok = _guess(llm, work_dir, input_dir, force=args.force, cache_dir=cache) and ok
            if args.export and llm.pending > pending:
                print(f"{llm.pending - pending} date requests wait for an answer in "
                      f"{requests_dir(work_dir)}; run this command again afterwards.")
    return 0 if ok else 1


def _make_linker(model: str, effort: str | None = None):
    from .llm import LlmLinker, LlmUnavailable
    try:
        return LlmLinker(model, effort)
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
    groups = sorted({(b["model"], b.get("effort")) for b in pending}, key=str)
    for model, effort in groups:
        llm = _make_linker(model, effort)
        if llm is None:
            return 1
        if args.wait:
            try:
                wait(llm, [b["id"] for b in pending
                           if (b["model"], b.get("effort")) == (model, effort)])
            except KeyboardInterrupt:
                print("\nStopped waiting.")
                return 1
        updated, failed = collect(llm, work_dir, debug=args.debug)
        print(f"{llm.tag}: {updated} pages updated" + (f", {failed} kept heuristic links" if failed else ""))
    left = len(pending_batches(work_dir))
    if left:
        print(f"{left} batch(es) still running; run llm-collect again later or add --wait.")
    elif load_guess_dates(work_dir):
        input_dir = load_input_dir(work_dir)
        llm = _make_linker(load_llm_model(work_dir) or pending[-1]["model"],
                           load_llm_effort(work_dir))
        if llm is None or input_dir is None or not _guess(llm, work_dir, input_dir):
            return 1
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
    app = create_app(work_dir, input_dir, llm_model=args.llm_model, llm_effort=args.llm_effort)
    base = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
    # Straight into review when the project is known, else the start page to choose one.
    url = base + ("/review" if work_dir and input_dir and (work_dir / "pages").is_dir() else "/")
    where = f"{work_dir} (scans in {input_dir})" if work_dir and input_dir else "start page"
    from .version import version_string
    print(f"albumsplit {version_string()}: {where} at {base}/  - Ctrl+C stops")
    if args.open if args.open is not None else args.host != "0.0.0.0":
        import threading
        import webbrowser
        threading.Timer(1.0, webbrowser.open, [url]).start()
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from .exiftool import ExiftoolError, find_exiftool, version
    from .export import export_work_dir

    work_dir = Path(args.work_dir).resolve()
    out_dir = Path(args.out).resolve()
    if not (work_dir / "pages").is_dir():
        print(f"No pages in {work_dir}; run detect first", file=sys.stderr)
        return 1
    input_dir = Path(args.input).resolve() if args.input else load_input_dir(work_dir)
    if input_dir is None or not input_dir.is_dir():
        print(f"Scan folder {input_dir} not found; pass --input", file=sys.stderr)
        return 1
    try:
        out_dir.relative_to(input_dir)
        print("The export folder may not be inside the scan folder", file=sys.stderr)
        return 1
    except ValueError:
        pass
    tool = None
    if args.metadata:
        tool = find_exiftool(args.exiftool)
        try:
            if tool is None:
                raise ExiftoolError("not found")
            ver = version(tool)
        except (ExiftoolError, OSError) as e:
            print(f"exiftool: {e}. Install it, set ALBUMSPLIT_EXIFTOOL or pass --exiftool PATH "
                  f"(or --no-metadata to export without captions in the files).", file=sys.stderr)
            return 1
        print(f"exiftool {ver}: {' '.join(tool)}")
    rep = export_work_dir(work_dir, out_dir, input_dir, include_proposed=args.all, only=args.only,
                          cards=args.cards, quality=args.quality, exiftool=tool, progress=print)
    note = f", {rep.skipped} pages not approved (--all exports them)" if rep.skipped else ""
    fails = f", {len(rep.failed)} failed" if rep.failed else ""
    print(f"Exported {rep.photos} photos from {rep.pages} pages to {out_dir}{note}{fails}")
    return 1 if rep.failed or not rep.photos else 0


def main(argv: list[str] | None = None) -> int:
    from .version import version_string

    ap = argparse.ArgumentParser(prog="albumsplit")
    ap.add_argument("--version", action="version", version=f"albumsplit {version_string()}")
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
    d.add_argument("--llm-effort", choices=EFFORTS,
                   help="Claude effort: lower is cheaper, less thinking (default: the model's)")
    d.add_argument("--llm-batch", action="store_true",
                   help="like --llm, but through the Batch API: half price, results within 24 h")
    d.add_argument("--guess-dates", action="store_true",
                   help="afterwards let Claude guess dates for photos with no stamp or dated "
                        "caption from the album order (albums scanned in order, IMG_0001 first)")
    d.set_defaults(func=cmd_detect)

    g = sub.add_parser("guess-dates", help="let Claude guess dates of undated photos from the "
                                           "album order")
    g.add_argument("work_dir")
    g.add_argument("--input", help="scan directory (default: the one detect used)")
    g.add_argument("--llm-model", help="Claude model (default: the project's, else "
                                       f"{DEFAULT_MODEL})")
    g.add_argument("--llm-effort", choices=EFFORTS,
                   help="Claude effort (default: the project's, else the model's)")
    g.add_argument("--force", action="store_true", help="ask again even where an answer is cached")
    g.set_defaults(func=cmd_guess_dates)

    cc = sub.add_parser("claude-code", help="read detected pages (and guess dates) with Claude "
                                            "Code, using its login instead of an API key")
    cc.add_argument("work_dir")
    cc.add_argument("--input", help="scan directory (default: the one detect used)")
    cc.add_argument("--model", default="opus",
                    help="Claude Code model alias: opus, sonnet, haiku, fable (default: opus)")
    cc.add_argument("--effort", choices=EFFORTS, help="Claude Code effort (default: its own)")
    cc.add_argument("--only", help="only pages whose id matches this glob")
    cc.add_argument("--flagged", action="store_true",
                    help="only pages flagged for a check (issues or a photo to review)")
    cc.add_argument("--dates", action="store_true",
                    help="afterwards guess dates from the album order")
    cc.add_argument("--dates-only", action="store_true", help="only guess dates")
    cc.add_argument("--export", action="store_true",
                    help="only write the requests to <work_dir>/claude-code/requests/ for a "
                         "Claude Code session to answer; answers already there are applied")
    cc.add_argument("--jobs", type=int, default=2, help="pages read at the same time (default: 2)")
    cc.add_argument("--timeout", type=float, default=600, help="seconds per request (default: 600)")
    cc.add_argument("--claude", help="Claude Code executable (default: $ALBUMSPLIT_CLAUDE, PATH, "
                                     "or the VS Code extension's)")
    cc.add_argument("--force", action="store_true",
                    help="ask again even where an answer is cached (dates)")
    cc.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True,
                    help="re-render debug images of updated pages (default: on)")
    cc.set_defaults(func=cmd_claude_code)

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
    r.add_argument("--llm-effort", choices=EFFORTS,
                   help="Claude effort for re-reading (default: the model's; the start page can choose another)")
    r.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to reach it from other machines")
    r.add_argument("--port", type=int, default=8000)
    r.add_argument("--open", action=argparse.BooleanOptionalAction, default=None,
                   help="open a browser (default: on unless --host 0.0.0.0)")
    r.set_defaults(func=cmd_review)

    e = sub.add_parser("export", help="write photo crops, caption cards and index.csv")
    e.add_argument("work_dir")
    e.add_argument("--out", required=True, help="export directory")
    e.add_argument("--input", help="scan directory (default: the one detect used)")
    e.add_argument("--all", action="store_true", help="also export pages not yet approved")
    e.add_argument("--only", help="only export pages whose id matches this glob")
    e.add_argument("--cards", action=argparse.BooleanOptionalAction, default=True,
                   help="also write <page>_<n>_card.jpg with the handwritten caption (default: on)")
    e.add_argument("--quality", type=int, default=95, help="JPEG quality (default: 95)")
    e.add_argument("--metadata", action=argparse.BooleanOptionalAction, default=True,
                   help="write caption and date into the files with exiftool (default: on)")
    e.add_argument("--exiftool", help=f"exiftool program or folder (default: $ALBUMSPLIT_EXIFTOOL, "
                                      f"PATH, or .venv/exiftool-*)")
    e.set_defaults(func=cmd_export)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
