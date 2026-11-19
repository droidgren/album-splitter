"""CLI entry point: python -m albumsplit <command> ..."""

from __future__ import annotations

import argparse
import fnmatch
import sys
import time
from pathlib import Path

from .detect import Params
from .imageio import find_images
from .model import page_id_from_path
from .pipeline import process_page


def cmd_detect(args: argparse.Namespace) -> int:
    input_dir = Path(args.input_dir).resolve()
    work_dir = Path(args.out).resolve()
    images = find_images(input_dir)
    if args.only:
        images = [p for p in images if fnmatch.fnmatch(p.name, args.only)]
    if not images:
        print(f"No images found in {input_dir}", file=sys.stderr)
        return 1

    ids: dict[str, Path] = {}
    for p in images:
        pid = page_id_from_path(p.relative_to(input_dir))
        if pid in ids:
            print(f"Page id collision: {p} and {ids[pid]} -> {pid}", file=sys.stderr)
            return 1
        ids[pid] = p

    params = Params(stamps=args.stamps)
    failures = 0
    for i, path in enumerate(images, 1):
        t0 = time.perf_counter()
        try:
            res = process_page(input_dir, path, work_dir, force=args.force, debug=args.debug,
                               params=params)
        except Exception as e:  # keep going on a bad scan
            failures += 1
            print(f"[{i}/{len(images)}] {path.name}: ERROR {e}", file=sys.stderr)
            continue
        if res.skipped:
            print(f"[{i}/{len(images)}] {res.page_id}: approved, skipped")
            continue
        pg = res.page
        review = sum(p.needs_review for p in pg.photos)
        unlinked = len({t.id for t in pg.texts} - {lk.text_id for lk in pg.links})
        stamps = [p.stamp for p in pg.photos if p.stamp]
        dated = sum(1 for st in stamps if st.date)
        print(f"[{i}/{len(images)}] {res.page_id}: {len(pg.photos)} photos"
              f"{f' ({review} need review)' if review else ''}, {len(pg.texts)} texts, "
              f"{len(pg.links)} links{f', {unlinked} unlinked' if unlinked else ''}"
              f"{f', {len(stamps)} date stamps ({dated} read)' if stamps else ''}"
              f"{', page outline' if pg.page_quad else ''}  ({time.perf_counter() - t0:.1f}s)")
    print(f"Wrote {work_dir / 'pages'}" + (f" and {work_dir / 'debug'}" if args.debug else ""))
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="albumsplit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("detect", help="detect photos and text blocks, link them heuristically")
    d.add_argument("input_dir")
    d.add_argument("--out", required=True, help="work directory")
    d.add_argument("--only", help="only process files whose name matches this glob")
    d.add_argument("--force", action="store_true", help="re-detect pages that are already approved")
    d.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True,
                   help="write debug images to <work_dir>/debug (default: on)")
    d.add_argument("--stamps", action=argparse.BooleanOptionalAction, default=True,
                   help="find and read camera date stamps inside the photos (default: on)")
    d.set_defaults(func=cmd_detect)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
