"""Detection run in a background thread, started from the start page."""

from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..detect import Params
from ..imageio import find_images
from ..model import Page, assign_page_ids, source_in_scope
from ..pipeline import page_json_path, process_page, save_input_dir


def _geometry(page: Page) -> tuple:
    """What Claude was shown: the photo frames and text boxes."""
    return ([(p.id, [tuple(c) for c in p.corners]) for p in page.photos],
            [(t.id, tuple(t.box)) for t in page.texts])


@dataclass
class DetectJob:
    input_dir: Path
    work_dir: Path
    scope: str = ""           # only scans/pages under this folder of input_dir ("" = all)
    redo: bool = False        # re-detect pages that exist but were not edited/approved
    force: bool = False       # re-detect every page, edited and approved ones too
    llm: object | None = None  # LlmLinker, or None for heuristic linking only
    batch: bool = False       # with llm: send the requests as a Message Batch (half price, later)
    # Only ask Claude again about the existing pages that are not approved: no detection, and
    # frames, boxes and links on the page stay (Claude only links texts without a link).
    claude_only: bool = False
    # Afterwards, let Claude guess dates from the album order (`dates.py`); with `dates_only`
    # that is all the job does.
    guess_dates: bool = False
    dates_only: bool = False
    page_lock: object | None = None  # the server's page-write lock, shared with review saves
    queued: int = 0           # pages whose Claude request went into a batch
    batches: list = field(default_factory=list)  # ids of the batches this job submitted
    total: int = 0
    done: int = 0
    detected: int = 0
    skipped: int = 0
    failed: int = 0
    current: str = ""
    state: str = "starting"   # starting | running | finished | cancelled | error
    phase: str = ""           # "dates" while the date guesses run
    error: str = ""
    started: float = field(default_factory=time.time)
    log: deque = field(default_factory=lambda: deque(maxlen=300))
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self, on_done: Callable[[], None] | None = None) -> None:
        def run() -> None:
            try:
                self._run()
            except Exception as e:  # report instead of dying silently in the thread
                self.state, self.error = "error", str(e)
                self.log.append(traceback.format_exc())
            if on_done:
                on_done()
        self._thread = threading.Thread(target=run, name="albumsplit-detect", daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        return self.state in ("starting", "running")

    def status(self) -> dict:
        return {
            "state": self.state, "total": self.total, "done": self.done,
            "detected": self.detected, "skipped": self.skipped, "failed": self.failed,
            "current": self.current, "error": self.error, "llm": self.llm is not None,
            "batch": self.batch, "queued": self.queued, "batches": list(self.batches),
            "claude_only": self.claude_only, "dates_only": self.dates_only,
            "phase": self.phase,
            "elapsed": round(time.time() - self.started, 1), "log": list(self.log)[-40:],
        }

    def _run(self) -> None:
        if self.dates_only:
            if self.llm is None:
                raise ValueError("Guessing dates needs a Claude model")
            self.state = "running"
            self._guess_dates()
            if self.state == "running":
                self.state = "finished"
            return
        if self.claude_only:
            if self.llm is None:
                raise ValueError("Claude-only needs a Claude model")
            pages = [p for p in sorted((self.work_dir / "pages").glob("*.json"))
                     if (pg := Page.load(p)).status != "approved"
                     and source_in_scope(pg.source, self.scope)]
            images, ids = [], {}
            self.total = len(pages)
        else:
            images = find_images(self.input_dir / self.scope)
            ids = assign_page_ids([p.relative_to(self.input_dir) for p in images])
            save_input_dir(self.work_dir, self.input_dir)
            self.total = len(images)
        queue = None
        if self.llm is not None and self.batch:
            from ..batch import BatchQueue
            queue = BatchQueue(self.llm, self.work_dir, self.input_dir, log=self.log.append,
                               keep_links=self.claude_only)
        self.state = "running"
        try:
            if self.claude_only:
                self._claude_all(pages, queue)
            else:
                self._detect_all(images, ids, queue)
        finally:
            if queue is not None:  # also after a cancel: queued pages still get their answers
                queue.flush()
                self.queued, self.batches = queue.queued, list(queue.submitted)
                if queue.queued:
                    self.log.append(f"{queue.queued} pages sent to Claude as a batch: half price, "
                                    f"answers usually within an hour (at most 24 h).")
        self.current = ""
        if self.state == "running":
            what = "read by Claude" if self.claude_only else "detected"
            self.log.append(f"Done: {self.detected} {what}, {self.skipped} skipped, "
                            f"{self.failed} failed.")
            if self.guess_dates and self.llm is not None:
                if self.queued:
                    self.log.append("Dates are guessed when Claude's batch answers are in.")
                else:
                    self._guess_dates()
        if self.state == "running":
            self.state = "finished"

    def _guess_dates(self) -> None:
        """Claude guesses dates for undated photos from the album order (direct calls)."""
        from ..dates import guess_dates

        def progress(i: int, n: int) -> None:
            self.done, self.total = i, n

        self.phase, self.done, self.total = "dates", 0, 0
        self.log.append("Guessing dates from the album order...")
        rep = guess_dates(self.llm, self.work_dir, self.input_dir, scope=self.scope,
                          log=self.log.append,
                          stop=self._stop.is_set, page_lock=self.page_lock, on_progress=progress)
        if rep.cancelled:
            self.state = "cancelled"
        self.failed += rep.failed
        self.log.append(f"Dates: {rep.guessed} photos have a guessed date, {rep.undated} still "
                        f"undated, {rep.suspects} dates out of order; {rep.pages_updated} pages "
                        f"updated ({rep.called} Claude calls, {rep.cached} cached"
                        + (f", {rep.failed} failed" if rep.failed else "") + ").")

    def _claude_all(self, pages: list[Path], queue) -> None:
        """Ask Claude about each existing page as it is now; no detection."""
        import cv2

        from ..debug import render_debug
        from ..imageio import load_bgr
        from ..llm import apply
        from ..pipeline import debug_path
        cache = self.work_dir / "llm"
        for path in pages:
            if self._stop.is_set():
                self.state = "cancelled"
                self.log.append("Cancelled.")
                return
            t0 = time.perf_counter()
            try:
                page = Page.load(path)
                self.current = page.page_id
                img = load_bgr(self.input_dir / page.source)
                if queue is not None:
                    res, req = self.llm.cached(page, img, cache)
                    if res is None:
                        queue.add(req)
                        self.queued = queue.queued
                        self.log.append(f"{page.page_id}: queued for Claude")
                        self.done += 1
                        continue
                    note = "Claude (cached)"
                else:
                    res = self.llm.link(page, img, cache)
                    if res is None:
                        self.failed += 1
                        self.log.append(f"{page.page_id}: Claude gave no usable answer, page unchanged")
                        self.done += 1
                        continue
                    note = "Claude (cached)" if res.cached else "read by Claude"
                with self.page_lock or nullcontext():
                    now = Page.load(path)  # the page may have been edited in review meanwhile
                    if _geometry(now) != _geometry(page) or now.status == "approved":
                        self.skipped += 1
                        self.log.append(f"{page.page_id}: changed in review meanwhile, not updated")
                        self.done += 1
                        continue
                    apply(now, res, keep_links=True)
                    now.save(path)
                out = debug_path(self.work_dir, now.page_id)
                out.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(out), render_debug(now, img), [cv2.IMWRITE_JPEG_QUALITY, 85])
                self.detected += 1
                read = sum(1 for t in now.texts if t.transcription)
                self.log.append(f"{now.page_id}: {read} of {len(now.texts)} texts read, {note}"
                                f" ({time.perf_counter() - t0:.1f}s)")
            except Exception as e:  # keep going on a bad page
                self.failed += 1
                self.log.append(f"{path.stem}: ERROR {e}")
            self.done += 1

    def _detect_all(self, images: list[Path], ids: dict, queue) -> None:
        for path in images:
            if self._stop.is_set():
                self.state = "cancelled"
                self.log.append("Cancelled.")
                return
            pid = ids[path.relative_to(self.input_dir)]
            self.current = pid
            if not (self.redo or self.force) and page_json_path(self.work_dir, pid).exists():
                self.skipped += 1
                self.done += 1
                continue
            t0 = time.perf_counter()
            try:
                res = process_page(self.input_dir, path, self.work_dir, page_id=pid,
                                   params=Params(), llm=self.llm, llm_queue=queue,
                                   force=self.force)
            except Exception as e:  # keep going on a bad scan
                self.failed += 1
                self.log.append(f"{pid}: ERROR {e}")
            else:
                if res.skipped:
                    self.skipped += 1
                else:
                    self.detected += 1
                    pg = res.page
                    note = {"failed": ", Claude failed (heuristic kept)", "called": ", read by Claude",
                            "cached": ", Claude (cached)", "queued": ", queued for Claude"
                            }.get(res.llm, "")
                    self.log.append(f"{pid}: {len(pg.photos)} photos, {len(pg.texts)} texts{note}"
                                    f" ({time.perf_counter() - t0:.1f}s)")
            self.done += 1
            if queue is not None:
                self.queued = queue.queued


@dataclass
class ExportJob:
    """`export.export_work_dir` in a background thread, with the same status shape as DetectJob."""
    input_dir: Path
    work_dir: Path
    out_dir: Path
    scope: str = ""           # only pages under this folder of input_dir ("" = all)
    include_proposed: bool = False
    cards: bool = True
    exiftool: list[str] | None = None
    total: int = 0
    done: int = 0
    photos: int = 0
    failed: int = 0
    current: str = ""
    state: str = "starting"   # starting | running | finished | cancelled | error
    error: str = ""
    started: float = field(default_factory=time.time)
    log: deque = field(default_factory=lambda: deque(maxlen=300))
    _stop: threading.Event = field(default_factory=threading.Event)
    _thread: threading.Thread | None = None

    def start(self) -> None:
        def run() -> None:
            try:
                self._run()
            except Exception as e:
                self.state, self.error = "error", str(e)
                self.log.append(traceback.format_exc())
        self._thread = threading.Thread(target=run, name="albumsplit-export", daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        return self.state in ("starting", "running")

    def status(self) -> dict:
        return {
            "state": self.state, "total": self.total, "done": self.done, "photos": self.photos,
            "failed": self.failed, "current": self.current, "error": self.error,
            "out_dir": str(self.out_dir), "elapsed": round(time.time() - self.started, 1),
            "log": list(self.log)[-40:],
        }

    def _run(self) -> None:
        from ..export import export_work_dir

        def on_total(n: int) -> None:
            self.total, self.state = n, "running"

        def on_page(pid: str) -> None:
            if self.current:
                self.done += 1
            self.current = pid

        rep = export_work_dir(self.work_dir, self.out_dir, self.input_dir,
                              include_proposed=self.include_proposed, cards=self.cards,
                              scope=self.scope,
                              exiftool=self.exiftool, progress=self.log.append,
                              on_total=on_total, on_page=on_page, stop=self._stop.is_set)
        if not rep.cancelled:
            self.done = self.total
        self.photos, self.failed, self.current = rep.photos, len(rep.failed), ""
        self.state = "cancelled" if rep.cancelled else "finished"
        self.log.append(f"Done: {rep.photos} photos from {rep.pages} pages in {self.out_dir}"
                        + (f", {len(rep.failed)} failed" if rep.failed else "") + ".")
