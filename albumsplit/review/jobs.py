"""Detection run in a background thread, started from the start page."""

from __future__ import annotations

import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..detect import Params
from ..imageio import find_images
from ..model import assign_page_ids
from ..pipeline import page_json_path, process_page, save_input_dir


@dataclass
class DetectJob:
    input_dir: Path
    work_dir: Path
    redo: bool = False        # re-detect pages that exist but were not edited/approved
    force: bool = False       # re-detect every page, edited and approved ones too
    llm: object | None = None  # LlmLinker, or None for heuristic linking only
    total: int = 0
    done: int = 0
    detected: int = 0
    skipped: int = 0
    failed: int = 0
    current: str = ""
    state: str = "starting"   # starting | running | finished | cancelled | error
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
            "elapsed": round(time.time() - self.started, 1), "log": list(self.log)[-40:],
        }

    def _run(self) -> None:
        images = find_images(self.input_dir)
        ids = assign_page_ids([p.relative_to(self.input_dir) for p in images])
        save_input_dir(self.work_dir, self.input_dir)
        self.total = len(images)
        self.state = "running"
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
                                   params=Params(), llm=self.llm, force=self.force)
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
                            "cached": ", Claude (cached)"}.get(res.llm, "")
                    self.log.append(f"{pid}: {len(pg.photos)} photos, {len(pg.texts)} texts{note}"
                                    f" ({time.perf_counter() - t0:.1f}s)")
            self.done += 1
        self.current = ""
        self.state = "finished"
        self.log.append(f"Done: {self.detected} detected, {self.skipped} skipped, {self.failed} failed.")
