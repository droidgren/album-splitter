"""--llm-batch: send the LLM requests through the Message Batches API (half price, async).

Detection runs as usual and saves each page with its heuristic links. Pages without a cached
answer are queued; full queues are submitted as batches while detection carries on. The ids
of submitted batches are kept in `<work_dir>/llm/batches.json`, so the results can still be
collected after an interruption (`python -m albumsplit llm-collect <work_dir>`).

Collecting writes each answer to the normal per-page cache, then re-applies it to the page
JSON from the cache. A page that was re-detected, approved or edited in review in the meantime
is left alone (its answer stays in the cache).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2

from .debug import render_debug
from .imageio import load_bgr
from .llm import LlmLinker, Request, apply, parse_message, write_cache
from .model import Page

MAX_BATCH_BYTES = 200 * 1024 * 1024  # API limit is 256 MB per batch
MAX_BATCH_REQUESTS = 10_000          # API limit is 100,000
POLL_SECONDS = 60


def state_path(work_dir: Path) -> Path:
    return work_dir / "llm" / "batches.json"


def load_state(work_dir: Path) -> dict:
    try:
        return json.loads(state_path(work_dir).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"batches": []}


def save_state(work_dir: Path, state: dict) -> None:
    p = state_path(work_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
    tmp.replace(p)


@dataclass
class BatchQueue:
    """Collects requests and submits a batch whenever the queue gets full."""
    linker: LlmLinker
    work_dir: Path
    input_dir: Path
    submitted: list[str] = field(default_factory=list)
    log: Callable[[str], None] = print
    # Claude-only re-read of reviewed pages: answers keep the links on the page and are
    # applied to pages edited in review too (only approved pages are left alone).
    keep_links: bool = False
    queued: int = 0  # requests added (and not deduplicated) so far
    _items: list[dict] = field(default_factory=list)
    _pages: dict[str, str] = field(default_factory=dict)  # custom_id -> page_id
    _bytes: int = 0

    def add(self, req: Request) -> None:
        item = {"custom_id": req.digest, "params": self.linker.params(req.content)}
        size = len(json.dumps(item))
        if self._items and (self._bytes + size > MAX_BATCH_BYTES
                            or len(self._items) >= MAX_BATCH_REQUESTS):
            self.flush()
        if req.digest in self._pages:  # identical page content queued twice
            return
        self._items.append(item)
        self._pages[req.digest] = req.page_id
        self._bytes += size
        self.queued += 1

    def flush(self) -> None:
        if not self._items:
            return
        batch = self.linker.client.messages.batches.create(requests=self._items)
        state = load_state(self.work_dir)
        state["batches"].append({"id": batch.id, "model": self.linker.model,
                                 "input_dir": str(self.input_dir), "pages": self._pages,
                                 "collected": False, "submitted": time.time(),
                                 "keep_links": self.keep_links})
        save_state(self.work_dir, state)
        self.log(f"  submitted batch {batch.id} ({len(self._items)} pages, "
                 f"{self._bytes / 1e6:.0f} MB)")
        self.submitted.append(batch.id)
        self._items, self._pages, self._bytes = [], {}, 0


def pending_batches(work_dir: Path) -> list[dict]:
    return [b for b in load_state(work_dir)["batches"] if not b["collected"]]


def wait(linker: LlmLinker, batch_ids: list[str]) -> None:
    """Poll until every batch has ended. Ctrl+C stops waiting; the batches keep running."""
    remaining = list(batch_ids)
    while remaining:
        still = []
        for bid in remaining:
            b = linker.client.messages.batches.retrieve(bid)
            if b.processing_status != "ended":
                still.append(bid)
                c = b.request_counts
                print(f"  {bid}: {b.processing_status}, {c.processing} processing, "
                      f"{c.succeeded} done, {c.errored} errored")
        remaining = still
        if remaining:
            time.sleep(POLL_SECONDS)


def batch_status(linker: LlmLinker, entry: dict) -> dict:
    """Processing state of one submitted batch, as the start page shows it."""
    b = linker.client.messages.batches.retrieve(entry["id"])
    c = b.request_counts
    return {"id": entry["id"], "model": entry["model"], "pages": len(entry["pages"]),
            "submitted": entry.get("submitted"), "status": b.processing_status,
            "processing": c.processing, "succeeded": c.succeeded, "errored": c.errored,
            "canceled": c.canceled, "expired": c.expired}


def collect(linker: LlmLinker, work_dir: Path, *, debug: bool = True,
            log: Callable[[str], None] = print) -> tuple[int, int]:
    """Fetch the results of every ended, not yet collected batch and apply them to the pages.
    Returns (pages updated, pages not updated)."""
    state = load_state(work_dir)
    updated = failed = 0
    for entry in state["batches"]:
        if entry["collected"] or entry["model"] != linker.model:  # cache keys depend on the model
            continue
        b = linker.client.messages.batches.retrieve(entry["id"])
        if b.processing_status != "ended":
            log(f"  {entry['id']}: still {b.processing_status}, collect again later")
            continue
        got: list[str] = []
        for r in linker.client.messages.batches.results(entry["id"]):
            page_id = entry["pages"].get(r.custom_id)
            if page_id is None:
                continue
            if r.result.type != "succeeded":
                log(f"  {page_id}: llm batch request {r.result.type}; keeping heuristic links")
                failed += 1
                continue
            data = parse_message(page_id, r.result.message)
            if data is None:
                failed += 1
                continue
            write_cache(work_dir / "llm" / f"{page_id}.json", r.custom_id, linker.model, data)
            got.append(page_id)
        for page_id in got:
            if _apply_cached(linker, work_dir, Path(entry["input_dir"]), page_id, debug, log,
                             keep_links=entry.get("keep_links", False)):
                updated += 1
            else:
                failed += 1
        entry["collected"] = True
        save_state(work_dir, state)
    return updated, failed


def _apply_cached(linker: LlmLinker, work_dir: Path, input_dir: Path, page_id: str,
                  debug: bool, log: Callable[[str], None] = print, *,
                  keep_links: bool = False) -> bool:
    from .pipeline import debug_path, page_json_path
    json_path = page_json_path(work_dir, page_id)
    try:
        page = Page.load(json_path)
    except FileNotFoundError:
        log(f"  {page_id}: page JSON is gone, answer kept in the llm cache")
        return False
    if page.status == "approved":
        log(f"  {page_id}: approved meanwhile, not changed")
        return False
    if page.edited and not keep_links:  # Claude's links would overwrite what was done in review
        log(f"  {page_id}: edited in review meanwhile, not changed (press L there to ask Claude)")
        return False
    img = load_bgr(input_dir / page.source)
    res, _ = linker.cached(page, img, work_dir / "llm")
    if res is None:  # page was re-detected after the request was sent
        log(f"  {page_id}: page changed since it was sent, llm answer not applied")
        return False
    apply(page, res, keep_links=keep_links)
    page.save(json_path)
    if debug:
        out = debug_path(work_dir, page_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), render_debug(page, img), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return True
