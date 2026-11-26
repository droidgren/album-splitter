"""Per-page detect pipeline: load -> detect -> link -> JSON + debug image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import json

import cv2

from . import __version__
from .debug import render_debug
from .detect import Params, detect_page
from .imageio import load_bgr
from .model import Page, page_id_from_path

if TYPE_CHECKING:
    from .batch import BatchQueue
    from .llm import LlmLinker

LlmStatus = Literal["off", "called", "cached", "failed", "queued"]


@dataclass
class Result:
    page_id: str
    skipped: bool
    page: Page | None = None
    skip_reason: str = ""  # "approved" or "edited" when skipped
    llm: LlmStatus = "off"


def page_json_path(work_dir: Path, page_id: str) -> Path:
    return work_dir / "pages" / f"{page_id}.json"


def debug_path(work_dir: Path, page_id: str) -> Path:
    return work_dir / "debug" / f"{page_id}.jpg"


def workdir_config_path(work_dir: Path) -> Path:
    return work_dir / "workdir.json"


def _load_config(work_dir: Path) -> dict:
    path = workdir_config_path(work_dir)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _update_config(work_dir: Path, **values: object) -> None:
    path = workdir_config_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        cfg = _load_config(work_dir)
    except (OSError, ValueError):
        cfg = {}
    cfg.update(values)
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def save_input_dir(work_dir: Path, input_dir: Path) -> None:
    """Remember where the scans are, so review and export can find the source images."""
    _update_config(work_dir, input_dir=str(input_dir))


def load_input_dir(work_dir: Path) -> Path | None:
    d = _load_config(work_dir).get("input_dir")
    return Path(d) if d else None


def save_export_dir(work_dir: Path, export_dir: Path) -> None:
    """Remember the last export folder chosen on the start page."""
    _update_config(work_dir, export_dir=str(export_dir))


def save_llm_model(work_dir: Path, model: str) -> None:
    """Remember the Claude model chosen on the start page for this project."""
    _update_config(work_dir, llm_model=model)


def load_llm_model(work_dir: Path) -> str | None:
    try:
        return _load_config(work_dir).get("llm_model") or None
    except (OSError, ValueError):
        return None


def save_llm_effort(work_dir: Path, effort: str | None) -> None:
    """Remember the Claude effort level chosen on the start page (None = the model's default)."""
    _update_config(work_dir, llm_effort=effort)


def load_llm_effort(work_dir: Path) -> str | None:
    try:
        return _load_config(work_dir).get("llm_effort") or None
    except (OSError, ValueError):
        return None


def save_guess_dates(work_dir: Path, on: bool) -> None:
    """Remember whether Claude should guess dates from the album order (`dates.py`)."""
    _update_config(work_dir, guess_dates=bool(on))


def load_guess_dates(work_dir: Path) -> bool:
    try:
        return bool(_load_config(work_dir).get("guess_dates"))
    except (OSError, ValueError):
        return False


def load_export_dir(work_dir: Path) -> Path | None:
    try:
        d = _load_config(work_dir).get("export_dir")
    except (OSError, ValueError):
        return None
    return Path(d) if d else None


def process_page(input_dir: Path, image_path: Path, work_dir: Path, *, force: bool = False,
                 debug: bool = True, params: Params | None = None,
                 page_id: str | None = None, llm: LlmLinker | None = None,
                 llm_queue: BatchQueue | None = None) -> Result:
    """Detect one page. With `llm`, Claude links and transcribes it right away; with
    `llm_queue` as well, the request is queued for the Batch API unless already cached."""
    rel = image_path.relative_to(input_dir)
    page_id = page_id or page_id_from_path(rel)
    json_path = page_json_path(work_dir, page_id)
    if json_path.exists() and not force:
        old = Page.load(json_path)
        # Keep reviewed work: approved pages and pages changed in review.
        if old.status == "approved" or old.edited:
            return Result(page_id, skipped=True,
                          skip_reason="approved" if old.status == "approved" else "edited")

    img = load_bgr(image_path)
    det = detect_page(img, params)
    page = Page(
        page_id=page_id,
        source=rel.as_posix(),
        width=img.shape[1],
        height=img.shape[0],
        photos=det.photos,
        texts=det.texts,
        links=det.links,
        page_quad=det.page_quad,
        status="proposed",
        detector_version=__version__,
    )
    llm_status: LlmStatus = "off"
    if llm is not None and llm_queue is not None:
        from .llm import apply
        res, req = llm.cached(page, img, work_dir / "llm")
        if res is None:
            llm_queue.add(req)
            llm_status = "queued"
        else:
            apply(page, res)
            llm_status = "cached"
    elif llm is not None:
        from .llm import apply
        res = llm.link(page, img, work_dir / "llm")
        if res is None:
            llm_status = "failed"
        else:
            apply(page, res)
            llm_status = "cached" if res.cached else "called"
    page.save(json_path)
    if debug:
        out = debug_path(work_dir, page_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), render_debug(page, img), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Result(page_id, skipped=False, page=page, llm=llm_status)


def page_geometry(page: Page) -> tuple:
    """What Claude was shown: the photo frames and text boxes."""
    return ([(p.id, [tuple(c) for c in p.corners]) for p in page.photos],
            [(t.id, tuple(t.box)) for t in page.texts])


ClaudeStatus = Literal["called", "cached", "queued", "failed", "skipped"]


@dataclass
class ClaudeResult:
    status: ClaudeStatus
    page: Page | None = None  # the page as saved (called/cached)


def claude_page(llm: LlmLinker, work_dir: Path, input_dir: Path, path: Path, *,
                queue: BatchQueue | None = None, page_lock=None, debug: bool = True,
                cache_dir: Path | None = None) -> ClaudeResult:
    """Ask Claude about an existing page as it is now; no detection. Frames, boxes and links
    stay (Claude only links texts without a link). With `queue`, a request that is not cached
    goes into the batch. A page approved or with frames/boxes changed meanwhile is left alone."""
    from contextlib import nullcontext

    from .llm import apply
    page = Page.load(path)
    img = load_bgr(input_dir / page.source)
    cache = cache_dir or work_dir / "llm"
    if queue is not None:
        res, req = llm.cached(page, img, cache)
        if res is None:
            queue.add(req)
            return ClaudeResult("queued", page)
    else:
        res = llm.link(page, img, cache)
        if res is None:
            return ClaudeResult("failed", page)
    with page_lock or nullcontext():
        now = Page.load(path)  # the page may have been edited in review meanwhile
        if page_geometry(now) != page_geometry(page) or now.status == "approved":
            return ClaudeResult("skipped", now)
        apply(now, res, keep_links=True)
        now.save(path)
    if debug:
        out = debug_path(work_dir, now.page_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), render_debug(now, img), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return ClaudeResult("cached" if res.cached else "called", now)


@dataclass
class StampsResult:
    asked: int = 0     # photos sent to Claude
    found: int = 0     # stamps Claude read a date from, now on the page
    unplaced: int = 0  # found, but not in a corner close-up: the photo is flagged instead
    failed: int = 0    # no usable answer
    page: Page | None = None


def stamps_page(llm: LlmLinker, work_dir: Path, input_dir: Path, path: Path, *,
                page_lock=None, debug: bool = True, cache_dir: Path | None = None
                ) -> StampsResult:
    """Let Claude look for the camera date stamp on every photo of the page that has no stamp
    date and no reviewer date. With no stamp box (the OCR found nothing), it gets the photo and
    full resolution close-ups of its corners (`llm.prepare_stamp`), and a stamp it reads becomes
    a stamp box over that corner. A box with no date (unreadable to the OCR and to the page
    request) gets a second, closer look on its own. Dated boxes are checked by the page request."""
    from contextlib import nullcontext

    from .llm import apply_stamp_answer, check_stamps_against_captions
    from .model import Issue
    res = StampsResult()
    page = Page.load(path)
    # No stamp box, or a box nobody could read a date from (OCR or the page request).
    todo = [ph for ph in page.photos
            if ph.date is None and (ph.stamp is None or not ph.stamp.date)]
    res.page = page
    if page.status == "approved" or not todo:
        return res
    img = load_bgr(input_dir / page.source)
    cache = cache_dir or work_dir / "llm"
    answers = {}
    for ph in todo:
        res.asked += 1
        ans = llm.read_stamp(page, ph, img, cache)
        if ans is None:
            res.failed += 1
        elif ans.found and ans.date:
            answers[ph.id] = ans
    if not answers:
        return res
    with page_lock or nullcontext():
        now = Page.load(path)  # it may have been edited in review meanwhile
        if now.status == "approved":
            return res
        photos = {p.id: p for p in now.photos}
        for pid, ans in answers.items():
            ph = photos.get(pid)
            old = next((p for p in page.photos if p.id == pid), None)
            if (ph is None or ph.date is not None or ph.corners != old.corners
                    or (ph.stamp is not None and ph.stamp.date)
                    or (ph.stamp is None) != (old.stamp is None)):
                continue
            if apply_stamp_answer(ph, ans):
                res.found += 1
            else:
                res.unplaced += 1
                ph.needs_review = True
                now.issues = [i for i in now.issues if not (i.source == "stamps"
                                                            and i.photo_ids == [pid])]
                now.issues.append(Issue(kind="other", photo_ids=[pid], source="stamps", note=(
                    f"P{pid}: Claude read the date stamp \"{ans.text}\" = {ans.date}, but not in "
                    "a corner: mark the stamp (S) and read it again")))
        check_stamps_against_captions(now)
        now.save(path)
    res.page = now
    if debug:
        out = debug_path(work_dir, now.page_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), render_debug(now, img), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return res
