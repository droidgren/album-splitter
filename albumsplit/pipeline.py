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


def _update_config(work_dir: Path, **values: str) -> None:
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
