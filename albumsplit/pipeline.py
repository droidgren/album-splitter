"""Per-page detect pipeline: load -> detect -> link -> JSON + debug image."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2

from . import __version__
from .debug import render_debug
from .detect import Params, detect_page
from .imageio import load_bgr
from .model import Page, page_id_from_path


@dataclass
class Result:
    page_id: str
    skipped: bool
    page: Page | None = None


def page_json_path(work_dir: Path, page_id: str) -> Path:
    return work_dir / "pages" / f"{page_id}.json"


def debug_path(work_dir: Path, page_id: str) -> Path:
    return work_dir / "debug" / f"{page_id}.jpg"


def process_page(input_dir: Path, image_path: Path, work_dir: Path, *, force: bool = False,
                 debug: bool = True, params: Params | None = None,
                 page_id: str | None = None) -> Result:
    rel = image_path.relative_to(input_dir)
    page_id = page_id or page_id_from_path(rel)
    json_path = page_json_path(work_dir, page_id)
    if json_path.exists() and not force and Page.load(json_path).status == "approved":
        return Result(page_id, skipped=True)

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
    page.save(json_path)
    if debug:
        out = debug_path(work_dir, page_id)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), render_debug(page, img), [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Result(page_id, skipped=False, page=page)
