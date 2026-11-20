"""Export approved pages: one clean crop and one caption card per photo, plus index.csv."""

from __future__ import annotations

import csv
import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PIL import Image

from .exiftool import tag_args, write_tags
from .imageio import load_bgr
from .llm import quad_transform, text_crop_full
from .model import Page, Photo, effective_caption, effective_date

INDEX_FIELDS = ["file", "card", "source", "page_id", "photo_id", "caption", "date"]


@dataclass
class ExportReport:
    pages: int = 0
    photos: int = 0
    skipped: int = 0  # not approved (without include_proposed)
    failed: list[tuple[str, str]] = field(default_factory=list)  # (page_id, error)
    cancelled: bool = False


def photo_crop(img_bgr: np.ndarray, photo: Photo) -> np.ndarray:
    """Perspective-corrected full-resolution crop of a photo.

    The part of the print hidden under another print (outside `photo.outline`) is painted
    white, and `photo.rotation` (clockwise degrees) is applied.
    """
    M, (w, h) = quad_transform(photo.corners)
    crop = cv2.warpPerspective(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    if photo.outline:
        poly = cv2.perspectiveTransform(np.asarray(photo.outline, np.float32).reshape(-1, 1, 2), M)
        visible = np.zeros((h, w), np.uint8)
        cv2.fillPoly(visible, [np.round(poly).astype(np.int32)], 255)
        crop[visible == 0] = 255
    k = (photo.rotation // 90) % 4
    return np.ascontiguousarray(np.rot90(crop, -k)) if k else crop


def card(img_bgr: np.ndarray, page: Page, photo: Photo, crop: np.ndarray) -> np.ndarray:
    """The photo with its linked handwritten captions stacked underneath, on white.

    Each caption is cut from its own box (white outside its paper outline and over any
    print), never from a union box, so no neighbouring photo shows. Captions keep the page
    scale, so the handwriting is as large next to the photo as in the album, unless a
    caption is wider than the photo: then it is shrunk to the photo's width.
    """
    ph, pw = crop.shape[:2]
    margin = max(16, round(0.04 * max(pw, ph)))
    strips = []
    for t in page.linked_texts(photo.id):
        tc = whiten_paper(text_crop_full(t, img_bgr, page.photos))
        if tc.shape[1] > pw:
            f = pw / tc.shape[1]
            tc = cv2.resize(tc, (pw, max(1, round(tc.shape[0] * f))), interpolation=cv2.INTER_AREA)
        strips.append(tc)
    gap = margin // 2
    height = margin + ph + sum(margin if i == 0 else gap for i in range(len(strips)))
    height += sum(s.shape[0] for s in strips) + margin
    out = np.full((height, pw + 2 * margin, 3), 255, np.uint8)
    out[margin:margin + ph, margin:margin + pw] = crop
    y = margin + ph
    for i, s in enumerate(strips):
        y += margin if i == 0 else gap
        sh, sw = s.shape[:2]
        x = margin + (pw - sw) // 2
        out[y:y + sh, x:x + sw] = s
        y += sh
    return out


def whiten_paper(crop: np.ndarray, max_gain: float = 1.6) -> np.ndarray:
    """Scale each channel so the paper (the bright majority of a caption crop) becomes white,
    so the strip blends into the card instead of showing as a cream or grey patch."""
    paper_px = crop[(crop < 255).any(axis=2)]  # leave out what was already blanked out
    if len(paper_px) < 50:
        return crop
    paper = np.percentile(paper_px, 75, axis=0).clip(1, 255)
    gain = np.minimum(255.0 / paper, max_gain)
    return np.clip(crop.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def save_jpeg(path: Path, img_bgr: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.jpg")
    Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).save(
        tmp, "JPEG", quality=quality, subsampling=0, optimize=True)
    tmp.replace(path)


def _stale_outputs(out_dir: Path, page_id: str, n_photos: int) -> list[Path]:
    """Files of an earlier export of this page that had more photos than now."""
    pat = re.compile(rf"{re.escape(page_id)}_(\d+)(_card)?\.jpg")
    stale = []
    for p in out_dir.glob(f"{glob_escape(page_id)}_*.jpg"):
        m = pat.fullmatch(p.name)
        if m and int(m.group(1)) > n_photos:
            stale.append(p)
    return stale


def glob_escape(s: str) -> str:
    return re.sub(r"([\[\]*?])", r"[\1]", s)


def export_page(page: Page, img_bgr: np.ndarray, out_dir: Path, *, cards: bool = True,
                quality: int = 95, exiftool: list[str] | None = None) -> list[dict]:
    """Write one page's photos (and cards); returns their index.csv rows."""
    rows, jobs = [], []
    photos = sorted(page.photos, key=lambda p: p.id)
    for n, photo in enumerate(photos, 1):
        caption, date = effective_caption(page, photo), effective_date(page, photo)
        crop = photo_crop(img_bgr, photo)
        name = f"{page.page_id}_{n}.jpg"
        save_jpeg(out_dir / name, crop, quality)
        tags = tag_args(caption, date)
        jobs.append((out_dir / name, tags))
        card_name = ""
        if cards:
            card_name = f"{page.page_id}_{n}_card.jpg"
            save_jpeg(out_dir / card_name, card(img_bgr, page, photo, crop), quality)
            jobs.append((out_dir / card_name, tags))
        rows.append({"file": name, "card": card_name, "source": page.source,
                     "page_id": page.page_id, "photo_id": photo.id,
                     "caption": caption or "", "date": date or ""})
    if exiftool:
        write_tags(exiftool, jobs)
    for p in _stale_outputs(out_dir, page.page_id, len(photos)):
        p.unlink()
    return rows


def export_work_dir(work_dir: Path, out_dir: Path, input_dir: Path, *,
                    include_proposed: bool = False, only: str | None = None, cards: bool = True,
                    quality: int = 95, exiftool: list[str] | None = None,
                    progress: Callable[[str], None] | None = None,
                    on_total: Callable[[int], None] | None = None,
                    on_page: Callable[[str], None] | None = None,
                    stop: Callable[[], bool] | None = None) -> ExportReport:
    """Export every approved page of a work dir (all pages with `include_proposed`).

    `progress` gets one line per page, `on_total` the number of pages to export, `on_page`
    each page id as it starts. When `stop()` turns true the run ends after the current page
    (index.csv still lists what was exported).
    """
    report = ExportReport()
    rows: list[dict] = []
    say = progress or (lambda _msg: None)
    todo: list[Page] = []
    for path in sorted((work_dir / "pages").glob("*.json")):
        try:
            page = Page.load(path)
        except Exception as e:  # a broken JSON should not stop the run
            report.failed.append((path.stem, f"cannot read {path.name}: {e}"))
            say(f"{path.stem}: ERROR {e}")
            continue
        if only and not fnmatch.fnmatch(page.page_id, only):
            continue
        if page.status != "approved" and not include_proposed:
            report.skipped += 1
            continue
        todo.append(page)
    if on_total:
        on_total(len(todo))
    for page in todo:
        if stop and stop():
            report.cancelled = True
            say("Cancelled.")
            break
        if on_page:
            on_page(page.page_id)
        if not page.photos:
            report.pages += 1
            say(f"{page.page_id}: no photos")
            continue
        try:
            img = load_bgr(input_dir / page.source)
            if img.shape[:2] != (page.height, page.width):
                raise ValueError(f"scan is {img.shape[1]}x{img.shape[0]}, page JSON says "
                                 f"{page.width}x{page.height}")
            page_rows = export_page(page, img, out_dir, cards=cards, quality=quality,
                                    exiftool=exiftool)
        except Exception as e:
            report.failed.append((page.page_id, str(e)))
            say(f"{page.page_id}: ERROR {e}")
            continue
        rows += page_rows
        report.pages += 1
        report.photos += len(page_rows)
        say(f"{page.page_id}: {len(page_rows)} photos")
    if rows:
        write_index(out_dir / "index.csv", rows)
    return report


def write_index(path: Path, rows: list[dict]) -> None:
    """index.csv in UTF-8 with BOM, so Excel shows åäö."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        w.writeheader()
        w.writerows(rows)
