"""Keep derived geometry consistent after edits made in review."""

from __future__ import annotations

import cv2
import numpy as np

from ..detect import _text_outline
from ..model import Page, Photo, Point


def text_outline(box: tuple[int, int, int, int], photos: list[Photo]) -> list[Point] | None:
    """Paper part of a text box that runs onto photos, as detect computes it (else None)."""
    x, y, w, h = box
    if w < 2 or h < 2:
        return None
    mask = np.zeros((h, w), np.uint8)
    for ph in photos:
        quad = np.array([(px - x, py - y) for px, py in ph.corners], np.float32)
        cv2.fillPoly(mask, [np.round(quad).astype(np.int32)], 255)
    # Full resolution, so a little more tolerance than detect uses on its working copy.
    ol = _text_outline((0, 0, w, h), mask, epsilon=3.0)
    if ol is None:
        return None
    return [(round(float(px) + x, 1), round(float(py) + y, 1)) for px, py in ol]


def _bbox_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def normalize(new: Page, old: Page | None) -> Page:
    """Fix up a page that the review UI sent, relative to the stored version.

    - Links to deleted photos/texts and duplicate links are dropped.
    - A photo whose corners moved loses its occlusion outline (it would no longer fit) and
      its needs_review flag (a person has looked at it).
    - A text box that is new, was moved or resized, or overlaps a moved/new/deleted photo
      gets its paper outline recomputed.
    """
    photo_ids = {p.id for p in new.photos}
    text_ids = {t.id for t in new.texts}
    seen: set[tuple[int, int]] = set()
    links = []
    for lk in new.links:
        key = (lk.text_id, lk.photo_id)
        if lk.text_id in text_ids and lk.photo_id in photo_ids and key not in seen:
            seen.add(key)
            links.append(lk)
    new.links = links

    old_photos = {p.id: p for p in old.photos} if old else {}
    old_texts = {t.id: t for t in old.texts} if old else {}

    changed: list[tuple[float, float, float, float]] = []  # bboxes of photos that changed
    for ph in new.photos:
        prev = old_photos.get(ph.id)
        if prev is None or _moved(prev.corners, ph.corners):
            ph.outline = None
            ph.needs_review = False
            changed.append(ph.bbox())
            if prev is not None:
                changed.append(prev.bbox())
    for pid, prev in old_photos.items():
        if pid not in photo_ids:
            changed.append(prev.bbox())

    for t in new.texts:
        t.box = tuple(int(round(v)) for v in t.box)
        prev = old_texts.get(t.id)
        dirty = prev is None or tuple(prev.box) != t.box
        if not dirty and any(_bbox_overlap(t.bbox(), b) for b in changed):
            dirty = True
        if dirty:
            t.outline = text_outline(t.box, new.photos)
    return new


def _moved(a: list[Point], b: list[Point], tol: float = 0.5) -> bool:
    return len(a) != len(b) or any(abs(p[0] - q[0]) > tol or abs(p[1] - q[1]) > tol
                                   for p, q in zip(a, b))

