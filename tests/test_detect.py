"""Detection on a synthetic page with known geometry."""

import cv2
import numpy as np
import pytest

from albumsplit.detect import detect_page

PAGE_W, PAGE_H = 2400, 3300
PAPER = (222, 232, 238)  # BGR, cream


def make_print(w: int, h: int, seed: int, light_corner: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    small = rng.integers(20, 200, size=(h // 40 + 1, w // 40 + 1, 3), dtype=np.uint8)
    img = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    img = cv2.GaussianBlur(img, (0, 0), 3)
    if light_corner:  # overcast sky in the top-left corner, almost paper-coloured
        img[: h // 3, : w // 2] = (232, 236, 240)
    return img


def make_page() -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    page = np.full((PAGE_H, PAGE_W, 3), PAPER, np.uint8)
    rects = [(120, 150, 1500, 1000), (700, 1300, 1550, 1050), (120, 2450, 1100, 760)]
    for i, (x, y, w, h) in enumerate(rects):
        page[y:y + h, x:x + w] = make_print(w, h, seed=i, light_corner=(i == 1))
        cv2.rectangle(page, (x, y), (x + w - 1, y + h - 1), (150, 150, 150), 2)  # print edge shadow
    # handwriting: blue ballpoint "cursive" (joined loops with ascenders/descenders) beside
    # photo 1 and left of photo 2. (OpenCV's fonts render as bold print, nothing like a pen.)
    rng = np.random.default_rng(7)
    for bx, by in [(1700, 330), (150, 1530)]:
        for line in range(3):
            x, base = bx, by + line * 95
            for word in range(3):
                for _ in range(int(rng.integers(3, 6))):
                    tall = rng.random() < 0.3
                    ry = 26 if tall else 13
                    cy = base - ry if not (rng.random() < 0.15) else base + 6
                    cv2.ellipse(page, (x + 11, cy), (11, ry), 0, 0, 360, (170, 60, 30), 3, cv2.LINE_AA)
                    cv2.line(page, (x + 22, base), (x + 30, base - 4), (170, 60, 30), 3, cv2.LINE_AA)
                    x += 28
                x += 30  # space between words
    # a sticker (solid star) and an arrow: neither is writing
    star = np.array([[2250 + 90 * np.cos(a) * (1 if k % 2 == 0 else 0.45),
                      2700 + 90 * np.sin(a) * (1 if k % 2 == 0 else 0.45)]
                     for k, a in enumerate(np.linspace(-np.pi / 2, 1.5 * np.pi, 10, endpoint=False))], np.int32)
    cv2.fillPoly(page, [star], (110, 110, 110))
    cv2.arrowedLine(page, (1400, 2900), (1800, 2900), (60, 60, 60), 5, cv2.LINE_AA, tipLength=0.08)
    return page, rects


@pytest.fixture(scope="module")
def detection():
    page, rects = make_page()
    return detect_page(page), rects


def test_finds_each_print(detection):
    det, rects = detection
    assert len(det.photos) == len(rects)
    for photo, (x, y, w, h) in zip(det.photos, rects):
        x0, y0, x1, y1 = photo.bbox()
        tol = 0.02 * max(PAGE_W, PAGE_H)
        assert abs(x0 - x) < tol and abs(y0 - y) < tol
        assert abs(x1 - (x + w)) < tol and abs(y1 - (y + h)) < tol


def test_light_corner_does_not_shrink_print(detection):
    det, rects = detection
    x, y, w, h = rects[1]
    x0, y0, _, _ = det.photos[1].bbox()
    assert abs(x0 - x) < 30 and abs(y0 - y) < 30


CAPTION_AREAS = [(1680, 260, 2300, 540), (130, 1460, 700, 1740)]  # x0, y0, x1, y1 (full res)


def test_each_caption_is_one_block_and_nothing_else_is_text(detection):
    det, _ = detection
    hits = [0, 0]
    for t in det.texts:
        cx, cy = t.center()
        inside = [x0 <= cx <= x1 and y0 <= cy <= y1 for x0, y0, x1, y1 in CAPTION_AREAS]
        assert any(inside), f"text block {t.box} is not a caption (sticker? arrow?)"
        hits[inside.index(True)] += 1
    assert hits == [1, 1]


def test_handwriting_links_to_nearest_photo(detection):
    from albumsplit.link import link_heuristic

    det, _ = detection
    linked = {lk.text_id: lk.photo_id for lk in link_heuristic(det.photos, det.texts)}
    for t in det.texts:
        expected = 1 if t.center()[1] < 1000 else 2  # beside photo 1 / left of photo 2
        assert linked[t.id] == expected


def test_handwriting_is_outside_photos(detection):
    det, _ = detection
    for t in det.texts:
        tx0, ty0, tx1, ty1 = t.bbox()
        for p in det.photos:
            px0, py0, px1, py1 = p.bbox()
            overlap = max(0, min(tx1, px1) - max(tx0, px0)) * max(0, min(ty1, py1) - max(ty0, py0))
            assert overlap < 0.05 * (tx1 - tx0) * (ty1 - ty0)


def test_text_margin_stops_at_photos_and_page_edge():
    from albumsplit.detect import _add_margin

    blocked = np.zeros((100, 200), np.uint8)
    blocked[:, 150:] = 255  # a photo on the right
    # free on all sides -> grows by 5; next to the photo -> stops at it; at the page edge -> clipped
    assert _add_margin([(50, 40, 20, 10)], blocked, 5) == [(45, 35, 30, 20)]
    assert _add_margin([(130, 40, 18, 10)], blocked, 5) == [(125, 35, 25, 20)]
    assert _add_margin([(2, 1, 10, 10)], blocked, 5) == [(0, 0, 17, 16)]


def _bbox(ph) -> tuple[float, float, float, float]:
    xs = [p[0] for p in ph.corners]
    ys = [p[1] for p in ph.corners]
    return min(xs), min(ys), max(xs), max(ys)


def test_touching_prints_split_at_hairline():
    """Two prints of the same width laid edge to edge, a hairline of paper between them
    (~7 px at full resolution, as on the samples)."""
    page = np.full((PAGE_H, PAGE_W, 3), PAPER, np.uint8)
    page[300:1300, 400:2000] = make_print(1600, 1000, seed=11)
    page[1307:2400, 400:2000] = make_print(1600, 1093, seed=12)
    det = detect_page(page)
    boxes = sorted(_bbox(ph) for ph in det.photos)
    assert len(boxes) == 2, boxes
    (_, a0, _, a1), (_, b0, _, b1) = sorted(boxes, key=lambda b: b[1])
    assert abs(a0 - 300) < 25 and abs(a1 - 1300) < 25
    assert abs(b0 - 1307) < 25 and abs(b1 - 2400) < 25


def test_print_cut_by_paper_white_band_stays_one():
    """Snow or a white blanket right across a print: the colour mask sees two pieces with
    paper colour between them; the print's frame says it is one."""
    page = np.full((PAGE_H, PAGE_W, 3), PAPER, np.uint8)
    img = make_print(1500, 1200, seed=13)
    xs = np.arange(1500)
    top = (450 + 40 * np.sin(xs / 90.0)).astype(int)  # snow has no straight outline
    bottom = (760 + 50 * np.sin(xs / 130.0 + 1.0)).astype(int)
    for x in xs:
        img[top[x]:bottom[x], x] = PAPER  # paper-coloured band right across the print
    page[400:1600, 400:1900] = img
    cv2.rectangle(page, (400, 400), (1899, 1599), (150, 150, 150), 2)  # print edge shadow
    det = detect_page(page)
    assert len(det.photos) == 1, [_bbox(ph) for ph in det.photos]
    x0, y0, x1, y1 = _bbox(det.photos[0])
    assert abs(x0 - 400) < 25 and abs(y0 - 400) < 25 and abs(x1 - 1900) < 25 and abs(y1 - 1600) < 25
