"""Photo and handwritten-text detection on scanned album pages.

All image processing happens on a downscaled working copy; returned geometry is in
full-resolution pixel coordinates of the (EXIF-oriented) source image.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import cv2
import numpy as np

from .imageio import downscale
from .link import link_heuristic
from .model import Link, Photo, Point, TextBlock
from .stamp import StampParams, find_stamp, harmonize


@dataclass
class Params:
    work_long_side: int = 1600
    # paper / foreground
    paper_dist: float = 14.0        # Lab distance (L down-weighted) above which a pixel is not paper
    paper_l_weight: float = 0.4     # weight of L difference vs chroma (tolerates uneven lighting)
    open_frac: float = 0.012        # opening kernel (fraction of long side) that erases handwriting
    snap_in_frac: float = 0.010     # how far a rect side may move inward when snapping
    snap_out_frac: float = 0.20     # ... and outward (also capped at 60 % of the rect size)
    snap_min_grad: float = 10.0     # median gradient along a side needed to accept an edge
    snap_out_gap: int = 4           # px beyond the edge where paper is expected
    snap_out_paper: float = 0.6     # fraction of paper-like pixels required just outside the edge
    snap_stop_paper: float = 0.92   # stop searching outward once a line is this much paper
    min_photo_frac: float = 0.012   # min photo area as fraction of page area
    good_rect: float = 0.86         # rectangularity of a clean print
    min_rect: float = 0.55          # below this, a component is not kept as a photo at all
    bite_paper: float = 0.5         # see _bite_paper: above this a blob may be several prints
    split_max_hull: float = 0.82    # split blobs less convex than this ...
    split_clear_bite: float = 0.9   # ... or whose missing area is at least this much clean paper
    max_aspect: float = 4.0
    frame_merge_reach: float = 0.12  # pieces this close (x long side) may be one framed print
    # text
    photo_margin_frac: float = 0.004
    ink_dark: float = 32.0          # darkness vs local background for any ink
    ink_dark_blue: float = 12.0     # darkness required if the pixel is also bluish
    ink_blue: float = 8.0           # b* below paper b* for "bluish"
    min_ink_px: int = 10
    dilate_x_frac: float = 0.020
    dilate_y_frac: float = 0.011
    min_block_ink: int = 120
    min_block_h_frac: float = 0.009
    solid_frac: float = 0.006       # shapes thicker than this (fraction of long side) are not pen strokes
    text_margin_frac: float = 0.0067  # margin around text boxes: ~2 mm on an A4-size page
                                      # (the scans carry no usable DPI, all say 72)
    # camera date stamps
    stamps: bool = True
    stamp: StampParams = field(default_factory=StampParams)


@dataclass
class Detection:
    photos: list[Photo] = field(default_factory=list)
    texts: list[TextBlock] = field(default_factory=list)
    page_quad: list[Point] | None = None
    links: list[Link] = field(default_factory=list)


# --------------------------------------------------------------------------- helpers

def order_corners(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = pts[:, 1] - pts[:, 0]
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)], pts[np.argmax(s)], pts[np.argmax(d)]],
                    dtype=np.float32)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, 255, thickness=cv2.FILLED)
    return out


def ellipse(k: int) -> np.ndarray:
    k = max(1, int(k))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def estimate_paper(lab: np.ndarray) -> np.ndarray:
    """Dominant colour of the page (Lab, float). Works for white, cream and grey paper."""
    px = lab[::3, ::3].reshape(-1, 3).astype(np.int32)
    px = px[px[:, 0] > 90]  # ignore dark surroundings / dark photo areas
    key = (px[:, 0] // 12) * 4096 + (px[:, 1] // 6) * 64 + px[:, 2] // 6
    best = np.bincount(key).argmax()
    return np.median(px[key == best], axis=0).astype(np.float32)


def flatten_lighting(lab: np.ndarray, paper: np.ndarray) -> np.ndarray:
    """Remove smooth lighting gradients (phone shots, curved pages) from L.

    Fits a quadratic surface to the L of paper-coloured pixels and subtracts it, so the paper
    has the same lightness everywhere. On flatbed scans the fit is ~flat and this is a no-op.
    """
    f = lab.astype(np.float32)
    h, w = f.shape[:2]
    step = max(1, max(h, w) // 200)
    s = f[::step, ::step]
    gy, gx = np.mgrid[0:h:step, 0:w:step].astype(np.float32)

    def design(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        x, y = x / w - 0.5, y / h - 0.5
        return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], axis=-1)

    chroma = np.hypot(s[..., 1] - paper[1], s[..., 2] - paper[2])
    sel = (chroma < 8) & (np.abs(s[..., 0] - paper[0]) < 45)
    coef = None
    for _ in range(4):
        if sel.sum() < 300:
            return f
        coef, *_ = np.linalg.lstsq(design(gx[sel], gy[sel]), s[..., 0][sel], rcond=None)
        res = s[..., 0] - design(gx, gy) @ coef
        sel = (chroma < 8) & (np.abs(res) < 12)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    field = design(xx, yy) @ coef
    f[..., 0] = np.clip(f[..., 0] - field + np.median(field[::step, ::step][sel]), 0, 255)
    return f


def paper_distance(lab: np.ndarray, paper: np.ndarray, l_weight: float) -> np.ndarray:
    f = lab.astype(np.float32) - paper
    return np.sqrt((f[..., 0] * l_weight) ** 2 + f[..., 1] ** 2 + f[..., 2] ** 2)


# --------------------------------------------------------------------------- photos

@dataclass
class _Region:
    contour: np.ndarray
    area: float
    rect: tuple  # cv2 RotatedRect
    rectangularity: float
    hull_rectangularity: float
    aspect: float
    touches_border: bool


def _region(contour: np.ndarray, shape: tuple[int, int]) -> _Region:
    h, w = shape
    area = cv2.contourArea(contour)
    rect = cv2.minAreaRect(contour)
    (rw, rh) = rect[1]
    rect_area = max(rw * rh, 1.0)
    x, y, bw, bh = cv2.boundingRect(contour)
    touches = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
    aspect = max(rw, rh) / max(min(rw, rh), 1.0)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    return _Region(contour, area, rect, area / rect_area, hull_area / rect_area, aspect, touches)


def _try_split(comp_mask: np.ndarray, min_area: float, p: Params, paper_strict: np.ndarray,
               grad: tuple[np.ndarray, np.ndarray]
               ) -> list[np.ndarray] | None:
    """Split a blob of touching prints. Returns piece masks or None."""
    # Only split when the evidence is clear: the blob is far from convex, or what it misses of
    # its bounding rectangle is almost all clean paper. In between (one print with a white sky
    # or wall at its edge looks like that too) keep it whole and flag it for review.
    bite = _bite_paper(comp_mask, paper_strict)
    cs, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    (rw, rh) = cv2.minAreaRect(c)[1]
    hull_rect = cv2.contourArea(cv2.convexHull(c)) / max(rw * rh, 1.0)
    if bite < p.bite_paper or (hull_rect >= p.split_max_hull and bite < p.split_clear_bite):
        return None
    long_side = max(comp_mask.shape)
    for frac in (0.006, 0.010, 0.015, 0.022, 0.030, 0.040):
        k = max(3, int(long_side * frac)) | 1
        er = cv2.erode(comp_mask, ellipse(k))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(er, connectivity=8)
        seeds = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area * 0.25]
        if len(seeds) < 2:
            continue
        pieces = []
        for i in seeds:
            piece = (labels == i).astype(np.uint8) * 255
            piece = cv2.dilate(piece, ellipse(k)) & comp_mask
            pieces.append(piece)
        rects = []
        for piece in pieces:
            cs, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            c = max(cs, key=cv2.contourArea)
            rects.append(_region(c, comp_mask.shape).rectangularity)
        areas = [cv2.countNonZero(pc) for pc in pieces]
        if min(areas) < 0.2 * sum(areas):
            continue  # a sliver, e.g. the part of one print beyond a light sky
        if np.mean(rects) < p.good_rect - 0.04:
            continue
        # Each piece must be a print in its own right: bordered by edges, not cut out of
        # one photo along an arbitrary line through its content.
        if all(_framed_sides(_piece_rect(pc), grad, p) >= 3 for pc in pieces):
            return pieces
    return _cut_split(comp_mask, min_area, p, paper_strict, grad)


def _piece_rect(mask: np.ndarray) -> tuple:
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.minAreaRect(max(cs, key=cv2.contourArea))


def _mask_rectangularity(mask: np.ndarray, covered: np.ndarray | None = None) -> float:
    """Area / bounding-rect area; parts of the rect under `covered` count as filled."""
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return 0.0
    c = max(cs, key=cv2.contourArea)
    rect = cv2.minAreaRect(c)
    (rw, rh) = rect[1]
    area = cv2.contourArea(c)
    if covered is not None:
        box = np.zeros_like(mask)
        cv2.fillPoly(box, [cv2.boxPoints(rect).astype(np.int32)], 255)
        area += cv2.countNonZero(box & covered & ~mask)
    return area / max(rw * rh, 1.0)


def _cut_split(comp_mask: np.ndarray, min_area: float, p: Params, paper_strict: np.ndarray,
               grad: tuple[np.ndarray, np.ndarray], depth: int = 0) -> list[np.ndarray] | None:
    """Split prints that touch along a full seam (no neck for erosion to find).

    Such a blob is a union of rectangles, so its outline has concave corners. A straight
    cut through one of its polygon vertices separates the prints; the cut that makes the
    pieces most rectangular wins, and pieces are split further if still needed.
    A cut must run along a seam: a straight edge along its whole length inside the blob.
    Touching prints always show one; a cut through the trees of a single photo does not.
    Prints are assumed to be roughly axis-aligned (a few degrees of rotation is fine).
    """
    if depth > 3:
        return None
    long_side = max(comp_mask.shape)
    cs, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    poly = cv2.approxPolyDP(c, 0.008 * long_side, True).reshape(-1, 2)
    x0, y0, bw, bh = cv2.boundingRect(c)
    parent = _mask_rectangularity(comp_mask)

    def pieces_of(half: np.ndarray) -> list[np.ndarray]:
        # Opening drops thin slivers a cut leaves along a neighbouring print's edge.
        half = cv2.morphologyEx(half, cv2.MORPH_OPEN, ellipse(9))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(half, connectivity=8)
        return [(labels == i).astype(np.uint8) * 255 for i in range(1, n)
                if stats[i, cv2.CC_STAT_AREA] >= min_area * 0.5]

    best: tuple[float, list[np.ndarray]] | None = None
    tried: set[tuple[str, int]] = set()
    for vx, vy in poly:
        for axis, pos in (("x", int(vx)), ("y", int(vy))):
            lo, hi = (x0, x0 + bw) if axis == "x" else (y0, y0 + bh)
            if pos - lo < 0.05 * long_side or hi - pos < 0.05 * long_side:
                continue  # cut too close to the blob's outer edge
            key = (axis, pos // 4)
            if key in tried:
                continue
            tried.add(key)
            pos, seam = _best_seam(comp_mask, grad, axis, pos, max(4, int(0.008 * long_side)))
            if seam < p.snap_min_grad:
                continue  # no straight edge here: not a boundary between prints
            a, b = comp_mask.copy(), comp_mask.copy()
            if axis == "x":
                a[:, pos:] = 0
                b[:, :pos] = 0
            else:
                a[pos:, :] = 0
                b[:pos, :] = 0
            pieces = pieces_of(a) + pieces_of(b)
            if len(pieces) < 2:
                continue
            areas = np.array([cv2.countNonZero(pc) for pc in pieces], np.float32)
            total = max(cv2.countNonZero(comp_mask), 1)
            if depth == 0 and areas.min() < 0.2 * total:
                continue  # slicing off a sliver: more likely a light area of one print
                # (deeper cuts may: they trim strips of a neighbour, re-merged by _remerge)
            covered = areas.sum() / total
            rects = np.array([_mask_rectangularity(pc) for pc in pieces])
            score = float((rects * areas).sum() / areas.sum()) * covered
            if best is None or score > best[0]:
                best = (score, pieces)
    if best is None or best[0] < parent + (0.08 if depth == 0 else 0.03):
        return None
    out: list[np.ndarray] = []
    for pc in best[1]:
        sub = None
        # Already known to be several prints: no bite check here (a piece's bites can be
        # covered by the neighbouring print, which is not paper). A piece that is nearly
        # rectangular but not framed by edges still carries a strip of a neighbour.
        rect = _mask_rectangularity(pc)
        if rect < p.good_rect or (rect < 0.98 and _framed_sides(_piece_rect(pc), grad, p) < 4):
            sub = _cut_split(pc, min_area, p, paper_strict, grad, depth + 1)
        out.extend(sub or [pc])
    if depth > 0:
        return out
    out = _remerge(out, p)
    # Every resulting piece must be a print in its own right, bordered by edges. A seam test
    # alone is fooled by straight lines inside one photo (a roofline, a treeline).
    for i, pc in enumerate(out):
        others = np.zeros_like(pc)
        for j, o in enumerate(out):
            if j != i:
                others |= cv2.dilate(o, ellipse(7))
        if _framed_sides(_piece_rect(pc), grad, p, covered=others) < 3:
            return None
    return out


def _best_seam(mask: np.ndarray, grad: tuple[np.ndarray, np.ndarray], axis: str, pos: int,
               search: int) -> tuple[int, float]:
    """Strongest straight edge near `pos` across the blob. Returns (position, strength).

    Strength is the 30th percentile of the gradient across the cut line over the part of
    the line inside the blob, so the edge must be there along most of the cut.
    """
    gx, gy = grad
    h, w = mask.shape
    best = (pos, 0.0)
    for q in range(pos - search, pos + search + 1):
        if not 1 <= q < (w if axis == "x" else h) - 1:
            continue
        if axis == "x":
            idx = np.nonzero(mask[:, q])[0]
            g = np.abs(gx[idx][:, q - 1:q + 2]).max(axis=1) if len(idx) else idx
        else:
            idx = np.nonzero(mask[q, :])[0]
            g = np.abs(gy[q - 1:q + 2][:, idx]).max(axis=0) if len(idx) else idx
        if len(idx) < 20:
            continue
        strength = float(np.percentile(g, 30))
        if strength > best[1]:
            best = (q, strength)
    return best


def _split_at_paper_lines(comp: np.ndarray, paperlike: np.ndarray,
                          grad: tuple[np.ndarray, np.ndarray], p: Params, min_area: float,
                          depth: int = 0) -> list[np.ndarray] | None:
    """Split prints that touch along a full side, with only a hairline of paper between them.

    Two prints of the same width laid edge to edge make one perfect rectangle in the colour
    mask (the opening closes the hairline). The hairline is still there: a row (or column) of
    paper colour right across the blob, with straight print edges along it. A horizon or a
    table edge inside one print is a step between two colours, not a line of paper.
    Prints are assumed to be roughly axis-aligned. Returns the pieces (each a clean
    rectangle), or None.
    """
    if depth > 3:
        return None
    ys, xs = np.nonzero(comp)
    if ys.size == 0:
        return None
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    across_y = np.abs(grad[1]) - np.abs(grad[0])
    best: tuple[float, str, int] | None = None
    for axis in ("y", "x"):
        lo, hi = (y0, y1) if axis == "y" else (x0, x1)
        span = hi - lo
        m = 0.08 * span  # not at the blob's own ends
        for q in range(int(lo + m), int(hi - m)):
            line = comp[q, x0:x1] if axis == "y" else comp[y0:y1, q]
            inside = np.nonzero(line)[0]
            if inside.size < 0.6 * ((x1 - x0) if axis == "y" else (y1 - y0)):
                continue
            # (+-4 px: the hairline between two slightly rotated prints is a little tilted)
            if axis == "y":
                pl = paperlike[max(0, q - 4):q + 5, x0:x1].max(axis=0)[inside]
            else:
                pl = paperlike[y0:y1, max(0, q - 4):q + 5].max(axis=1)[inside]
            frac = float(pl.mean())
            if frac < 0.8:
                continue
            # straight edges just beside the line (either side)
            strength = 0.0
            for o in (-4, -3, -2, -1, 1, 2, 3, 4):
                r = q + o
                if axis == "y" and 0 <= r < comp.shape[0]:
                    g = across_y[r, x0:x1][inside]
                elif axis == "x" and 0 <= r < comp.shape[1]:
                    g = -across_y[y0:y1, r][inside]
                else:
                    continue
                strength = max(strength, float(np.percentile(g, 30)))
            if strength >= 2 * p.snap_min_grad and (best is None or frac > best[0]):
                best = (frac, axis, q)
    if best is None:
        return None
    _, axis, q = best
    a, b = comp.copy(), comp.copy()
    if axis == "y":
        a[q:, :] = 0
        b[:q + 1, :] = 0
    else:
        a[:, q:] = 0
        b[:, :q + 1] = 0
    pieces = []
    for half in (a, b):
        if cv2.countNonZero(half) < 0.5 * min_area:
            return None
        sub = _split_at_paper_lines(half, paperlike, grad, p, min_area, depth + 1)
        pieces.extend(sub or [half])
    # Only a clean cut into whole prints is taken here; blobs of overlapping prints that need
    # more than straight cuts are left to _try_split.
    if any(_mask_rectangularity(pc) < p.good_rect for pc in pieces):
        return None
    return pieces


def _bite_paper(mask: np.ndarray, paper_strict: np.ndarray) -> float:
    """Fraction of clean paper in the part of the blob's bounding rectangle the blob misses.

    High for several prints touching (the gaps around them are paper), low for one print
    whose light areas (sky, white wall) were missed by the colour mask.
    """
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cs, key=cv2.contourArea)
    box = np.zeros_like(mask)
    cv2.fillPoly(box, [cv2.boxPoints(cv2.minAreaRect(c)).astype(np.int32)], 255)
    bite = (box > 0) & (mask == 0)
    if bite.sum() < 50:
        return 0.0
    return float(paper_strict[bite].mean())


def _remerge(pieces: list[np.ndarray], p: Params) -> list[np.ndarray]:
    """Undo over-cutting: a cut through one print's corner also slices its neighbour.

    Two pieces are re-joined when together they form a rectangle -- counting the part of
    that rectangle covered by another piece, since a print lying partly under another one
    is visible as an L-shape.
    """
    pieces = list(pieces)
    k = ellipse(5)
    merged = True
    while merged:
        merged = False
        for i in range(len(pieces)):
            for j in range(i + 1, len(pieces)):
                if not cv2.countNonZero(cv2.dilate(pieces[i], k) & pieces[j]):
                    continue
                union = pieces[i] | pieces[j]
                others = np.zeros_like(union)
                for m, pc in enumerate(pieces):
                    if m not in (i, j):
                        others |= pc
                if _mask_rectangularity(union, others) >= p.good_rect + 0.04:
                    pieces[i] = union
                    del pieces[j]
                    merged = True
                    break
            if merged:
                break
    return pieces


def _foreground(dist: np.ndarray, p: Params) -> np.ndarray:
    """Solid non-paper regions. Opening first erases handwriting strokes before anything is merged."""
    long_side = max(dist.shape[:2])
    fg = (dist > p.paper_dist).astype(np.uint8) * 255
    fg = cv2.medianBlur(fg, 3)
    k = max(3, int(long_side * p.open_frac)) | 1
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, ellipse(k))
    return fill_holes(fg)


def _page_hull(paper_strict: np.ndarray) -> np.ndarray:
    """Convex hull of all sizeable paper areas: the page itself.

    On flatbed scans paper shows up all around the prints, so this is ~the whole image. On a
    photographed album it excludes the table, neighbouring pages and the binding.
    """
    m = cv2.morphologyEx((paper_strict > 0).astype(np.uint8) * 255, cv2.MORPH_OPEN, ellipse(5))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    hull = np.zeros_like(m)
    big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 0.002 * m.size]
    if not big:
        hull[:] = 255
        return hull
    pts = cv2.findNonZero(np.isin(labels, big).astype(np.uint8))
    cv2.fillConvexPoly(hull, cv2.convexHull(pts), 255)
    return hull


def _merge_fragments(fg: np.ndarray, paper_strict: np.ndarray, p: Params) -> np.ndarray:
    """Join pieces of one print that the colour mask split apart (white walls, skies, clouds).

    Two pieces are joined when together they form a rectangle and what lies between them is
    not clean paper -- the gap between two separate prints always is.
    """
    h, w = fg.shape
    g = int(max(h, w) * 0.03)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    parts = [c for c in contours if cv2.contourArea(c) >= 0.003 * h * w]
    parts.sort(key=cv2.contourArea, reverse=True)
    out = fg.copy()
    changed = True
    while changed:
        changed = False
        for i in range(len(parts)):
            xi, yi, wi, hi = cv2.boundingRect(parts[i])
            for j in range(i + 1, len(parts)):
                xj, yj, wj, hj = cv2.boundingRect(parts[j])
                if xj > xi + wi + g or xi > xj + wj + g or yj > yi + hi + g or yi > yj + hj + g:
                    continue
                union = np.vstack([parts[i], parts[j]])
                hull = cv2.convexHull(union)
                (rw, rh) = cv2.minAreaRect(union)[1]
                if (cv2.contourArea(hull) / max(rw * rh, 1) < 0.9
                        or max(rw, rh) / max(min(rw, rh), 1) > p.max_aspect):
                    continue
                hull_m = np.zeros((h, w), np.uint8)
                cv2.fillConvexPoly(hull_m, hull, 255)
                mi = np.zeros((h, w), np.uint8)
                mj = np.zeros((h, w), np.uint8)
                cv2.drawContours(mi, [parts[i]], -1, 255, cv2.FILLED)
                cv2.drawContours(mj, [parts[j]], -1, 255, cv2.FILLED)
                # Only the corridor close to both pieces counts; light bites along a single
                # photo's own edge (sky, white wall) say nothing about what separates them.
                near = cv2.dilate(mi, ellipse(2 * g + 1)) & cv2.dilate(mj, ellipse(2 * g + 1))
                gap = (hull_m > 0) & (near > 0) & (mi == 0) & (mj == 0)
                if gap.sum() == 0 or paper_strict[gap].mean() > 0.35:
                    continue  # touching, or clean paper between them: two prints
                out |= hull_m
                parts[i] = hull
                del parts[j]
                changed = True
                break
            if changed:
                break
    return out


def _merge_framed(fg: np.ndarray, grad: tuple[np.ndarray, np.ndarray], paperlike: np.ndarray,
                  paper_strict: np.ndarray, p: Params) -> np.ndarray:
    """Join pieces of one print separated by paper-coloured content (snow, a white blanket).

    _merge_fragments leaves these apart: clean paper colour between pieces is also what
    separates two prints. What tells one print apart is its frame. A piece is tried together
    with its 1, 2, 3 nearest neighbours (a print can fall apart into several pieces around a
    white area, and only all of them together span it); the group is one print if the
    rectangle around it, snapped to the edges, is framed by print edges on all four sides and
    no straight seam runs across it (`_frames_one_print`). The rectangle is filled in whole.
    """
    h, w = fg.shape
    reach = p.frame_merge_reach * max(h, w)
    out = fg.copy()
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    parts = [c for c in contours if cv2.contourArea(c) >= 0.003 * h * w]
    # A piece that is already a whole print (rectangular, framed all round) is no fragment:
    # inside a bigger print its own border would be a seam. Leaving those out saves most of
    # the (costly) tries.
    def whole(c: np.ndarray) -> bool:
        r = _region(c, (h, w))
        return r.rectangularity >= p.good_rect and _framed_sides(r.rect, grad, p) == 4
    parts = [c for c in parts if not whole(c)]
    parts.sort(key=cv2.contourArea, reverse=True)

    def gap_between(a: tuple, b: tuple) -> float:
        dx = max(0, max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2]))
        dy = max(0, max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3]))
        return float(np.hypot(dx, dy))

    tried: set[frozenset] = set()
    changed = True
    while changed:
        changed = False
        boxes = [cv2.boundingRect(c) for c in parts]
        for i in range(len(parts)):
            near = sorted((gap_between(boxes[i], boxes[j]), j) for j in range(len(parts))
                          if j != i and gap_between(boxes[i], boxes[j]) <= reach)[:4]
            groups = [[i, *combo] for k in (1, 2, 3)
                      for combo in itertools.combinations([j for _, j in near], k)]
            for group in groups:
                key = frozenset(cv2.boundingRect(parts[m]) for m in group)
                if key in tried:
                    continue
                tried.add(key)
                union = np.vstack([parts[m] for m in group])
                (rw, rh) = cv2.minAreaRect(union)[1]
                if max(rw, rh) / max(min(rw, rh), 1) > p.max_aspect:
                    continue
                pieces = np.zeros((h, w), np.uint8)
                cv2.drawContours(pieces, [parts[m] for m in group], -1, 255, cv2.FILLED)
                rect = _frames_one_print(union, pieces, grad, paperlike, paper_strict, p)
                if rect is None:
                    continue
                box = cv2.boxPoints(rect).astype(np.int32)
                cv2.fillPoly(out, [box], 255)
                parts = [c for m, c in enumerate(parts) if m not in group] + [box.reshape(-1, 1, 2)]
                changed = True
                break
            if changed:
                break
    return out


def _frames_one_print(union: np.ndarray, pieces: np.ndarray, grad: tuple[np.ndarray, np.ndarray],
                      paperlike: np.ndarray, paper_strict: np.ndarray, p: Params) -> tuple | None:
    """The print rectangle around a group of pieces, if they are one print (else None).

    One print: its rectangle, snapped to the edges, shows a print edge along all four sides.
    Two prints with a narrow paper gap between them can pass that too (the gap is a small part
    of each side), but each shows its own straight edge along the gap: a seam running right
    across the rectangle, between the pieces. (The seam test is cheap, the snapping is not:
    it runs first, on the unsnapped rectangle, too.)
    """
    rect = cv2.minAreaRect(union)
    if _seam_across(rect, pieces, grad, p):
        return None
    rect, unconfirmed = _snap_rect(rect, grad, paperlike, paper_strict, p)
    (rw, rh) = rect[1]
    if unconfirmed or min(rw, rh) < 1 or max(rw, rh) / min(rw, rh) > p.max_aspect:
        return None
    if _framed_sides(rect, grad, p) < 4 or _seam_across(rect, pieces, grad, p):
        return None
    return rect


def _seam_across(rect: tuple, pieces: np.ndarray, grad: tuple[np.ndarray, np.ndarray], p: Params
                 ) -> bool:
    """True if a straight edge runs right across the rect where the pieces leave it open.

    Stays clear of the rect's own sides (a print's border texture runs along them). A seam
    between two prints is a straight edge: strong, and the gradient runs across it, not along.
    Busy content (foliage, a treeline) has strong gradients in every direction.
    """
    gx, gy = grad
    h, w = gx.shape
    bx = cv2.boxPoints(rect)
    x0, x1 = int(max(1, bx[:, 0].min() + 4)), int(min(w - 2, bx[:, 0].max() - 4))
    y0, y1 = int(max(1, bx[:, 1].min() + 4)), int(min(h - 2, bx[:, 1].max() - 4))
    if x1 - x0 < 20 or y1 - y0 < 20:
        return False
    inside = np.zeros((h, w), np.uint8)
    cv2.fillPoly(inside, [bx.astype(np.int32)], 255)
    gap = (inside[y0:y1, x0:x1] > 0) & (pieces[y0:y1, x0:x1] == 0)
    if not gap.any():
        return False
    my, mx = max(6, int(0.04 * (y1 - y0))), max(6, int(0.04 * (x1 - x0)))
    strong = 2 * p.snap_min_grad
    across = np.abs(gy[y0 - 1:y1 + 1, x0:x1]) - np.abs(gx[y0 - 1:y1 + 1, x0:x1])
    # horizontal seams (+-1 row: slightly rotated prints), on rows the pieces leave open
    rows = np.nonzero(gap.any(axis=1))[0]
    rows = rows[(rows >= my) & (rows <= y1 - y0 - my)]
    if rows.size:
        g = np.maximum(np.maximum(across[rows], across[rows + 1]), across[rows + 2])
        if (np.percentile(g, 30, axis=1) >= strong).any():
            return True
    across = np.abs(gx[y0:y1, x0 - 1:x1 + 1]) - np.abs(gy[y0:y1, x0 - 1:x1 + 1])
    cols = np.nonzero(gap.any(axis=0))[0]
    cols = cols[(cols >= mx) & (cols <= x1 - x0 - mx)]
    if cols.size:
        g = np.maximum(np.maximum(across[:, cols], across[:, cols + 1]), across[:, cols + 2])
        if (np.percentile(g, 30, axis=0) >= strong).any():
            return True
    return False


def _edge_strength(lab: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Gradient (gx, gy) of the strongest Lab channel per pixel, chroma boosted."""
    f = cv2.GaussianBlur(lab, (0, 0), 1.2).astype(np.float32)
    gxs, gys = [], []
    for ch, wgt in ((0, 1.0), (1, 2.0), (2, 2.0)):
        gxs.append(cv2.Sobel(f[..., ch], cv2.CV_32F, 1, 0, ksize=3) * wgt)
        gys.append(cv2.Sobel(f[..., ch], cv2.CV_32F, 0, 1, ksize=3) * wgt)
    gx, gy = np.stack(gxs), np.stack(gys)
    pick = np.argmax(gx * gx + gy * gy, axis=0)[None]
    return np.take_along_axis(gx, pick, 0)[0], np.take_along_axis(gy, pick, 0)[0]


def _snap_rect(rect: tuple, grad: tuple[np.ndarray, np.ndarray], paperlike: np.ndarray,
               paper_strict: np.ndarray, p: Params, covered: np.ndarray | None = None,
               lab: np.ndarray | None = None) -> tuple[tuple, int]:
    """Move each side of a rotated rect onto the print edge. Returns (rect, unconfirmed_sides).

    A print edge is a straight line with a strong gradient along most of its length and paper
    just outside it. Integrating along the whole side makes this robust to light photo areas
    (sky, white tablecloths) that the colour mask misses; handwriting next to a photo never
    forms a long straight edge. The outward search stops as soon as it reaches pure paper, so it
    cannot jump across a gap onto a neighbouring print.
    """
    gx, gy = grad
    h, w = gx.shape
    long_side = max(h, w)
    (cx, cy), (rw, rh), ang = rect
    t = np.deg2rad(ang)
    u = np.array([np.cos(t), np.sin(t)], np.float32)   # along rw
    v = np.array([-np.sin(t), np.cos(t)], np.float32)  # along rh
    c = np.array([cx, cy], np.float32)
    d_in = max(3, int(long_side * p.snap_in_frac))
    bounds = {"u-": -rw / 2, "u+": rw / 2, "v-": -rh / 2, "v+": rh / 2}

    def line(axis: str, pos: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n, along, length = (u, v, rh) if axis == "u" else (v, u, rw)
        ts = np.linspace(-0.42, 0.42, 150) * length
        pts = c + n * pos + ts[:, None] * along
        xs, ys = pts[:, 0], pts[:, 1]
        ok = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
        xi, yi = xs[ok].astype(np.int32), ys[ok].astype(np.int32)
        if covered is not None:  # stretches under a neighbouring print say nothing about this one
            free = covered[yi, xi] == 0
            xi, yi = xi[free], yi[free]
        return xi, yi, n

    unconfirmed = 0
    new = dict(bounds)
    for key, pos in bounds.items():
        axis, sign = key[0], (1 if key[1] == "+" else -1)
        perp = rw if axis == "u" else rh
        d_out = int(min(perp * 0.6, long_side * p.snap_out_frac))
        cands: list[tuple[int, float]] = []
        for o in range(-d_in, d_out + 1):
            xi, yi, n = line(axis, pos + sign * o)
            if len(xi) < 40:
                break  # left the image: the print touches the border here
            on_paper = paper_strict[yi, xi].mean()
            if o > 0 and on_paper >= p.snap_stop_paper:
                break  # in open paper beyond the print
            edge = float(np.median(np.abs(gx[yi, xi] * n[0] + gy[yi, xi] * n[1])))
            xo, yo, _ = line(axis, pos + sign * (o + p.snap_out_gap))
            if len(xo) < 40:
                continue
            out_paper = paperlike[yo, xo].mean()
            if edge >= p.snap_min_grad and out_paper >= p.snap_out_paper:
                # Real print edges have clean paper outside; edges inside a photo next to a
                # sky or white wall only have paper-ish colours there.
                cands.append((o, edge * (0.3 + paper_strict[yo, xo].mean())))
        if cands:
            s_max = max(e for _, e in cands)
            best_o = max(cands, key=lambda oe: oe[1] - 0.15 * s_max * abs(oe[0]) / max(d_out, 1))[0]
            new[key] = pos + sign * best_o
        else:
            xi, _, _ = line(axis, pos + sign * (d_in + 2))
            unconfirmed += 0 if len(xi) < 40 else 1  # sides on the image border count as fine

    # Second pass: a pale sky or white wall at a print's edge is paper-coloured, so the search
    # above stops early. Push each side outward (by at most ~6 % of the page) while either
    #  - both neighbouring sides still show their edge, across clean paper-coloured area, or
    #  - one side edge continues and the line is still tinted: its median colour (not
    #    brightness) clearly differs from the paper just beyond it (10-20 px further out). Medians over whole lines ignore paper grain (phone
    #    shots) and thin pen strokes (captions); comparing with the nearby paper rather than a
    #    page-wide paper colour copes with paper whose shade varies across the page.
    axes = {"u": (u, v), "v": (v, u)}  # side axis -> (its normal, the direction along it)
    for key in ("u-", "u+", "v-", "v+"):
        axis, sign = key[0], (1 if key[1] == "+" else -1)
        n_side, along = axes[axis]
        other = "v" if axis == "u" else "u"
        adj = (new[other + "-"], new[other + "+"])  # positions of the two adjacent sides
        perp = (new["u+"] - new["u-"]) if axis == "u" else (new["v+"] - new["v-"])
        # A long pale sky can run past one step's reach: keep going while the evidence
        # (side edges or tint) lasts right up to the end of the step.
        for _ in range(3):
            step = int(min(perp * 0.6, long_side * 0.06))
            grown = _push_side(new, key, step, axes, line, c, grad, paperlike, lab, p)
            new[key] += sign * grown
            if grown < step - 6:
                break

    nu0, nu1, nv0, nv1 = new["u-"], new["u+"], new["v-"], new["v+"]
    nc = c + u * (nu0 + nu1) / 2 + v * (nv0 + nv1) / 2
    return ((float(nc[0]), float(nc[1])), (float(nu1 - nu0), float(nv1 - nv0)), ang), unconfirmed


def _push_side(new: dict, key: str, step: int, axes: dict, line, c: np.ndarray,
               grad: tuple[np.ndarray, np.ndarray], paperlike: np.ndarray,
               lab: np.ndarray | None, p: Params) -> int:
    """How far (px, up to `step`) one side of a snapped rect may move outward across
    paper-coloured print content. See the second pass in _snap_rect."""
    gx, gy = grad
    h, w = gx.shape
    axis, sign = key[0], (1 if key[1] == "+" else -1)
    n_side, along = axes[axis]
    other = "v" if axis == "u" else "u"
    adj = (new[other + "-"], new[other + "+"])  # positions of the two adjacent sides
    last_ok, miss = 0, 0
    for o in range(1, step + 1):
        pos = new[key] + sign * o
        xi, yi, _ = line(axis, pos)
        if len(xi) < 40:
            break
        on_paper = paperlike[yi, xi].mean()
        if on_paper < 0.5:
            break  # into another print
        edges = []
        for a in adj:
            # A real edge, not grain: the median over a 9 px stretch of the side line
            # (allowing it to sit 2 px either way) must be strong.
            best = 0.0
            for d in range(-2, 3):
                vals = []
                for k in range(-4, 5):
                    q = c + n_side * (pos + k) + along * (a + d)
                    qx, qy = int(round(q[0])), int(round(q[1]))
                    if 0 <= qx < w and 0 <= qy < h:
                        vals.append(abs(gx[qy, qx] * along[0] + gy[qy, qx] * along[1]))
                if vals:
                    best = max(best, float(np.median(vals)))
            edges.append(best >= 1.5 * p.snap_min_grad
                         and _paper_beyond(c, n_side, along, pos, a, paperlike))
        tinted = False
        if lab is not None and any(edges):
            bx, by = [], []
            for k in range(10, 21, 2):
                qx, qy, _ = line(axis, pos + sign * k)
                bx.append(qx)
                by.append(qy)
            bx, by = np.concatenate(bx), np.concatenate(by)
            if len(bx) >= 40 and paperlike[by, bx].mean() >= 0.8:  # paper out there
                # Colour only: paper brightness drifts near prints (shading, phone shots).
                diff = np.median(lab[yi, xi, 1:], axis=0) - np.median(lab[by, bx, 1:], axis=0)
                tinted = float(np.linalg.norm(diff)) > 2.5
        if not tinted and on_paper < 0.9 and not all(edges):
            break  # writing or other marks, and the print's side edges have ended
        seen = tinted or all(edges)
        if seen:
            last_ok, miss = o, 0
        else:
            miss += 1
            if miss > 6:
                break  # the side edges have ended: this is the print's border
    return last_ok if last_ok >= 4 else 0


def _paper_beyond(c: np.ndarray, n_side: np.ndarray, along: np.ndarray, pos: float, a: float,
                  paperlike: np.ndarray) -> bool:
    """Is there paper just outside the side line at `a`, around position `pos` of the pushed
    side? A print's own edge has paper beyond it. A neighbouring print lying flush against
    this one continues the side line with *its* edge, and has print content beyond it; that
    must not count as this print going on (a caption under a print between two neighbours
    would otherwise be swallowed)."""
    h, w = paperlike.shape
    out = 1.0 if a >= 0 else -1.0
    vals = []
    for k in range(-4, 5, 2):
        for d in range(4, 11, 2):
            q = c + n_side * (pos + k) + along * (a + out * d)
            qx, qy = int(round(q[0])), int(round(q[1]))
            if 0 <= qx < w and 0 <= qy < h:
                vals.append(paperlike[qy, qx])
    return not vals or float(np.mean(vals)) >= 0.5


def _framed_by_edges(rect: tuple, grad: tuple[np.ndarray, np.ndarray], p: Params) -> bool:
    """True if every side of the rect lies on a real straight edge (or the image border).

    A single print is framed by its own border on all four sides, even where its content is
    paper-coloured. The bounding rectangle of several touching prints has sides that run
    through open paper, where there is no edge.
    """
    return _framed_sides(rect, grad, p) == 4


def _framed_sides(rect: tuple, grad: tuple[np.ndarray, np.ndarray], p: Params,
                  covered: np.ndarray | None = None) -> int:
    """How many sides of the rect lie on a real straight edge (or the image border).

    `covered` marks other prints lying on top: stretches of a side under them are not
    expected to show an edge and are ignored.
    """
    gx, gy = grad
    h, w = gx.shape
    (cx, cy), (rw, rh), ang = rect
    t = np.deg2rad(ang)
    u = np.array([np.cos(t), np.sin(t)], np.float32)
    v = np.array([-np.sin(t), np.cos(t)], np.float32)
    c = np.array([cx, cy], np.float32)
    framed = 0
    for n, along, half, length in ((u, v, rw / 2, rh), (v, u, rh / 2, rw)):
        for sign in (-1, 1):
            best = 0.0
            for o in range(-3, 4):
                ts = np.linspace(-0.45, 0.45, 200) * length
                pts = c + n * (sign * (half + o)) + ts[:, None] * along
                xs, ys = pts[:, 0], pts[:, 1]
                ok = (xs >= 0) & (xs <= w - 1) & (ys >= 0) & (ys <= h - 1)
                if ok.sum() < 0.5 * len(ts) or (ys[ok] < 2).all() or (ys[ok] > h - 3).all() \
                        or (xs[ok] < 2).all() or (xs[ok] > w - 3).all():
                    best = np.inf  # on/off the image border: the scan cut the print here
                    break
                xi, yi = xs[ok].astype(np.int32), ys[ok].astype(np.int32)
                if covered is not None:
                    free = covered[yi, xi] == 0
                    if free.sum() < 0.2 * len(ts):
                        best = np.inf  # side almost entirely under another print
                        break
                    xi, yi = xi[free], yi[free]
                g = np.abs(gx[yi, xi] * n[0] + gy[yi, xi] * n[1])
                # 10th percentile: the edge must run along ~90 % of the side
                best = max(best, float(np.percentile(g, 10)))
            framed += best >= p.snap_min_grad
    return framed


def _corner_print(r: _Region, grad: tuple[np.ndarray, np.ndarray], p: Params) -> bool:
    """A print lying in a corner of a flatbed scan, cut off by the scanner on two sides.

    No paper shows beyond it there, so it falls outside the page hull like the table or
    binding around a photographed page. Those touch three or four image sides, or are no
    clean rectangle (the page edge runs at an angle in a photo). A corner print touches two
    adjacent sides, and its other two sides are print edges.
    """
    h, w = grad[0].shape
    x, y, bw, bh = cv2.boundingRect(r.contour)
    left, top, right, bottom = x <= 1, y <= 1, x + bw >= w - 1, y + bh >= h - 1
    if left + top + right + bottom > 2 or (left and right) or (top and bottom):
        return False
    return (r.rectangularity >= p.good_rect and r.aspect <= p.max_aspect
            and _framed_sides(r.rect, grad, p) == 4)


def _inside(contour: np.ndarray, mask: np.ndarray) -> float:
    m = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(m, [contour], -1, 255, cv2.FILLED)
    return cv2.countNonZero(m & mask) / max(cv2.countNonZero(m), 1)


def detect_photos(lab: np.ndarray, paper: np.ndarray, p: Params
                  ) -> tuple[list[tuple[np.ndarray, bool]], np.ndarray, np.ndarray, np.ndarray]:
    """Returns ([(box_points, needs_review)], photo_mask, nonpage_mask, page_hull) in working coords."""
    h, w = lab.shape[:2]
    dist = paper_distance(lab, paper, p.paper_l_weight)
    paperlike = (dist <= p.paper_dist).astype(np.float32)
    paper_strict = (dist <= p.paper_dist * 0.5).astype(np.float32)
    grad = _edge_strength(lab)
    fg = _merge_fragments(_foreground(dist, p), paper_strict, p)
    fg = _merge_framed(fg, grad, paperlike, paper_strict, p)
    page_hull = _page_hull(paper_strict)
    min_area = p.min_photo_frac * h * w
    photo_mask = np.zeros((h, w), np.uint8)
    nonpage = np.zeros((h, w), np.uint8)
    found: list[tuple[np.ndarray, bool]] = []

    def accept(contour: np.ndarray, review: bool, siblings: np.ndarray | None = None) -> None:
        rect, unconfirmed = _snap_rect(cv2.minAreaRect(contour), grad, paperlike, paper_strict, p,
                                       covered=siblings, lab=lab)
        box = order_corners(cv2.boxPoints(rect))
        if review and _framed_sides(rect, grad, p, covered=siblings) == 4:
            review = False  # whatever made it look odd, the print border runs all the way round
        found.append((box, review or unconfirmed >= 2))
        cv2.fillPoly(photo_mask, [box.astype(np.int32)], 255)

    def accept_pieces(pieces: list[np.ndarray]) -> None:
        for i, piece in enumerate(pieces):
            cs, _ = cv2.findContours(piece, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            pc = max(cs, key=cv2.contourArea)
            pr = _region(pc, (h, w))
            siblings = np.zeros((h, w), np.uint8)
            for j, other in enumerate(pieces):
                if j != i:
                    siblings |= cv2.dilate(other, ellipse(7))
            if pr.area >= min_area * 0.5 and pr.aspect <= p.max_aspect:
                accept(pc, pr.rectangularity < p.good_rect, siblings)

    def covered(contour: np.ndarray) -> float:
        m = np.zeros((h, w), np.uint8)
        cv2.drawContours(m, [contour], -1, 255, cv2.FILLED)
        return cv2.countNonZero(m & photo_mask) / max(cv2.countNonZero(m), 1)

    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(contours, key=cv2.contourArea, reverse=True):
        r = _region(c, (h, w))
        if r.area < min_area:
            break
        if r.area > 0.8 * h * w:
            continue  # paper colour estimate failed; nothing sensible to say
        if covered(c) > 0.5:
            continue  # fragment of a photo already found (e.g. separated by a light sky)
        if r.touches_border and _inside(c, page_hull) < 0.5 and not _corner_print(r, grad, p):
            # Mostly outside the page: table, other pages, spiral binding of a photographed album.
            cv2.drawContours(nonpage, [c], -1, 255, cv2.FILLED)
            continue
        comp = np.zeros((h, w), np.uint8)
        cv2.drawContours(comp, [c], -1, 255, cv2.FILLED)
        pieces = _split_at_paper_lines(comp, paperlike, grad, p, min_area)
        if pieces:
            accept_pieces(pieces)
            continue
        # Other prints inside the blob's rectangle are not light content of this one: as far
        # as telling one print from several goes, they count as paper.
        strict_c = np.maximum(paper_strict, ((fg > 0) & (comp == 0)).astype(np.float32))
        bites_are_paper = _bite_paper(comp, strict_c) >= p.bite_paper
        if r.aspect <= p.max_aspect and (r.rectangularity >= p.good_rect or r.hull_rectangularity >= 0.95
                                         or not bites_are_paper):
            # Bites out of the rectangle that are not clean paper (sky, white walls) mean one
            # print with light areas, not several prints.
            accept(c, r.rectangularity < p.good_rect - 0.15)
            continue
        if r.aspect <= p.max_aspect and _framed_by_edges(r.rect, grad, p):
            # Paper-coloured bites, but the print border runs all the way round: one print.
            accept(c, True)
            continue
        pieces = _try_split(comp, min_area, p, strict_c, grad)
        if pieces:
            accept_pieces(pieces)
            continue
        if r.rectangularity >= p.min_rect and r.aspect <= p.max_aspect:
            accept(c, True)
    return found, photo_mask, nonpage, page_hull


def _border_strength_inside(box: np.ndarray, other: np.ndarray,
                            grad: tuple[np.ndarray, np.ndarray]) -> float:
    """Median edge strength along the parts of a print's border that run across another
    print (`other`: that print's mask). Visible if this print lies on top, hidden if not."""
    gx, gy = grad
    h, w = other.shape
    inner = cv2.erode(other, ellipse(9))  # stay clear of the other print's own border
    vals = []
    for a, b in zip(box, np.roll(box, -1, axis=0)):
        d = b - a
        length = float(np.hypot(*d))
        if length < 1:
            continue
        n = np.array([-d[1], d[0]], np.float32) / length
        ts = np.linspace(0, 1, max(20, int(length)))
        pts = a + ts[:, None] * d
        xi, yi = pts[:, 0].astype(np.int32), pts[:, 1].astype(np.int32)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        xi, yi = xi[ok], yi[ok]
        ok = inner[yi, xi] > 0
        for x, y in zip(xi[ok], yi[ok]):
            best = 0.0
            for o in range(-2, 3):
                qx, qy = int(x + o * n[0]), int(y + o * n[1])
                if 0 <= qx < w and 0 <= qy < h:
                    best = max(best, abs(gx[qy, qx] * n[0] + gy[qy, qx] * n[1]))
            vals.append(best)
    return float(np.median(vals)) if len(vals) >= 10 else 0.0


def occlusion_outlines(boxes: list[np.ndarray], grad: tuple[np.ndarray, np.ndarray],
                       shape: tuple[int, int]) -> list[np.ndarray | None]:
    """Visible outline of each print that lies partly under another one (else None).

    Where two prints overlap, the one on top shows its border across the other's picture;
    the border of the one underneath is hidden. Whichever border is stronger inside the
    overlap is taken to be on top.
    """
    h, w = shape
    masks = []
    for box in boxes:
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [box.astype(np.int32)], 255)
        masks.append(m)
    covered = [np.zeros((h, w), np.uint8) for _ in boxes]
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            inter = masks[i] & masks[j]
            if cv2.countNonZero(inter) < 0.01 * min(cv2.countNonZero(masks[i]), cv2.countNonZero(masks[j])):
                continue
            si = _border_strength_inside(boxes[i], masks[j], grad)
            sj = _border_strength_inside(boxes[j], masks[i], grad)
            if si >= sj:
                covered[j] |= masks[i]
            else:
                covered[i] |= masks[j]
    outlines: list[np.ndarray | None] = []
    for m, cov in zip(masks, covered):
        visible = m & ~cov
        area = cv2.countNonZero(m)
        if cv2.countNonZero(visible) > 0.97 * area or cv2.countNonZero(visible) < 0.2 * area:
            outlines.append(None)
            continue
        cs, _ = cv2.findContours(visible, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        outlines.append(cv2.approxPolyDP(c, 0.003 * max(h, w), True).reshape(-1, 2).astype(np.float32))
    return outlines


# --------------------------------------------------------------------------- text

def detect_texts(lab: np.ndarray, paper: np.ndarray, photo_mask: np.ndarray,
                 nonpage: np.ndarray, page_hull: np.ndarray, p: Params) -> list[tuple[int, int, int, int]]:
    """Returns text block boxes (x, y, w, h) in working coordinates."""
    h, w = lab.shape[:2]
    long_side = max(h, w)
    L = lab[..., 0]
    # Median background: robust to sparse strokes and centred on paper grain (phone shots).
    bg = cv2.medianBlur(np.clip(L, 0, 255).astype(np.uint8), max(3, int(long_side * 0.02)) | 1)
    dark = bg.astype(np.float32) - L.astype(np.float32)
    blue = paper[2] - lab[..., 2].astype(np.float32)
    ink = ((dark > p.ink_dark) | ((dark > p.ink_dark_blue) & (blue > p.ink_blue))).astype(np.uint8) * 255

    margin = max(3, int(long_side * p.photo_margin_frac))
    excluded = cv2.dilate(photo_mask, ellipse(2 * margin + 1)) | cv2.dilate(nonpage, ellipse(2 * margin + 1))
    # Solid shapes (stickers, printed logos) survive an opening that erases pen strokes.
    nonpaper = (paper_distance(lab, paper, p.paper_l_weight) > p.paper_dist).astype(np.uint8) * 255
    k_solid = int(long_side * p.solid_frac) | 1
    solid = cv2.morphologyEx(nonpaper, cv2.MORPH_OPEN, ellipse(k_solid))
    # Bold marker strokes survive that opening too, but only just: a solid shape is thicker
    # than the kernel somewhere (the star sticker on the samples: 35 px from its centre to its
    # edge; marker writing: at most 8 px).
    thick = cv2.distanceTransform(nonpaper, cv2.DIST_L2, 5)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if n > 1:
        peak = np.zeros(n, np.float32)
        np.maximum.at(peak, labels.ravel(), thick.ravel())
        solid = ((peak >= k_solid)[labels] & (labels > 0)).astype(np.uint8) * 255
    excluded |= cv2.dilate(solid, ellipse(2 * margin + 1))
    ink[excluded > 0] = 0
    if cv2.countNonZero(nonpage) > 0.03 * nonpage.size:
        ink[page_hull == 0] = 0  # photographed page: ignore everything off the page

    # Component-level cleanup: dust, punch holes, long thin lines (page/photo edges, scanner borders).
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    keep = np.zeros(n, bool)
    hole_like = np.zeros(n, bool)
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < p.min_ink_px:
            continue
        longest = max(bw, bh)
        if longest > 0.08 * long_side and area / longest < 4.0:
            continue  # thin line
        near_edge = x < 0.06 * w or x + bw > 0.94 * w
        if near_edge and 0.7 < bw / max(bh, 1) < 1.4 and 0.006 * long_side < longest < 0.03 * long_side:
            cs, _ = cv2.findContours((labels[y:y + bh, x:x + bw] == i).astype(np.uint8),
                                     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            c = max(cs, key=cv2.contourArea)
            per = cv2.arcLength(c, True)
            if per > 0 and 4 * np.pi * cv2.contourArea(c) / per ** 2 > 0.7:
                # Punch hole (solid or ring) at the binding edge -- or a round letter ("D",
                # "O") starting a caption there. Decided once blocks are formed.
                hole_like[i] = True
        keep[i] = True
    ink = (keep[labels]).astype(np.uint8) * 255
    hole_ink = hole_like[labels]

    kx = max(3, int(long_side * p.dilate_x_frac))
    ky = max(3, int(long_side * p.dilate_y_frac))
    blocks = cv2.dilate(ink, cv2.getStructuringElement(cv2.MORPH_RECT, (kx, ky)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(blocks, connectivity=8)
    boxes: list[tuple[int, int, int, int]] = []
    edge_marks: list[tuple[int, int, int, int]] = []
    fragments: list[tuple[int, int, int, int]] = []
    for i in range(1, n):
        member = (labels == i) & (ink > 0)
        if not (member & ~hole_ink).any():
            continue  # nothing but punch-hole-like marks
        cnt = int(member.sum())
        ys, xs = np.nonzero(member)
        x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
        if cnt < p.min_block_ink or max(y1 - y0, x1 - x0) < p.min_block_h_frac * long_side * 2 \
                or min(y1 - y0, x1 - x0) < p.min_block_h_frac * long_side:
            if cnt >= p.min_block_ink // 6:
                # Too small to be a caption -- but it may be a letter of one standing a
                # little apart (the "i" of "en dag i"). Kept only if it joins a caption.
                fragments.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0)))
            continue
        bw, bh = x1 - x0, y1 - y0
        if bh > 6 * bw or bw > 25 * bh:
            continue  # page/scanner edge, not handwriting
        if bw * bh > 0.15 * h * w or cnt / (bw * bh) > 0.55:  # (bold marker writing: ~0.43)
            continue  # an undetected photo or printed graphic, not a caption
        if bw > 2.5 * bh and _is_single_stroke(member[y0:y1, x0:x1]):
            continue  # an arrow or a line, not writing
        box = (int(x0), int(y0), int(bw), int(bh))
        at_edge = x0 < 0.06 * w or x1 > 0.94 * w
        if at_edge and max(bw, bh) < 0.04 * long_side:
            edge_marks.append(box)  # punch-hole remnant -- or the first letter of a caption
        else:
            boxes.append(box)
    pieces = _merge_boxes(boxes + edge_marks + fragments, gap=max(2, int(long_side * 0.003)))
    merged = _merge_lines(_split_at_photo_gaps(pieces, ink, photo_mask), ink, photo_mask, blue)
    # Again on the merged blocks: a short piece (a date like "27/8 95") between two captions
    # under neighbouring photos can bridge them; only now are both halves caption-sized.
    merged = _split_at_photo_gaps(merged, ink, photo_mask, min_gap=0.8)
    # Keep a small mark at the binding edge only if it joined a caption ("D" of "David");
    # left on its own it is a punch hole or its shadow. Likewise small fragments: dust.
    merged = [b for b in merged if any(_overlaps(b, r) for r in boxes)]
    return merged


def _text_outline(box: tuple[int, int, int, int], photo_mask: np.ndarray,
                  epsilon: float = 2.0) -> np.ndarray | None:
    """The paper part of a text box that runs onto a photo, as a polygon (else None).

    A caption may wrap around a print's corner: one line above it, the rest beside it. Its
    box then covers a corner of that print, which must not end up in the caption crop.
    """
    x, y, bw, bh = box
    sub = photo_mask[y:y + bh, x:x + bw]
    if cv2.countNonZero(sub) < 0.03 * bw * bh:
        return None  # grazing a photo's edge
    free = np.zeros((bh + 2, bw + 2), np.uint8)  # 1 px frame: contours along the box sides
    free[1:-1, 1:-1] = np.where(sub > 0, 0, 255)
    cs, _ = cv2.findContours(free, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    poly = cv2.approxPolyDP(c, epsilon, True).reshape(-1, 2).astype(np.float32)
    return poly + np.array([x - 1, y - 1], np.float32)


def _overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]


def _add_margin(boxes: list[tuple[int, int, int, int]], blocked: np.ndarray, margin: int
                ) -> list[tuple[int, int, int, int]]:
    """Grow each box by `margin` px per side, stopping early at a photo or the page edge."""
    h, w = blocked.shape
    out = []
    for x, y, bw, bh in boxes:
        x0, y0, x1, y1 = x, y, x + bw, y + bh
        for _ in range(margin):
            if x0 > 0 and not blocked[y0:y1, x0 - 1].any():
                x0 -= 1
            if x1 < w and not blocked[y0:y1, x1].any():
                x1 += 1
            if y0 > 0 and not blocked[y0 - 1, x0:x1].any():
                y0 -= 1
            if y1 < h and not blocked[y1, x0:x1].any():
                y1 += 1
        out.append((x0, y0, x1 - x0, y1 - y0))
    return out


def _is_single_stroke(ink: np.ndarray) -> bool:
    """A drawn line or arrow rather than writing: most of the ink lies along one straight line.
    Writing (even one line of cursive) spreads over the letter height around its baseline."""
    ys, xs = np.nonzero(ink)
    if len(xs) < 10:
        return True
    pts = np.column_stack([xs, ys]).astype(np.float32)
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
    dist = np.abs((pts[:, 0] - x0) * vy - (pts[:, 1] - y0) * vx)
    # Within a pen-stroke's width of the line (working image is a fixed size, so px are
    # comparable). On the samples: arrow 0.46, captions at most 0.23.
    if float((dist <= 2.0).mean()) < 0.35:
        return False
    # ... and drawn in one go: a line or arrow is one connected stroke, while writing (even
    # neat print lettering, whose ink also hugs its baseline) breaks into letters and words.
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
    return n > 1 and stats[1:, cv2.CC_STAT_AREA].max() >= 0.7 * len(xs)


def _line_height(ink: np.ndarray, r: list[int]) -> float:
    """Typical height of the writing in a box: median height of its words/letters."""
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink[r[1]:r[3], r[0]:r[2]], connectivity=8)
    hs = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= 20]
    return float(np.median(hs)) if hs else float(r[3] - r[1])


def _between_photos(a: list[int], b: list[int], photo_mask: np.ndarray) -> bool:
    """True if the gap between two pieces on one line continues a gap between two photos
    right above or below them, and both pieces are caption-sized (not a word or two): then
    each piece is the caption of its own photo."""
    h, w = photo_mask.shape
    long_side = max(h, w)
    if min(a[2] - a[0], b[2] - b[0]) < 0.12 * long_side:
        return False  # a few words next to a caption are more likely part of it
    band = max(8, int(0.04 * long_side))
    left, right = (a, b) if a[0] <= b[0] else (b, a)
    g0, g1 = left[2], right[0]
    top, bottom = min(a[1], b[1]), max(a[3], b[3])
    for y0, y1 in ((max(0, top - band), top), (bottom, min(h, bottom + band))):
        if y1 <= y0:
            continue
        cols = photo_mask[y0:y1].any(axis=0)
        if cols[left[0]:left[2]].any() and cols[right[0]:right[2]].any() and                 g1 > g0 and not cols[g0:g1].all():
            # photo above (below) each piece, and paper somewhere in the gap between them
            xs_left = np.nonzero(cols[left[0]:g1])[0]
            xs_right = np.nonzero(cols[g0:right[2]])[0]
            free = np.nonzero(~cols[g0:g1])[0] + g0
            if xs_left.size and xs_right.size and                     (free > xs_left.min() + left[0]).any() and (free < xs_right.max() + g0).any():
                return True
    return False


def _split_at_photo_gaps(boxes: list[tuple[int, int, int, int]], ink: np.ndarray,
                         photo_mask: np.ndarray, min_gap: float = 0.0
                         ) -> list[tuple[int, int, int, int]]:
    """Split blocks that the dilation joined across a gap between two photos.

    Two captions side by side under two neighbouring photos can sit closer than the dilation
    reaches. Where a block spans the continuation of the paper gap between two photos (right
    above or below it) and its ink leaves an empty column there, it is cut in two -- if both
    halves are caption-sized (`_between_photos`). With `min_gap`, the empty run of columns
    at the cut must be at least that many line heights wide: a gap between words, not
    between two letters of a title running above both photos.
    """
    h, w = photo_mask.shape
    band = max(8, int(0.04 * max(h, w)))
    out = []
    for x, y, bw, bh in boxes:
        cut = None
        for y0, y1 in ((max(0, y - band), y), (y + bh, min(h, y + bh + band))):
            if y1 <= y0:
                continue
            cols = photo_mask[y0:y1, x:x + bw].any(axis=0)
            if cols.all() or not cols.any():
                continue
            empty = ~ink[y:y + bh, x:x + bw].any(axis=0)
            on = np.nonzero(cols)[0]
            need = min_gap * _line_height(ink, [x, y, x + bw, y + bh]) if min_gap else 0.0
            # photo-free columns between photo columns: a gap between two photos
            for c in np.nonzero(~cols)[0]:
                if on.min() < c < on.max() and empty[c] and _empty_run(empty, c) >= need:
                    cut = c
                    break
            if cut is not None:
                break
        if cut is None:
            out.append((x, y, bw, bh))
            continue
        parts = []
        for a0, a1 in ((0, cut), (cut + 1, bw)):
            ys, xs = np.nonzero(ink[y:y + bh, x + a0:x + a1])
            if xs.size:
                parts.append([x + a0 + xs.min(), y + ys.min(), x + a0 + xs.max() + 1, y + ys.max() + 1])
        if len(parts) == 2 and _between_photos(parts[0], parts[1], photo_mask):
            out.extend((r[0], r[1], r[2] - r[0], r[3] - r[1]) for r in parts)
        else:
            out.append((x, y, bw, bh))
    return out


def _empty_run(empty: np.ndarray, c: int) -> int:
    """Width of the run of True values in `empty` that contains index c."""
    lo, hi = c, c
    while lo > 0 and empty[lo - 1]:
        lo -= 1
    while hi < len(empty) - 1 and empty[hi + 1]:
        hi += 1
    return hi - lo + 1


def _ink_blue(ink: np.ndarray, blue: np.ndarray, r: list[int]) -> float:
    """Median blueness (b* below the paper's) of the ink in a box."""
    m = ink[r[1]:r[3], r[0]:r[2]] > 0
    return float(np.median(blue[r[1]:r[3], r[0]:r[2]][m])) if m.any() else 0.0


def _merge_lines(boxes: list[tuple[int, int, int, int]], ink: np.ndarray, photo_mask: np.ndarray,
                 blue: np.ndarray | None = None) -> list[tuple[int, int, int, int]]:
    """Join pieces of one caption that the fixed-size dilation left apart.

    Large or widely spaced handwriting leaves gaps between words and lines that scale with
    the writing, so gaps are measured in line heights (not box heights: a two-line caption
    is twice as tall but written no larger). Words on the same line are joined when the gap
    is under 2.2 line heights (separate captions sit ~4 or more apart); stacked lines when they overlap horizontally and the gap is
    under 1 line height (separate captions sit ~2 or more apart). Never across a photo, and
    never between a blue ballpoint and a neutral (black/grey) piece: one caption is written with
    one pen, and printed matter next to a caption (a cut-out ticket) is neutral.
    """
    rects = [[x, y, x + bw, y + bh] for x, y, bw, bh in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(rects)):
            for j in range(len(rects)):
                if i == j:
                    continue
                a, b = rects[i], rects[j]
                ha, hb = a[3] - a[1], b[3] - b[1]
                v_overlap = min(a[3], b[3]) - max(a[1], b[1])
                h_overlap = min(a[2], b[2]) - max(a[0], b[0])
                h_gap = max(a[0], b[0]) - min(a[2], b[2])
                v_gap = max(a[1], b[1]) - min(a[3], b[3])
                if h_gap > 0 and v_gap > 0:
                    continue  # diagonal neighbours
                # The larger: a lone letter ("i") is shorter than the writing it belongs to.
                line_h = max(_line_height(ink, a), _line_height(ink, b))
                same_line = v_overlap >= 0.5 * min(ha, hb) and h_gap <= 2.2 * line_h
                stacked = (h_overlap >= 0.5 * min(a[2] - a[0], b[2] - b[0])
                           and 0 <= v_gap <= 1.0 * line_h)
                if not (same_line or stacked):
                    continue
                if same_line and not stacked and h_gap > 0 and _between_photos(a, b, photo_mask):
                    continue  # side by side under (or over) two photos: one caption each
                if blue is not None:
                    ba, bb = _ink_blue(ink, blue, a), _ink_blue(ink, blue, b)
                    if max(ba, bb) > 8 and min(ba, bb) < 5:
                        continue  # blue ballpoint next to neutral ink: different pens
                    # (A caption corrected in black still has a bluish median: not neutral.)
                u = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]

                def on_photo(r: list[int]) -> int:
                    return cv2.countNonZero(photo_mask[r[1]:r[3], r[0]:r[2]])

                # Only photo area the merge would newly cover counts (a box hugging a photo's
                # edge already overlaps it a little).
                if on_photo(u) - on_photo(a) - on_photo(b) > 0.02 * (u[2] - u[0]) * (u[3] - u[1]):
                    continue
                rects[i] = u
                del rects[j]
                merged = True
                break
            if merged:
                break
    out = [(r[0], r[1], r[2] - r[0], r[3] - r[1]) for r in rects]
    return _merge_boxes(out, gap=0)  # merged boxes may now overlap others


def _merge_boxes(boxes: list[tuple[int, int, int, int]], gap: int) -> list[tuple[int, int, int, int]]:
    rects = [[x, y, x + bw, y + bh] for x, y, bw, bh in boxes]
    merged = True
    while merged:
        merged = False
        for i in range(len(rects)):
            for j in range(i + 1, len(rects)):
                a, b = rects[i], rects[j]
                if a[0] - gap <= b[2] and b[0] - gap <= a[2] and a[1] - gap <= b[3] and b[1] - gap <= a[3]:
                    rects[i] = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    del rects[j]
                    merged = True
                    break
            if merged:
                break
    return [(r[0], r[1], r[2] - r[0], r[3] - r[1]) for r in rects]


# --------------------------------------------------------------------------- page

def _reading_order(items: list, center) -> list:
    """Sort top-to-bottom in rows, left-to-right within a row."""
    items = sorted(items, key=lambda it: center(it)[1])
    rows: list[list] = []
    for it in items:
        cy = center(it)[1]
        if rows and abs(cy - np.mean([center(o)[1] for o in rows[-1]])) < _half_height(rows[-1][0]):
            rows[-1].append(it)
        else:
            rows.append([it])
    return [it for row in rows for it in sorted(row, key=lambda o: center(o)[0])]


def _half_height(item) -> float:
    x0, y0, x1, y1 = item.bbox()
    return (y1 - y0) / 2


def detect_page(img_bgr: np.ndarray, p: Params | None = None) -> Detection:
    p = p or Params()
    H, W = img_bgr.shape[:2]
    small, scale = downscale(img_bgr, p.work_long_side)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    lab = flatten_lighting(lab, estimate_paper(lab))
    paper = estimate_paper(lab)

    found, photo_mask, nonpage, page_hull = detect_photos(lab, paper, p)
    text_boxes = detect_texts(lab, paper, photo_mask, nonpage, page_hull, p)

    def to_full(pt) -> Point:
        x = float(np.clip(pt[0] / scale, 0, W - 1))
        y = float(np.clip(pt[1] / scale, 0, H - 1))
        return round(x, 1), round(y, 1)

    outlines = occlusion_outlines([box for box, _ in found], _edge_strength(lab), lab.shape[:2])
    photos = [Photo(id=0, corners=[to_full(pt) for pt in box], needs_review=review,
                    outline=[to_full(pt) for pt in ol] if ol is not None else None)
              for (box, review), ol in zip(found, outlines)]
    photos = _reading_order(photos, lambda ph: ph.center())
    for i, ph in enumerate(photos, 1):
        ph.id = i
    if p.stamps:
        # On the full-resolution image: the digits are only ~10 px tall in the working copy.
        for ph in photos:
            ph.stamp = find_stamp(img_bgr, ph.corners, ph.outline, p.stamp)
        harmonize([ph.stamp for ph in photos if ph.stamp])

    def box_to_full(b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        x0, y0 = to_full((b[0], b[1]))
        x1, y1 = to_full((b[0] + b[2], b[1] + b[3]))
        return round(x0), round(y0), round(x1 - x0), round(y1 - y0)

    # Link on the tight ink boxes: the margin would close the gaps the linker measures.
    margined = _add_margin(text_boxes, photo_mask | nonpage,
                           int(round(max(lab.shape[:2]) * p.text_margin_frac)))
    texts = [TextBlock(id=0, box=box_to_full(b)) for b in text_boxes]
    padded = {id(t): box_to_full(m) for t, m in zip(texts, margined)}
    outlines_t = {id(t): _text_outline(m, photo_mask) for t, m in zip(texts, margined)}
    texts = _reading_order(texts, lambda t: t.center())
    for i, t in enumerate(texts, 1):
        t.id = i
    links = link_heuristic(photos, texts)
    for t in texts:
        t.box = padded[id(t)]
        ol = outlines_t[id(t)]
        t.outline = [to_full(pt) for pt in ol] if ol is not None else None

    page_quad = None
    if cv2.countNonZero(nonpage) > 0.03 * nonpage.size:
        page = cv2.bitwise_not(nonpage)
        cs, _ = cv2.findContours(page, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            box = order_corners(cv2.boxPoints(cv2.minAreaRect(max(cs, key=cv2.contourArea))))
            page_quad = [to_full(pt) for pt in box]

    return Detection(photos=photos, texts=texts, page_quad=page_quad, links=links)
