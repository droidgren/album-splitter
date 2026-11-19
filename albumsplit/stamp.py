"""Burnt-in camera date stamps: find the orange seven-segment digits inside a photo and read them.

Compact cameras of the 80s/90s exposed the date onto the film with small orange LEDs, so the
stamp is part of the print: e.g. `6 5 '95` or `'93 7 14`. In the camera's orientation it sits at
the bottom right, running along the long edge. A print can lie on the page in any 90° rotation, so
once the photo is rectified to landscape the stamp is either at the bottom right (upright) or at
the top left (upside down). Both are searched.

The LEDs add red light, so the digits are redder than their surroundings. Depending on the
background that makes them saturated orange (grass, water) or only faintly pink (sky).

1. Warm contrast: a* (+ some b*) minus its median over about one digit height, so thin strokes
   stand out while warm backgrounds (brick paths, red walls) do not. Thresholded against the
   band's own noise, at each of a few digit sizes.
2. Glyphs: blobs of digit or apostrophe size, grouped into a horizontal line. Each line is
   segmented once more from its own neighbourhood (thresholds relative to its brightest strokes),
   which removes glow halos and keeps faint segments.
3. Reading: each glyph's seven segment positions give a cost for every digit. The line is read as
   the cheapest combination of digits that forms a valid date. Near-ties are kept (`date_alts`)
   and settled against the other stamps of the page (`harmonize`).

Segment-like noise in textured photos rarely forms a line that reads as a valid date.
"""

from __future__ import annotations

import datetime as dt
import itertools
from dataclasses import dataclass, field

import cv2
import numpy as np

from .model import DateStamp, Point


@dataclass
class StampParams:
    long_side: int = 1600        # rectified photo size (long side, px)
    grow: float = 0.012          # outward margin around the photo quad (frames can stop short)
    band: float = 0.36           # search band along each long edge (fraction of the short side)
    digit_h: tuple[float, ...] = (0.018, 0.024, 0.031, 0.040, 0.052, 0.068)  # x short side
    b_weight: float = 0.3        # warm = a* + b_weight * b*
    low_abs: float = 4.0         # hysteresis thresholds on the warm contrast (Lab units) ...
    high_abs: float = 9.0
    low_k: float = 3.0           # ... or this many robust sigmas of the band's noise
    high_k: float = 5.5
    reread_rel: tuple[float, ...] = (0.25, 0.35, 0.5)  # local thresholds (x line peak)
    min_digits: int = 3          # decoded digits needed to accept a stamp
    min_score: float = 1.0
    year_pivot: int = 70         # two-digit years >= pivot are 19YY, below 20YY
    years: tuple[int, int] = (1975, 2015)  # era of date-imprinting film cameras
    alt_gap: float = 0.5         # keep other dates whose readings cost less extra than this


# Seven-segment patterns, segments "abcdefg" (a top, b upper right, c lower right, d bottom,
# e lower left, f upper left, g middle). Alternates: 6 without top, 7 with upper left,
# 9 without bottom.
_SEGMENTS = {
    "abcdef": "0", "bc": "1", "abdeg": "2", "abcdg": "3", "bcfg": "4", "acdfg": "5",
    "acdefg": "6", "cdefg": "6", "abc": "7", "abcf": "7", "abcdefg": "8", "abcdfg": "9",
    "abcfg": "9",
}
# Where to sample each segment: (y0, y1, x0, x1) as fractions of the glyph's tight box, the
# middle part of each segment so neighbouring segments' ends do not leak in.
_ZONES = {
    "a": (0.00, 0.18, 0.30, 0.70), "b": (0.15, 0.38, 0.70, 1.00), "c": (0.62, 0.85, 0.70, 1.00),
    "d": (0.82, 1.00, 0.30, 0.70), "e": (0.62, 0.85, 0.00, 0.30), "f": (0.15, 0.38, 0.00, 0.30),
    "g": (0.41, 0.59, 0.30, 0.70),
}
_MAX_COST = 2.2    # a digit is read if its pattern explains the segment levels this well ...
_MIN_MARGIN = 0.2  # ... and clearly better than any other digit
_ALT_COST = 0.6    # when looking for a valid date, also try digits this close to the best


@dataclass
class _Glyph:
    x0: int
    y0: int
    x1: int
    y1: int
    kind: str          # "digit" or "tick" (the apostrophe before a two-digit year)
    char: str = "?"
    costs: dict[str, float] = field(default_factory=dict)  # digit -> how badly it fits
    patch: np.ndarray | None = None  # warm contrast under the glyph

    @property
    def h(self) -> int:
        return self.y1 - self.y0


@dataclass
class Candidate:
    """A line of seven-segment glyphs in the (rotated) search band."""
    glyphs: list[_Glyph]
    digit_h: float
    box: tuple[int, int, int, int]       # x0, y0, x1, y1 in the oriented crop
    flipped: bool                         # found in the crop rotated by 180°
    contrast: float
    score: float = 0.0
    text: str = ""
    fields: list[str] = field(default_factory=list)
    year_field: int | None = None
    date: str | None = None
    date_alts: list[str] = field(default_factory=list)  # nearly as likely dates


# --------------------------------------------------------------------------- public API

def find_stamp(img_bgr: np.ndarray, corners: list[Point], outline: list[Point] | None = None,
               sp: StampParams | None = None) -> DateStamp | None:
    """Locate and read the date stamp of one photo. `corners` are TL, TR, BR, BL (full res)."""
    sp = sp or StampParams()
    crop, M, valid = rectify(img_bgr, corners, outline, sp)
    cands = candidates(crop, valid, sp)
    if not cands:
        return None
    best = max(cands, key=lambda c: c.score)
    if best.score < sp.min_score:
        return None
    return DateStamp(corners=_box_to_page(best, crop.shape[:2], M), text=best.text, date=best.date,
                     date_alts=best.date_alts, score=round(best.score, 2))


def harmonize(stamps: list[DateStamp]) -> None:
    """Settle near-ties with the other stamps of the page.

    Prints on one album page are usually from the same occasion. Going from the clearest stamp
    to the faintest, a stamp whose own best reading matches none of the clearer stamps but one
    of its near-tie readings does, takes that reading (a blurred 6 read as a 5).
    """
    settled: set[str] = set()
    for st in sorted(stamps, key=lambda s: -s.score):
        if st.date and st.date not in settled:
            match = next((a for a in st.date_alts if a in settled), None)
            if match:
                st.date_alts = [st.date] + [a for a in st.date_alts if a != match]
                st.date = match
        if st.date:
            settled.add(st.date)


def parse_stamp(fields: list[str], year_field: int | None, pivot: int = 70) -> str | None:
    """Turn the three digit fields of a stamp into an ISO date (None if invalid or ambiguous).

    `year_field` is the index of the field preceded by an apostrophe ('93), if one was seen.
    Year first (`'93 7 14`) is Y M D; year last (`6 5 '95`) is D M Y, or M D Y if the middle
    field cannot be a month.
    """
    if len(fields) != 3 or any(not f.isdigit() or len(f) > 2 for f in fields):
        return None
    n = [int(f) for f in fields]
    if year_field is None:
        # No apostrophe seen: the year is whichever end field cannot be a day.
        first_year = len(fields[0]) == 2 and n[0] > 31
        last_year = len(fields[2]) == 2 and n[2] > 31
        if first_year == last_year:
            return None  # ambiguous (or nonsense)
        year_field = 0 if first_year else 2
    if year_field == 0:
        y, m, d = n
    elif year_field == 2:
        d, m, y = n
        if m > 12 and d <= 12:
            d, m = m, d
    else:
        return None
    if len(fields[year_field]) != 2:
        return None
    year = 1900 + y if y >= pivot else 2000 + y
    try:
        return dt.date(year, m, d).isoformat()
    except ValueError:
        return None


# --------------------------------------------------------------------------- geometry

def rectify(img_bgr: np.ndarray, corners: list[Point], outline: list[Point] | None, sp: StampParams
            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Warp the photo to an upright landscape crop (portrait prints are turned 90° clockwise).

    Returns (crop, M, valid) where M maps page -> crop coordinates and `valid` masks out the
    part of the print hidden under another one.
    """
    q = np.asarray(corners, np.float32).reshape(4, 2)
    w = (np.linalg.norm(q[1] - q[0]) + np.linalg.norm(q[2] - q[3])) / 2
    h = (np.linalg.norm(q[3] - q[0]) + np.linalg.norm(q[2] - q[1])) / 2
    ux = (q[1] - q[0] + q[2] - q[3]) / 2
    uy = (q[3] - q[0] + q[2] - q[1]) / 2
    ux /= max(np.linalg.norm(ux), 1e-6)
    uy /= max(np.linalg.norm(uy), 1e-6)
    g = sp.grow * max(w, h)
    sx, sy = np.array([-1, 1, 1, -1], np.float32), np.array([-1, -1, 1, 1], np.float32)
    src = q + g * (sx[:, None] * ux + sy[:, None] * uy)
    w, h = w + 2 * g, h + 2 * g
    s = min(1.0, sp.long_side / max(w, h))
    if w >= h:
        cw, ch = max(1, round(w * s)), max(1, round(h * s))
        dst = np.array([(0, 0), (cw, 0), (cw, ch), (0, ch)], np.float32)
    else:  # turn clockwise: TL -> TR, TR -> BR, ...
        cw, ch = max(1, round(h * s)), max(1, round(w * s))
        dst = np.array([(cw, 0), (cw, ch), (0, ch), (0, 0)], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    crop = cv2.warpPerspective(img_bgr, M, (cw, ch), flags=cv2.INTER_AREA,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=(128, 128, 128))
    valid = np.full((ch, cw), 255, np.uint8)
    if outline:
        valid[:] = 0
        poly = cv2.perspectiveTransform(np.asarray(outline, np.float32).reshape(-1, 1, 2), M)
        cv2.fillPoly(valid, [np.round(poly).astype(np.int32)], 255)
    return crop, M, valid


def _box_to_page(c: Candidate, shape: tuple[int, int], M: np.ndarray) -> list[Point]:
    ch, cw = shape
    x0, y0, x1, y1 = c.box
    pad = 0.35 * c.digit_h
    pts = np.array([(x0 - pad, y0 - pad), (x1 + pad, y0 - pad), (x1 + pad, y1 + pad),
                    (x0 - pad, y1 + pad)], np.float32)
    if c.flipped:
        pts = np.array([(cw - x, ch - y) for x, y in pts], np.float32)
    page = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), np.linalg.inv(M)).reshape(-1, 2)
    return [(round(float(x), 1), round(float(y), 1)) for x, y in page]


# --------------------------------------------------------------------------- detection

def warmth(crop: np.ndarray, sp: StampParams) -> tuple[np.ndarray, np.ndarray]:
    """(warm, reddish): warm = a* + b_weight * b* (Lab units). reddish marks pixels whose red
    channel is the strongest, as LED light makes it."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
    warm = (lab[..., 1] - 128) + sp.b_weight * (lab[..., 2] - 128)
    b, g, r = (crop[..., i].astype(np.int16) for i in range(3))
    return warm, (r > g) & (r > b)


def warm_contrast(warm: np.ndarray, reddish: np.ndarray, kernel: int) -> np.ndarray:
    """How much warmer each pixel is than the median of a `kernel`-sized neighbourhood.

    The kernel is about one digit tall: the thin strokes stay out of the median, while a warm
    background around the stamp (a brick path, a red wall) becomes part of it.
    """
    # medianBlur needs uint8 for big kernels: store warm at half-unit precision.
    enc = np.clip(warm * 2 + 128, 0, 255).astype(np.uint8)
    # A median over a big window is smooth: compute it on a smaller copy (much faster).
    f = max(1, kernel // 15)
    if f > 1:
        hh, ww = enc.shape
        small = cv2.resize(enc, (max(1, ww // f), max(1, hh // f)), interpolation=cv2.INTER_AREA)
        med = cv2.resize(cv2.medianBlur(small, (kernel // f) | 1), (ww, hh),
                         interpolation=cv2.INTER_LINEAR)
    else:
        med = cv2.medianBlur(enc, kernel)
    bg = (med.astype(np.float32) - 128) / 2
    d = warm - bg
    d[~reddish] = 0
    return d


def stamp_mask(d: np.ndarray, valid: np.ndarray, sp: StampParams) -> tuple[np.ndarray, float]:
    """Hysteresis threshold of the warm contrast, adapted to the band's noise."""
    v = d[valid > 0]
    if v.size == 0:
        return np.zeros(d.shape, np.uint8), 1.0
    sigma = 1.4826 * float(np.median(np.abs(v - np.median(v)))) + 0.5
    lo = max(sp.low_abs, sp.low_k * sigma)
    hi = max(sp.high_abs, sp.high_k * sigma)
    low = ((d > lo) & (valid > 0)).astype(np.uint8)
    n, labels = cv2.connectedComponents(low, connectivity=8)
    strong = np.zeros(n, bool)
    strong[np.unique(labels[(d > hi) & (low > 0)])] = True
    strong[0] = False
    return strong[labels].astype(np.uint8) * 255, lo


def candidates(crop: np.ndarray, valid: np.ndarray, sp: StampParams) -> list[Candidate]:
    """All stamp-like lines in both search bands, at every digit size, each with its reading."""
    ch, cw = crop.shape[:2]
    short = min(ch, cw)
    band = max(8, int(sp.band * short))
    y_off = ch - band
    warm, reddish = warmth(crop, sp)
    out: list[Candidate] = []
    for frac in sp.digit_h:
        h = frac * short
        k = max(5, int(1.3 * h)) | 1
        pad = min(k // 2 + 1, y_off)  # rows beyond the band that the median window reaches
        for flipped in (False, True):
            # Only the band is needed (plus the window's reach, for identical borders).
            w_, r_, v_ = (np.rot90(a, 2) if flipped else a for a in (warm, reddish, valid))
            db = np.ascontiguousarray(
                warm_contrast(np.ascontiguousarray(w_[y_off - pad:]), r_[y_off - pad:], k)[pad:])
            vb = np.ascontiguousarray(v_[y_off:])
            mask, lo = stamp_mask(db, vb, sp)
            if not mask.any():
                continue
            for c in _lines(mask, db, h, lo):
                best = c
                _read(c, sp)
                c.score = _score(c, cw, sp)
                for r in _reread(c, db, lo, sp):
                    _read(r, sp)
                    r.score = _score(r, cw, sp)
                    if r.score > best.score:
                        best = r
                best.flipped = flipped
                best.box = (best.box[0], best.box[1] + y_off, best.box[2], best.box[3] + y_off)
                out.append(best)
    return out


def _reread(c: Candidate, d: np.ndarray, lo: float, sp: StampParams) -> list[Candidate]:
    """Segment a found line again from its own neighbourhood, at thresholds relative to its
    brightest strokes.

    The band-wide threshold suits finding stamps, but a bright stamp then comes with glow
    halos (digits swell and merge) and a faint one may lose segments. Locally, a fraction of
    the stamp's own peak separates strokes from glow and background.
    """
    h = c.digit_h
    x0, y0, x1, y1 = c.box
    rx0, ry0 = max(0, int(x0 - 3 * h)), max(0, int(y0 - 0.5 * h))
    rx1, ry1 = min(d.shape[1], int(x1 + 3 * h)), min(d.shape[0], int(y1 + 0.5 * h))
    region = np.ascontiguousarray(d[ry0:ry1, rx0:rx1])
    inner = d[y0:y1, x0:x1]
    if inner.size == 0:
        return []
    peak = float(np.percentile(inner, 98))
    vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, int(round(0.4 * h)))))
    out = []
    for rel, hh in itertools.product(sp.reread_rel, (h, 0.8 * h)):
        # (The line's digit height was measured on glow-swollen glyphs: also try smaller.)
        t = max(lo, rel * peak)
        mask = (region > t).astype(np.uint8) * 255
        # The local mask is clean, so upper and lower segments that sit far apart (a "1" of
        # two dots) can safely be joined vertically.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, vert)
        lines = [r for r in _lines(mask, region, hh, t)
                 if r.box[0] + rx0 < x1 and r.box[2] + rx0 > x0]  # overlaps the original
        if not lines:
            continue
        r = max(lines, key=lambda r: len(r.glyphs))
        r.box = (r.box[0] + rx0, r.box[1] + ry0, r.box[2] + rx0, r.box[3] + ry0)
        for g in r.glyphs:  # glyphs stay in band coordinates like the original's
            g.x0, g.x1, g.y0, g.y1 = g.x0 + rx0, g.x1 + rx0, g.y0 + ry0, g.y1 + ry0
        r.contrast = c.contrast  # judged by the band-wide noise, like every other candidate
        out.append(r)
    return out


def _glyphs(mask: np.ndarray, h: float) -> list[_Glyph]:
    """Digit- and apostrophe-sized pieces of the mask, for digits about `h` px tall.

    A closing of 0.2 h groups the segments of a digit into one blob. Digits of one field sit
    only ~0.15 h apart, so it may merge them as well: blobs wider than a digit are cut again at
    the thinnest columns of a lighter closing (0.1 h, which keeps the gaps between digits).
    An apostrophe touching its digit is split off later, per line (_split_merged).
    """
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ellipse(max(2, int(round(0.2 * h)))))
    light = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ellipse(max(2, int(round(0.1 * h)))))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
    out: list[_Glyph] = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if 0.12 * h <= bh <= 0.5 * h and 0.08 * h <= bw <= 0.42 * h:
            out.append(_Glyph(x, y, x + bw, y + bh, "tick"))
            continue
        if not (0.7 * h <= bh <= 1.35 * h and 0.08 * h <= bw <= 4 * h):
            continue  # too small, too tall, or a long red streak rather than a few digits
        own = labels[y:y + bh, x:x + bw] == i
        seg = (light[y:y + bh, x:x + bw] > 0) & own
        out.extend(_split_wide(seg, (mask[y:y + bh, x:x + bw] > 0) & own, x, y, h))
    return out


def _split_wide(seg: np.ndarray, raw: np.ndarray, x: int, y: int, h: float) -> list[_Glyph]:
    """Cut a blob of touching digits at its thinnest columns until no piece is wider than a
    digit. Ties go to the column nearest the middle of the piece. `raw` is the blob before
    closing, used to tell digits (with holes) from solid blobs."""
    w = seg.shape[1]
    pieces = [(0, w)]
    while True:
        a, b = max(pieces, key=lambda p: p[1] - p[0])
        if b - a <= 0.9 * h:
            break
        counts = seg[:, a:b].sum(axis=0).astype(np.float32)
        lo_, hi_ = int(0.3 * h), (b - a) - int(0.3 * h)
        if hi_ <= lo_:
            break
        mid = (b - a) / 2
        cols = np.arange(lo_, hi_)
        cut = a + int(cols[np.argmin(counts[lo_:hi_] + np.abs(cols - mid) / (b - a))])
        if counts[cut - a] > 0.25 * h:
            if b - a <= 1.25 * h:
                break  # maybe a digit with its apostrophe attached: see _split_merged
            return []  # one wide blob, not touching digits
        pieces.remove((a, b))
        pieces += [(a, cut), (cut + 1, b)]
    out = []
    for a, b in sorted(pieces):
        ys = np.nonzero(seg[:, a:b].any(axis=1))[0]
        if ys.size == 0 or ys[-1] + 1 - ys[0] < 0.7 * h:
            continue
        xs = np.nonzero(seg[:, a:b].any(axis=0))[0]
        a, b = a + int(xs[0]), a + int(xs[-1]) + 1
        g = _Glyph(x + a, y + int(ys[0]), x + b, y + int(ys[-1]) + 1, "digit")
        fill = float(raw[ys[0]:ys[-1] + 1, a:b].mean())
        if g.x1 - g.x0 < 0.4 * h or 0.12 <= fill <= 0.9:  # "1" is a solid bar; others hollow
            out.append(g)
    return out


def _lines(mask: np.ndarray, d: np.ndarray, h: float, lo: float) -> list[Candidate]:
    """Group glyphs into horizontal lines, for digits about `h` px tall."""
    glyphs = _glyphs(mask, h)
    glyphs.sort(key=lambda g: g.x0)

    lines: list[list[_Glyph]] = []
    for g in glyphs:
        placed = False
        for ln in lines:
            last = ln[-1]
            if g.x0 - last.x1 > 2.6 * h or g.x0 < last.x1 - 0.3 * h:
                continue
            digits = [o for o in ln if o.kind == "digit"]
            if not digits:  # a lone tick: the next digit must follow right after it ('93)
                if g.kind == "digit" and g.x0 - last.x1 < 0.6 * h and abs(g.y0 - last.y0) < 0.3 * h:
                    ln.append(g)
                    placed = True
                    break
                continue
            top = _median([o.y0 for o in digits])
            if (g.kind == "digit" and abs(g.y0 - top) < 0.3 * h
                    and abs(g.y1 - _median([o.y1 for o in digits])) < 0.3 * h):
                ln.append(g)
                placed = True
                break
            if g.kind == "tick" and abs(g.y0 - top) < 0.3 * h and g.y1 < top + 0.6 * h:
                ln.append(g)
                placed = True
                break
        if not placed:
            lines.append([g])  # (a tick can start a line: '93 7 14)

    out = []
    for ln in lines:
        digits = [g for g in ln if g.kind == "digit"]
        if len(digits) < 3:
            continue
        ln = _split_merged(ln, mask, h)
        # Drop ticks that are not just before a digit (stray specks at the ends).
        ln = [g for i, g in enumerate(ln) if g.kind == "digit" or
              (i + 1 < len(ln) and ln[i + 1].kind == "digit" and ln[i + 1].x0 - g.x1 < 0.6 * h)]
        x0 = min(g.x0 for g in ln)
        x1 = max(g.x1 for g in ln)
        y0 = min(g.y0 for g in digits)
        y1 = max(g.y1 for g in digits)
        if not 1.5 <= (x1 - x0) / h <= 14:
            continue
        for g in ln:  # the warm contrast under each glyph, for decoding
            g.patch = np.clip(d[g.y0:g.y1, g.x0:g.x1], 0, None)
        sub = d[y0:y1, x0:x1][mask[y0:y1, x0:x1] > 0]
        contrast = float(np.median(sub)) / lo if sub.size else 0.0
        out.append(Candidate(glyphs=ln, digit_h=float(np.median([g.h for g in digits])),
                             box=(x0, y0, x1, y1), flipped=False, contrast=contrast))
    return out


def _split_merged(ln: list[_Glyph], mask: np.ndarray, h: float) -> list[_Glyph]:
    """Split glyphs that are wider than the other digits of their line.

    Either an apostrophe touches its digit ('9 of '95): columns at the start that hold ink only
    in the upper half are cut off as the apostrophe. (A lone 5 or 4 also starts with an
    upper-half stroke, so width is what tells them apart.) Or two digits touch (a faint "12"
    whose glow fills the gap): cut at the thinnest column.
    """
    widths = [g.x1 - g.x0 for g in ln if g.kind == "digit" and g.x1 - g.x0 >= 0.4 * h]
    if len(widths) < 2:
        return ln
    med = float(np.median(widths))
    out: list[_Glyph] = []
    for g in ln:
        w, gh = g.x1 - g.x0, g.y1 - g.y0
        if g.kind != "digit" or w <= 1.12 * med:
            out.append(g)
            continue
        m = mask[g.y0:g.y1, g.x0:g.x1] > 0
        if not (out and out[-1].kind == "tick"):
            lowest = [int(np.nonzero(m[:, c])[0].max()) if m[:, c].any() else -1
                      for c in range(min(w, int(0.45 * h) + 1))]
            lead = 0
            while lead < len(lowest) and lowest[lead] < 0.5 * gh:
                lead += 1
            rest = np.nonzero(m[:, lead:].any(axis=0))[0]
            if max(2, 0.1 * h) <= lead <= 0.45 * h and rest.size and w - lead - int(rest[0]) >= 0.4 * h:
                out.append(_Glyph(g.x0, g.y0, g.x0 + lead, g.y0 + max(lowest[:lead]) + 1, "tick"))
                out.append(_Glyph(g.x0 + lead + int(rest[0]), g.y0, g.x1, g.y1, "digit"))
                continue
        if w > 1.25 * med:
            counts = m.sum(axis=0)
            lo_, hi_ = max(1, int(0.12 * w)), w - max(1, int(0.12 * w))
            cut = lo_ + int(np.argmin(counts[lo_:hi_])) if hi_ > lo_ else -1
            if cut > 0 and counts[cut] <= 0.25 * gh:
                parts = []
                for a, b in ((0, cut), (cut + 1, w)):
                    cols = np.nonzero(m[:, a:b].any(axis=0))[0]
                    rows = np.nonzero(m[:, a:b].any(axis=1))[0]
                    if cols.size and rows[-1] + 1 - rows[0] >= 0.6 * gh:
                        parts.append(_Glyph(g.x0 + a + int(cols[0]), g.y0 + int(rows[0]),
                                            g.x0 + a + int(cols[-1]) + 1, g.y0 + int(rows[-1]) + 1,
                                            "digit"))
                if len(parts) == 2:
                    out.extend(parts)
                    continue
        out.append(g)
    return out


def _median(v: list[int]) -> float:
    """Median of a short list (np.median is slow for a handful of values)."""
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2


def ellipse(k: int) -> np.ndarray:
    k = max(1, int(k))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


# --------------------------------------------------------------------------- reading

def segment_levels(patch: np.ndarray) -> dict[str, float]:
    """How lit each of the 7 segment positions of a tight glyph box is, from the glyph's
    background (0) to its brightest segment (1).

    A segment counts as lit as far as *both* halves of it are bright. The end of a
    perpendicular stroke (the foot of a narrow 7) brightens only one half.
    """
    gh, gw = patch.shape
    lv = {}
    for seg, (y0, y1, x0, x1) in _ZONES.items():
        z = patch[int(y0 * gh):max(int(round(y1 * gh)), int(y0 * gh) + 1),
                  int(x0 * gw):max(int(round(x1 * gw)), int(x0 * gw) + 1)]
        if z.size == 0:
            lv[seg] = 0.0
            continue
        if seg in "adg":
            k = max(1, z.shape[1] // 2)
            lv[seg] = float(min(z[:, :k].mean(), z[:, -k:].mean()))
        else:
            k = max(1, z.shape[0] // 2)
            lv[seg] = float(min(z[:k].mean(), z[-k:].mean()))
    # Blur and glow light up everything a little. The centres of the two "holes" of a digit
    # are dark in every digit, so take the dimmer one as the glyph's own background level.
    def mean(y0: float, y1: float, x0: float, x1: float) -> float:
        z = patch[int(y0 * gh):max(int(y1 * gh), int(y0 * gh) + 1),
                  int(x0 * gw):max(int(x1 * gw), int(x0 * gw) + 1)]
        return float(z.mean()) if z.size else 0.0
    floor = min(mean(0.22, 0.32, 0.4, 0.6), mean(0.68, 0.78, 0.4, 0.6), 0.3 * max(lv.values()))
    top = max(lv.values())
    if top <= floor:
        return {k: 0.0 for k in lv}
    return {k: float(np.clip((v - floor) / (top - floor), 0, 1)) for k, v in lv.items()}


def _digit_costs(patch: np.ndarray, h: float) -> dict[str, float]:
    """How badly each digit explains one glyph (0 = perfectly; missing = impossible).

    Wide glyphs are read from their segment levels: a digit costs (1 - level) for each
    segment it needs lit plus the level of each segment it needs dark. A narrow glyph can only
    be a "1": one vertical bar (sometimes two dots) spanning the digit height.
    """
    # The glyph box comes from a low threshold and includes the glow; fit the zones to the
    # strokes themselves.
    if patch is None or patch.size == 0 or patch.max() <= 0:
        return {}
    ys, xs = np.nonzero(patch > 0.4 * patch.max())
    patch = patch[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    gh, gw = patch.shape
    if gh < 0.6 * h:
        return {}
    if gw < 0.33 * h:
        prof = patch.mean(axis=1)
        filled = float((prof > 0.25 * prof.max()).mean())
        return {"1": 3 * (1 - filled)} if filled > 0.6 else {}
    lv = segment_levels(patch)
    cost: dict[str, float] = {}
    for segs, ch in _SEGMENTS.items():
        if ch == "1":
            continue
        c = sum(1 - lv[k] if k in segs else lv[k] for k in "abcdefg")
        cost[ch] = min(cost.get(ch, 99.0), c)
    return cost


def _best_char(cost: dict[str, float]) -> str:
    if not cost:
        return "?"
    ranked = sorted(cost.items(), key=lambda kv: kv[1])
    ch, c1 = ranked[0]
    c2 = ranked[1][1] if len(ranked) > 1 else c1 + 9
    return ch if c1 <= _MAX_COST and c2 - c1 >= _MIN_MARGIN else "?"


def _read(c: Candidate, sp: StampParams) -> None:
    """Decode the glyphs, split them into fields (at wide gaps and at the apostrophe) and find
    the most likely valid date.

    Single digits are often ambiguous on a blurred or partly cut stamp (6 or 8, 7 or 9). The
    date settles that: among the cheap readings of every glyph, take the cheapest combination
    that is a valid date in the camera-date era. If an apostrophe-like speck stands in the
    way of any valid date, the line is read once more without it.
    """
    for g in c.glyphs:
        if g.kind == "digit":
            g.costs = _digit_costs(g.patch, c.digit_h)
            g.char = _best_char(g.costs)
    for use_ticks in (True, False):
        groups, year_field = _fields(c, use_ticks)
        found = _best_date(groups, year_field, sp)
        if found is not None or not use_ticks:
            break
    if found is None:
        groups, year_field = _fields(c, True)
    c.year_field = year_field
    c.fields = ["".join(g.char for g in grp) for grp in groups]
    c.date = None
    if found is not None:
        c.date, c.fields, chars, c.date_alts = found
        for g, ch in zip([g for grp in groups for g in grp], chars):
            g.char = ch
    c.text = " ".join(("'" if i == year_field else "") + f for i, f in enumerate(c.fields))


def _fields(c: Candidate, use_ticks: bool) -> tuple[list[list[_Glyph]], int | None]:
    """Group the digits into fields: a new field after a gap or an apostrophe."""
    h = c.digit_h
    groups: list[list[_Glyph]] = []
    year_field = None
    prev: _Glyph | None = None
    for g in c.glyphs:
        if g.kind == "tick":
            if use_ticks:
                year_field = len(groups)
                prev = g
            continue
        if not groups or prev is None or prev.kind == "tick" or g.x0 - prev.x1 > 0.4 * h:
            groups.append([])
        groups[-1].append(g)
        prev = g
    if year_field is not None and year_field >= len(groups):
        year_field = None
    return groups, year_field


def _best_date(groups: list[list[_Glyph]], year_field: int | None, sp: StampParams
               ) -> tuple[str, list[str], list[str], list[str]] | None:
    """Cheapest reading of three digit fields that is a valid date.

    Returns (date, fields, chars, alts): alts are the next dates (up to 3) whose readings cost
    less than `alt_gap` more, i.e. the digits could almost as well be read that way.
    """
    flat = [g for grp in groups for g in grp]
    if len(groups) != 3 or any(len(grp) > 2 for grp in groups) or not all(g.costs for g in flat):
        return None
    options = []
    for g in flat:
        best = min(g.costs.values())
        options.append(sorted((v, ch) for ch, v in g.costs.items() if v <= best + _ALT_COST)[:3])
    by_date: dict[str, tuple[float, list[str], list[str]]] = {}
    for combo in itertools.product(*options):
        chars = iter(ch for _, ch in combo)
        fields = ["".join(next(chars) for _ in grp) for grp in groups]
        date = parse_stamp(fields, year_field, sp.year_pivot)
        if date is None or not sp.years[0] <= int(date[:4]) <= sp.years[1]:
            continue
        total = sum(v for v, _ in combo)
        if date not in by_date or total < by_date[date][0]:
            by_date[date] = (total, fields, [ch for _, ch in combo])
    if not by_date:
        return None
    ranked = sorted(by_date.items(), key=lambda kv: kv[1][0])
    date, (total, fields, chars) = ranked[0]
    alts = [other for other, (v, _, _) in ranked[1:4] if v - total < sp.alt_gap]
    return date, fields, chars, alts


def _score(c: Candidate, cw: int, sp: StampParams) -> float:
    """How stamp-like a read line is: enough well-fitting, evenly sized digits with clear warm
    contrast near the expected corner, and most of all, a valid date."""
    digits = [g for g in c.glyphs if g.kind == "digit"]
    read = sum(g.char != "?" for g in digits)
    if read < sp.min_digits or len(digits) > 8:
        return 0.0  # too little to be a date, or too much (a date has at most 8 digits)
    if sum(g.char not in "1?" for g in digits) < 2:
        return 0.0  # vertical stripes read as a row of ones
    hs = np.array([g.h for g in digits], np.float32)
    regular = max(0.0, 1.0 - 2.0 * float(hs.std() / max(hs.mean(), 1)))
    # How well the chosen digits explain the glyphs (cost 0 -> 1, cost 3 -> 0).
    fit = float(np.mean([max(0.0, 1 - g.costs.get(g.char, 3.0) / 3) for g in digits]))
    # Prior: the stamp sits at the right end of the bottom edge of the upright crop.
    cx = (c.box[0] + c.box[2]) / 2 / cw
    pos = 0.5 + 0.5 * float(np.clip((cx - 0.3) / 0.4, 0, 1))
    shape = 1.5 if c.date else (0.6 if len(c.fields) == 3 else 0.4)
    return fit * min(read, 6) / 3 * min(c.contrast, 3.0) * regular * pos * shape
