"""Camera date stamps on synthetic prints with known digits."""

import cv2
import numpy as np
import pytest

from albumsplit.model import DateStamp
from albumsplit.stamp import find_stamp, harmonize, parse_stamp

PAPER = (222, 232, 238)  # BGR, cream
ORANGE = (30, 110, 255)  # BGR of the LED light
PRINT = (300, 250, 1500, 1000)  # x, y, w, h on the page
H = 36  # digit height in px (~3.5 % of the print's short side, as on the samples)

# Segment end points in a unit digit box (x right, y down).
_SEG = {"a": ((0, 0), (1, 0)), "b": ((1, 0), (1, .5)), "c": ((1, .5), (1, 1)),
        "d": ((0, 1), (1, 1)), "e": ((0, .5), (0, 1)), "f": ((0, 0), (0, .5)),
        "g": ((0, .5), (1, .5))}
_DIGITS = {"0": "abcdef", "1": "bc", "2": "abdeg", "3": "abcdg", "4": "bcfg", "5": "acdfg",
           "6": "acdefg", "7": "abc", "8": "abcdefg", "9": "abcdfg"}


def stamp_mask(text: str, h: int = H, cell_gap: float = 0.25) -> np.ndarray:
    """Seven-segment rendering of e.g. "6 5 '95" (1.0 = lit), with a gap at each segment end.
    "_" is a digit cell left dark (a digit too faint to see). `cell_gap` (x h) is the space
    between digit cells, which differs between cameras."""
    w, t, gap = int(0.55 * h), max(2, int(0.13 * h)), 0.12
    width = int(len(text) * (w + 0.4 * h)) + h
    m = np.zeros((h + 2 * t, width), np.float32)
    x = t
    for ch in text:
        if ch == " ":
            x += int(0.9 * h)
        elif ch == "_":
            x += w + int(cell_gap * h)
        elif ch == "'":
            cv2.line(m, (x, t), (x - 2, t + int(0.3 * h)), 1.0, t)
            x += int(0.3 * h)
        else:
            for seg in _DIGITS[ch]:
                (ax, ay), (bx, by) = _SEG[seg]
                # shorten each segment a little so segments do not touch, like real LED digits
                sx, sy = ax + gap * (bx - ax), ay + gap * (by - ay)
                ex, ey = bx - gap * (bx - ax), by - gap * (by - ay)
                cv2.line(m, (x + int(sx * w), t + int(sy * h)), (x + int(ex * w), t + int(ey * h)),
                         1.0, t)
            x += w + int(cell_gap * h)
    m = m[:, :x + t]
    return cv2.GaussianBlur(m, (0, 0), 0.05 * h)  # a little optical glow


def make_print(seed: int, light_corner: bool = False, stripes: bool = False) -> np.ndarray:
    _, _, w, h = PRINT
    rng = np.random.default_rng(seed)
    small = rng.integers(30, 190, size=(h // 60 + 1, w // 60 + 1, 3), dtype=np.uint8)
    small[..., 2] = small[..., 2] // 2  # greenish/bluish scene: little red, like grass and sky
    img = cv2.GaussianBlur(cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC), (0, 0), 4)
    if light_corner:  # pale overcast sky where the stamp will be
        img[int(0.6 * h):, int(0.5 * w):] = (205, 200, 196)
    if stripes:  # red and white awning stripes plus a red barn wall: red, but not a stamp
        for i in range(0, 300, 30):
            img[int(0.75 * h):int(0.95 * h), w - 400 + i:w - 400 + i + 14] = (40, 40, 200)
        img[int(0.8 * h):int(0.97 * h), int(0.2 * w):int(0.45 * w)] = (45, 50, 150)
    noise = rng.normal(0, 3, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


def add_stamp(img: np.ndarray, text: str, alpha: float = 0.9, cell_gap: float = 0.25
              ) -> tuple[int, int, int, int]:
    """Burn the stamp into the bottom right of the print; returns its box (x0, y0, x1, y1)."""
    m = stamp_mask(text, cell_gap=cell_gap)
    h, w = img.shape[:2]
    x0, y0 = int(0.9 * w) - m.shape[1], int(0.93 * h) - m.shape[0]
    roi = img[y0:y0 + m.shape[0], x0:x0 + m.shape[1]].astype(np.float32)
    a = (alpha * m)[..., None]
    img[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = (roi * (1 - a) + np.array(ORANGE) * a).astype(np.uint8)
    return x0, y0, x0 + m.shape[1], y0 + m.shape[0]


def on_page(img: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float]]]:
    x, y, w, h = PRINT
    page = np.full((y + h + 300, x + w + 300, 3), PAPER, np.uint8)
    page[y:y + h, x:x + w] = img
    return page, [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def inside(st: DateStamp, box: tuple[int, int, int, int], tol: float = H) -> bool:
    xs = [p[0] for p in st.corners]
    ys = [p[1] for p in st.corners]
    x, y = PRINT[:2]
    return (abs(min(xs) - (box[0] + x)) < tol and abs(max(xs) - (box[2] + x)) < tol
            and abs(min(ys) - (box[1] + y)) < tol and abs(max(ys) - (box[3] + y)) < tol)


def test_reads_a_clear_stamp():
    img = make_print(1)
    box = add_stamp(img, "6 5 '95")
    page, corners = on_page(img)
    st = find_stamp(page, corners)
    assert st is not None and st.date == "1995-05-06"
    assert inside(st, box)


def test_year_first_stamp():
    img = make_print(2)
    add_stamp(img, "'93 7 14")
    st = find_stamp(*on_page(img))
    assert st is not None and st.date == "1993-07-14"


def test_faint_stamp_on_sky():
    img = make_print(3, light_corner=True)
    box = add_stamp(img, "12 5 '92", alpha=0.3)
    page, corners = on_page(img)
    st = find_stamp(page, corners)
    assert st is not None and st.date == "1992-05-12"
    assert inside(st, box)


def test_print_turned_on_the_page():
    """A portrait-held camera: the stamp runs vertically; the print may lie either way round."""
    img = make_print(4)
    add_stamp(img, "98 6 26")
    page, corners = on_page(img)
    for rot in (cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180):
        turned = cv2.rotate(page, rot)
        ph, pw = page.shape[:2]

        def move(p, rot=rot):
            x, y = p
            if rot == cv2.ROTATE_90_CLOCKWISE:
                return ph - 1 - y, x
            if rot == cv2.ROTATE_90_COUNTERCLOCKWISE:
                return y, pw - 1 - x
            return pw - 1 - x, ph - 1 - y

        pts = np.array([move(p) for p in corners], np.float32)
        # TL, TR, BR, BL of the turned print as the page shows it
        s, dd = pts.sum(axis=1), pts[:, 1] - pts[:, 0]
        quad = [tuple(pts[i]) for i in (np.argmin(s), np.argmin(dd), np.argmax(s), np.argmax(dd))]
        st = find_stamp(turned, quad)
        assert st is not None and st.date == "1998-06-26", rot


@pytest.mark.parametrize("seed", [5, 6])
def test_no_stamp_on_red_things(seed):
    st = find_stamp(*on_page(make_print(seed, stripes=True)))
    assert st is None


def test_parse_stamp_formats():
    assert parse_stamp(["6", "5", "95"], 2) == "1995-05-06"       # D M 'YY
    assert parse_stamp(["93", "7", "14"], 0) == "1993-07-14"      # 'YY M D
    assert parse_stamp(["98", "6", "26"], None) == "1998-06-26"   # YY M D, no apostrophe seen
    assert parse_stamp(["14", "5", "92"], None) == "1992-05-14"   # D M YY, no apostrophe seen
    assert parse_stamp(["5", "14", "92"], 2) == "1992-05-14"      # M D 'YY (14 is no month)
    assert parse_stamp(["3", "4", "02"], 2) == "2002-04-03"       # below the pivot: 20YY
    assert parse_stamp(["12", "5", "11"], None) is None           # which end is the year?
    assert parse_stamp(["31", "2", "95"], 2) is None              # no such day
    assert parse_stamp(["6", "5"], None) is None                  # incomplete
    assert parse_stamp(["6", "?", "95"], 2) is None               # unread digit


def test_harmonize_prefers_the_pages_date_for_a_near_tie():
    clear = DateStamp(corners=[], text="96 5 26", date="1996-05-26", score=4.6)
    blurred = DateStamp(corners=[], text="95 5 26", date="1995-05-26",
                        date_alts=["1995-05-25", "1996-05-26"], score=3.5)
    other = DateStamp(corners=[], text="8 5 '93", date="1993-05-08", date_alts=["1993-05-06"],
                      score=4.0)
    harmonize([blurred, clear, other])
    assert blurred.date == "1996-05-26" and "1995-05-26" in blurred.date_alts
    assert clear.date == "1996-05-26"
    assert other.date == "1993-05-08"  # its alternative matches nothing: unchanged


@pytest.mark.parametrize("text,date", [("'99 11 15", "1999-11-15"), ("6 11 '05", "2005-11-06"),
                                       ("12 12 '05", "2005-12-12")])
def test_ones_are_narrow_and_stay_in_their_field(text, date):
    """A "1" lights only the right side of its cell: it must not read as a wide digit, and
    the gap before it must not split "11" into two fields."""
    img = make_print(8)
    add_stamp(img, text)
    st = find_stamp(*on_page(img))
    assert st is not None and st.date == date



def _line(glyphs: list[tuple[str, int, int]], h: int = 34, transform=None) -> "Candidate":
    """A read-ready line from (char, x0, x1) glyphs ("'" = apostrophe), each digit's patch a
    rendered seven-segment digit (passed through `transform` if given)."""
    from albumsplit.stamp import Candidate, _Glyph
    gl = []
    for ch, x0, x1 in glyphs:
        if ch == "'":
            gl.append(_Glyph(x0, 0, x1, int(0.3 * h), "tick"))
            continue
        m = stamp_mask(ch, h)
        if transform is not None:
            m = transform(m)
        ys, xs = np.nonzero(m > 0.3)
        g = _Glyph(x0, 0, x1, h, "digit")
        g.patch = m[ys.min():ys.max() + 1, xs.min():xs.max() + 1] * 40
        gl.append(g)
    return Candidate(glyphs=gl, digit_h=float(h), box=(gl[0].x0, 0, gl[-1].x1, h), flipped=False,
                     contrast=2.0)


# Glyph positions measured on a real stamp (IMG_0055.tif): `21 6 '92`, digit height 34-35 px.
# This camera leaves a wide gap before the "1", as wide as a gap between fields.
_REAL_21_6_92 = [("2", 1167, 1190), ("1", 1213, 1221), ("6", 1267, 1290), ("'", 1303, 1311),
                 ("9", 1314, 1336), ("2", 1344, 1368)]


def test_widely_spaced_one_stays_in_its_field():
    from albumsplit.stamp import StampParams, _read
    c = _line(_REAL_21_6_92)
    _read(c, StampParams())
    assert c.date == "1992-06-21"


def test_missing_digit_gives_no_date():
    """With the faint 6 not found, the rest must not be read as `2 1 '92` = 1992-01-02:
    the gap where the 6 was has room for a whole digit."""
    from albumsplit.stamp import StampParams, _read
    c = _line([g for g in _REAL_21_6_92 if g[0] != "6"])
    _read(c, StampParams())
    assert c.date is None


def _shear(m: np.ndarray, slant: float) -> np.ndarray:
    """Lean a rendered stamp to the right (italic LED digits): the top moves right by
    slant * height / 2, the bottom left by as much."""
    h, w = m.shape
    pad = int(slant * h) + 2
    mat = np.float32([[1, -slant, slant * h / 2 + pad], [0, 1, 0]])
    return cv2.warpAffine(m, mat, (w + 2 * pad, h))


@pytest.mark.parametrize("slant", [0.1, 0.15, 0.2])
def test_italic_digits_are_read_upright(slant):
    """Some cameras' LED digits lean right, so segments fall outside upright sample zones (the
    lower right of an 8 looks dark). The line's slant is measured and taken out before
    reading."""
    from albumsplit.stamp import StampParams, _mean_cost, _read
    glyphs = [("9", 0, 24), ("8", 30, 54), ("8", 90, 114), ("9", 150, 174)]
    italic = _line(glyphs, transform=lambda m: _shear(m, slant))
    _read(italic, StampParams())
    assert italic.date == "1998-08-09"
    assert abs(italic.slant - slant) <= 0.05
    upright = _line(glyphs, transform=lambda m: _shear(m, slant))
    _read(upright, StampParams(slants=(0.0,)))
    assert _mean_cost(italic) < _mean_cost(upright)


def test_noisy_high_resolution_scan():
    """An uncompressed 600 dpi scan: the print is 2.5x larger than the search crop and every
    pixel carries its own colour noise (film grain, scanner). The crop must average it away
    rather than sample single pixels."""
    img = make_print(10, light_corner=True)
    add_stamp(img, "12 5 '92", alpha=0.45)
    page, corners = on_page(img)
    f = 2.5
    big = cv2.resize(page, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC).astype(np.float32)
    rng = np.random.default_rng(10)
    big += rng.normal(0, 20, big.shape).astype(np.float32)
    big = np.clip(big, 0, 255).astype(np.uint8)
    st = find_stamp(big, [(x * f, y * f) for x, y in corners])
    assert st is not None and st.date == "1992-05-12"
