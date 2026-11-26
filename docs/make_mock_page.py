"""Paint a mock album page for the README: procedurally drawn "prints", handwritten
captions and orange camera date stamps on off-white paper, at 600 dpi like a real scan.

    python docs/make_mock_page.py <out_dir>

No real photos are used. Fonts: Ink Free (Windows) for the handwriting.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

RNG = np.random.default_rng(7)
DPI = 600
MM = DPI / 25.4


def lerp(a, b, t):
    return np.asarray(a, np.float32) * (1 - t) + np.asarray(b, np.float32) * t


def vgrad(h, w, top, bottom):
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    return np.broadcast_to(lerp(top, bottom, t), (h, w, 3)).copy()


def ridge(w, base, amp, freq, seed):
    r = np.random.default_rng(seed)
    x = np.arange(w, dtype=np.float32)
    y = np.full(w, base, np.float32)
    for k in range(1, 5):
        y += amp / k * np.sin(x * freq * k + r.uniform(0, 6.3))
    return y


def fill_below(img, ys, color, alpha=1.0):
    h, w, _ = img.shape
    mask = (np.arange(h)[:, None] >= ys[None, :]).astype(np.float32)[..., None] * alpha
    img[:] = img * (1 - mask) + np.asarray(color, np.float32) * mask


def blob(img, center, axes, color, angle=0):
    cv2.ellipse(img, (int(center[0]), int(center[1])), (int(axes[0]), int(axes[1])), angle, 0, 360,
                tuple(float(c) for c in color), -1, cv2.LINE_AA)


def pine(img, x, y, h, color):
    w = h * 0.42
    for i in range(5):
        t = i / 5
        top = y - h + t * h * 0.75
        half = w * (0.35 + 0.65 * (i + 1) / 5) / 2
        pts = np.array([[x, top], [x - half, top + h * 0.3], [x + half, top + h * 0.3]], np.int32)
        cv2.fillPoly(img, [pts], tuple(float(c) for c in color), cv2.LINE_AA)
    cv2.rectangle(img, (int(x - w * 0.05), int(y - h * 0.1)), (int(x + w * 0.05), int(y)),
                  (60, 45, 35), -1)


def birch_crowns(img, ys_base, n, color, size, seed):
    r = np.random.default_rng(seed)
    h, w, _ = img.shape
    for _ in range(n):
        x = r.uniform(0, w)
        y = ys_base[min(int(x), w - 1)]
        c = np.asarray(color) * r.uniform(0.8, 1.15)
        blob(img, (x, y - size * 0.4), (size * r.uniform(0.5, 1.0), size * r.uniform(0.5, 0.9)), c)


def cottage(img, x, y, s):
    red, white, roof = (150, 35, 30), (235, 232, 225), (55, 50, 50)
    cv2.rectangle(img, (int(x), int(y - s * 0.6)), (int(x + s), int(y)), red, -1)
    roof_pts = np.array([[x - s * 0.08, y - s * 0.6], [x + s * 0.5, y - s * 0.95],
                         [x + s * 1.08, y - s * 0.6]], np.int32)
    cv2.fillPoly(img, [roof_pts], roof, cv2.LINE_AA)
    for wx in (0.15, 0.62):
        x0, y0 = int(x + s * wx), int(y - s * 0.45)
        cv2.rectangle(img, (x0, y0), (x0 + int(s * 0.22), y0 + int(s * 0.2)), white, -1)
        cv2.rectangle(img, (x0 + int(s * 0.03), y0 + int(s * 0.03)),
                      (x0 + int(s * 0.19), y0 + int(s * 0.17)), (40, 50, 70), -1)
    cv2.rectangle(img, (int(x + s * 0.42), int(y - s * 0.32)), (int(x + s * 0.56), int(y)), white, -1)
    for cx in (x, x + s):
        cv2.line(img, (int(cx), int(y - s * 0.6)), (int(cx), int(y)), white, max(2, int(s * 0.03)))


def scene_lake(h, w):
    img = vgrad(h, w, (120, 170, 220), (215, 225, 235))
    blob(img, (w * 0.78, h * 0.18), (h * 0.06, h * 0.06), (255, 245, 210))
    hills = ridge(w, h * 0.42, h * 0.04, 0.004, 1)
    fill_below(img, hills, (95, 125, 140))
    shore = ridge(w, h * 0.52, h * 0.02, 0.006, 2)
    birch_crowns(img, shore, 70, (55, 100, 45), h * 0.07, 3)
    fill_below(img, shore, (60, 95, 45))
    water = np.full(w, h * 0.56, np.float32)
    lake = vgrad(h, w, (70, 110, 150), (40, 70, 105))
    m = (np.arange(h)[:, None] >= water[None, :]).astype(np.float32)[..., None]
    img[:] = img * (1 - m) + lake * m
    for i in range(60):  # glints
        y = RNG.uniform(h * 0.58, h * 0.85)
        x = RNG.uniform(w * 0.5, w * 0.95)
        cv2.line(img, (int(x), int(y)), (int(x + RNG.uniform(20, 90)), int(y)), (200, 210, 220), 3)
    cottage(img, w * 0.18, h * 0.555, h * 0.2)
    jetty_y = int(h * 0.6)
    cv2.rectangle(img, (int(w * 0.3), jetty_y), (int(w * 0.48), jetty_y + int(h * 0.02)), (110, 85, 60), -1)
    return img


def scene_winter(h, w):
    img = vgrad(h, w, (175, 190, 210), (230, 230, 235))
    blob(img, (w * 0.25, h * 0.2), (h * 0.05, h * 0.05), (250, 235, 215))
    far = ridge(w, h * 0.5, h * 0.03, 0.005, 4)
    for i in range(45):
        x = RNG.uniform(0, w)
        pine(img, x, far[int(x)] + 10, h * RNG.uniform(0.15, 0.28), (40, 65, 55))
    snow = ridge(w, h * 0.6, h * 0.02, 0.003, 5)
    fill_below(img, snow, (238, 240, 245))
    shade = vgrad(h, w, (238, 240, 245), (200, 210, 228))
    m = (np.arange(h)[:, None] >= snow[None, :]).astype(np.float32)[..., None]
    img[:] = img * (1 - m) + shade * m
    for x in (w * 0.08, w * 0.9):
        pine(img, x, h * 0.95, h * 0.6, (30, 55, 45))
    # snowman
    cx = w * 0.55
    for cy, r in ((h * 0.82, h * 0.11), (h * 0.66, h * 0.08), (h * 0.535, h * 0.055)):
        blob(img, (cx + r * 0.1, cy + r * 0.1), (r, r), (190, 200, 220))
        blob(img, (cx, cy), (r, r), (250, 250, 252))
    for dx in (-0.018, 0.018):
        blob(img, (cx + w * dx * 0.5, h * 0.52), (h * 0.006, h * 0.006), (30, 30, 30))
    nose = np.array([[cx, h * 0.535], [cx + h * 0.06, h * 0.545], [cx, h * 0.55]], np.int32)
    cv2.fillPoly(img, [nose], (230, 120, 30), cv2.LINE_AA)
    cv2.rectangle(img, (int(cx - h * 0.04), int(h * 0.45)), (int(cx + h * 0.04), int(h * 0.485)),
                  (30, 30, 35), -1)
    cv2.rectangle(img, (int(cx - h * 0.06), int(h * 0.48)), (int(cx + h * 0.06), int(h * 0.492)),
                  (30, 30, 35), -1)
    for i in range(400):  # falling snow
        blob(img, (RNG.uniform(0, w), RNG.uniform(0, h * 0.6)), (4, 4), (250, 250, 252))
    return img


def scene_sea(h, w):
    img = vgrad(h, w, (90, 150, 215), (190, 215, 235))
    for i in range(5):
        cx, cy = RNG.uniform(0, w), RNG.uniform(h * 0.08, h * 0.3)
        for j in range(6):
            blob(img, (cx + j * h * 0.04, cy + RNG.uniform(-10, 10)),
                 (h * 0.05, h * 0.03), (245, 248, 250))
    horizon = int(h * 0.48)
    sea = vgrad(h, w, (40, 95, 150), (60, 140, 170))
    img[horizon:] = sea[horizon:]
    sand = ridge(w, h * 0.72, h * 0.03, 0.002, 6)
    fill_below(img, sand - 12, (230, 235, 235))  # surf line
    fill_below(img, sand, (220, 200, 160))
    # sailing boat
    bx, by = w * 0.62, h * 0.44
    hull = np.array([[bx - h * 0.08, by], [bx + h * 0.08, by], [bx + h * 0.06, by + h * 0.03],
                     [bx - h * 0.06, by + h * 0.03]], np.int32)
    cv2.fillPoly(img, [hull], (240, 240, 240), cv2.LINE_AA)
    sail = np.array([[bx, by - h * 0.2], [bx, by - h * 0.01], [bx + h * 0.08, by - h * 0.01]], np.int32)
    cv2.fillPoly(img, [sail], (245, 245, 240), cv2.LINE_AA)
    sail2 = np.array([[bx - 6, by - h * 0.17], [bx - 6, by - h * 0.01], [bx - h * 0.07, by - h * 0.01]], np.int32)
    cv2.fillPoly(img, [sail2], (215, 80, 60), cv2.LINE_AA)
    # beach ball and towel
    cv2.rectangle(img, (int(w * 0.12), int(h * 0.84)), (int(w * 0.34), int(h * 0.92)), (200, 60, 70), -1)
    for k in range(4):
        x = int(w * 0.12 + k * w * 0.055)
        cv2.rectangle(img, (x, int(h * 0.84)), (x + int(w * 0.02), int(h * 0.92)), (245, 240, 230), -1)
    blob(img, (w * 0.75, h * 0.86), (h * 0.05, h * 0.05), (240, 200, 40))
    cv2.ellipse(img, (int(w * 0.75), int(h * 0.86)), (int(h * 0.05), int(h * 0.05)), 0, 200, 340,
                (220, 50, 50), int(h * 0.025), cv2.LINE_AA)
    return img


def print_look(img):
    """Soft focus, warm fade and grain like a 90s colour print."""
    img = cv2.GaussianBlur(img, (0, 0), 2.2)
    img = img * np.array([1.02, 0.98, 0.9], np.float32) + np.array([8, 6, 0], np.float32)
    h, w, _ = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    vig = 1 - 0.18 * (((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) * 2.5)
    img = img * vig[..., None]
    grain = cv2.GaussianBlur(RNG.normal(0, 7, (h, w)).astype(np.float32), (0, 0), 1.0)
    return np.clip(img + grain[..., None], 0, 255)


# seven-segment layout: a b c d e f g
SEGS = {
    "0": "abcdef", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc", "5": "afgcd",
    "6": "afgedc", "7": "abc", "8": "abcdefg", "9": "abcdfg",
}


def stamp(img, text, x, y, digit_h):
    """Burn an orange seven-segment date into an RGB float image (text reads left to right)."""
    h, w, _ = img.shape
    layer = np.zeros((h, w), np.float32)
    dw, t = digit_h * 0.55, max(3, int(digit_h * 0.1))
    cx = x
    for ch in text:
        if ch == " ":
            cx += dw * 1.1
            continue
        if ch == "'":
            cv2.line(layer, (int(cx + dw * 0.2), int(y)), (int(cx + dw * 0.1), int(y + digit_h * 0.2)), 1.0, t)
            cx += dw * 0.55
            continue
        p = {"a": ((0, 0), (1, 0)), "b": ((1, 0), (1, 0.5)), "c": ((1, 0.5), (1, 1)),
             "d": ((0, 1), (1, 1)), "e": ((0, 0.5), (0, 1)), "f": ((0, 0), (0, 0.5)),
             "g": ((0, 0.5), (1, 0.5))}
        for s in SEGS[ch]:
            (ax, ay), (bx, by) = p[s]
            ax, bx = ax * 0.85 + 0.075, bx * 0.85 + 0.075
            shrink = 0.2
            sx0 = cx + dw * (ax + (bx - ax) * shrink)
            sx1 = cx + dw * (bx - (bx - ax) * shrink)
            sy0 = y + digit_h * (ay + (by - ay) * shrink)
            sy1 = y + digit_h * (by - (by - ay) * shrink)
            cv2.line(layer, (int(sx0), int(sy0)), (int(sx1), int(sy1)), 1.0, t, cv2.LINE_AA)
        cx += dw * 1.45
    glow = cv2.GaussianBlur(layer, (0, 0), digit_h * 0.08)
    core = cv2.GaussianBlur(layer, (0, 0), 1.5)
    a = np.clip(core * 0.9 + glow * 0.6, 0, 1)[..., None]
    orange = np.array([255, 150, 50], np.float32)
    img[:] = img * (1 - a) + orange * a


def paper(h, w):
    base = np.full((h, w, 3), (238, 236, 228), np.float32)
    fib = cv2.GaussianBlur(RNG.normal(0, 5, (h // 4, w // 4)).astype(np.float32), (0, 0), 2)
    fib = cv2.resize(fib, (w, h))
    return base + fib[..., None]


def paste(page, photo, center, angle):
    """Rotate a print by `angle` degrees and lay it on the page with a faint shadow."""
    ph, pw, _ = photo.shape
    m = cv2.getRotationMatrix2D((pw / 2, ph / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(pw * cos + ph * sin) + 4, int(pw * sin + ph * cos) + 4
    m[0, 2] += nw / 2 - pw / 2
    m[1, 2] += nh / 2 - ph / 2
    warped = cv2.warpAffine(photo, m, (nw, nh), flags=cv2.INTER_LINEAR)
    alpha = cv2.warpAffine(np.ones((ph, pw), np.float32), m, (nw, nh))[..., None]
    x0, y0 = int(center[0] - nw / 2), int(center[1] - nh / 2)
    region = page[y0:y0 + nh, x0:x0 + nw]
    shadow = cv2.GaussianBlur(alpha[..., 0], (0, 0), 6)[..., None]
    shadow = np.roll(np.roll(shadow, 6, 0), 6, 1)
    region[:] = region * (1 - 0.25 * shadow)
    region[:] = region * (1 - alpha) + warped * alpha


def handwrite(page, lines, x, y, size, angle, color=(30, 35, 80)):
    font = ImageFont.truetype("C:/Windows/Fonts/Inkfree.ttf", size)
    pad = size
    w = int(max(font.getlength(s) for s in lines)) + 2 * pad
    h = int(len(lines) * size * 1.25) + 2 * pad
    im = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(im)
    for i, s in enumerate(lines):
        d.text((pad + i * size * 0.15, pad + i * size * 1.25), s, font=font, fill=255,
               stroke_width=max(1, size // 45), stroke_fill=255)
    im = im.rotate(angle, resample=Image.BICUBIC, expand=True)
    a = np.asarray(im, np.float32)[..., None] / 255 * 0.92
    hh, ww = a.shape[:2]
    region = page[y:y + hh, x:x + ww]
    region[:] = region * (1 - a) + np.asarray(color, np.float32) * a


def main(out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    H, W = int(250 * MM), int(300 * MM)
    page = paper(H, W)
    ph, pw = int(89 * MM), int(127 * MM)  # 9x13 cm prints

    prints = [
        (scene_lake(ph, pw), "'93 7 14", (W * 0.265, H * 0.25), -1.2),
        (scene_winter(ph, pw), "98 12 26", (W * 0.74, H * 0.27), 0.8),
        (scene_sea(ph, pw), "6 5 '95", (W * 0.33, H * 0.73), 0.4),
    ]
    for img, date, center, angle in prints:
        img = print_look(img)
        dh = int(2.8 * MM)
        n = len(date.replace(" ", "").replace("'", ""))
        stamp(img, date, pw - int(n * 1.45 * dh * 0.55 + 18 * MM), ph - int(7 * MM), dh)
        paste(page, img, center, angle)

    size = int(6.5 * MM)
    handwrite(page, ["Summer cottage by", "the lake, July -93"], int(W * 0.11), int(H * 0.45), size, 1.5)
    handwrite(page, ["Boxing Day -98.", "Our snowman in the yard"], int(W * 0.55), int(H * 0.47), size, -1)
    handwrite(page, ["Midsummer at the beach", "with Lisa and Erik", "1995"], int(W * 0.6), int(H * 0.64),
              size, 2)

    out = out_dir / "Album1" / "IMG_0001.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    rgb = np.clip(page, 0, 255).astype(np.uint8)
    cv2.imwrite(str(out), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return out


if __name__ == "__main__":
    print(main(Path(sys.argv[1] if len(sys.argv) > 1 else "mock_scans")))
