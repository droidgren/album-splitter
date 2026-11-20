"""Image loading helpers (read-only access to source scans)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Uncompressed scans are big: an A4 page at 1200 dpi is ~140 MP, which Pillow would refuse
# as a "decompression bomb" at its default limit (~89 MP). Source scans are trusted input.
Image.MAX_IMAGE_PIXELS = 1_000_000_000


def load_bgr(path: Path) -> np.ndarray:
    """Load an image as a BGR uint8 array with EXIF orientation applied.

    Handles 8- and 16-bit JPEG/PNG/TIFF in RGB, grayscale, palette and CMYK. Multi-page
    TIFFs use the first page.
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        rgb = _to_rgb8(im)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _to_rgb8(im: Image.Image) -> np.ndarray:
    # Pillow's convert("RGB") clips 16-bit grayscale (modes I;16*, I) instead of scaling it.
    if im.mode.startswith("I"):
        a = np.asarray(im)
        if a.dtype.itemsize > 1:
            a = (np.clip(a, 0, 65535).astype(np.uint32) * 255 + 32767) // 65535
        g = a.astype(np.uint8)
        return np.dstack([g, g, g])
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        # Composite on white paper rather than letting transparent areas turn black.
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return np.asarray(im.convert("RGB"))


def downscale(img: np.ndarray, long_side: int) -> tuple[np.ndarray, float]:
    """Resize so the longest side is `long_side`. Returns (image, scale) where full = small / scale."""
    h, w = img.shape[:2]
    scale = min(1.0, long_side / max(h, w))
    if scale >= 1.0:
        return img, 1.0
    small = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return small, scale


def find_images(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
