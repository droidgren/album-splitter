"""Image loading helpers (read-only access to source scans)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def load_bgr(path: Path) -> np.ndarray:
    """Load an image as a BGR uint8 array with EXIF orientation applied."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        rgb = np.asarray(im.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


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
