import numpy as np
from PIL import Image

from albumsplit.imageio import load_bgr


def test_load_16bit_gray_png(tmp_path):
    a = np.array([[0, 32896, 65535]], dtype=np.uint16)
    p = tmp_path / "g16.png"
    Image.fromarray(a).save(p)
    img = load_bgr(p)
    assert img.dtype == np.uint8 and img.shape == (1, 3, 3)
    assert img[0, :, 0].tolist() == [0, 128, 255]


def test_load_16bit_rgb_tiff(tmp_path):
    import cv2
    a = np.zeros((2, 2, 3), dtype=np.uint16)
    a[0, 0] = [65535, 0, 0]  # BGR: pure blue
    p = tmp_path / "c16.tif"
    cv2.imencode(".tif", a)[1].tofile(str(p))
    img = load_bgr(p)
    assert img.dtype == np.uint8
    assert img[0, 0].tolist() == [255, 0, 0]


def test_load_rgba_on_white(tmp_path):
    p = tmp_path / "a.png"
    Image.new("RGBA", (2, 2), (0, 0, 0, 0)).save(p)
    assert load_bgr(p)[0, 0].tolist() == [255, 255, 255]
