import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from albumsplit.__main__ import main
from albumsplit.exiftool import exif_datetime, find_exiftool, read_tags, tag_args, xmp_date
from albumsplit.export import card, export_work_dir, photo_crop
from albumsplit.model import Link, Page, Photo, TextBlock
from albumsplit.pipeline import page_json_path, save_input_dir

RED, BLUE = (0, 0, 220), (220, 0, 0)  # BGR


def _quad(cx, cy, w, h, deg):
    a = np.deg2rad(deg)
    u, v = np.array([np.cos(a), np.sin(a)]), np.array([-np.sin(a), np.cos(a)])
    c = np.array([cx, cy])
    return [tuple(map(float, c + sx * w / 2 * u + sy * h / 2 * v))
            for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]


def _scene(tmp_path: Path, status="approved") -> tuple[Path, Path, Page, np.ndarray]:
    """White page: a slightly rotated red print, a blue print right of it, a caption under red."""
    img = np.full((600, 800, 3), 255, np.uint8)
    red = _quad(220, 200, 300, 200, 3)
    blue = Photo(id=2, corners=[(420, 60), (700, 60), (700, 360), (420, 360)])
    cv2.fillPoly(img, [np.round(red).astype(np.int32)], RED)
    cv2.rectangle(img, (420, 60), (700, 360), BLUE, cv2.FILLED)
    cv2.putText(img, "Pask -84", (90, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (20, 20, 20), 3)
    scans, work = tmp_path / "scans", tmp_path / "work"
    scans.mkdir()
    cv2.imwrite(str(scans / "page.png"), img)
    # The caption box runs into the blue print's area margin; it must not show any blue.
    page = Page(page_id="page", source="page.png", width=800, height=600, status=status,
                photos=[Photo(id=1, corners=red), blue],
                texts=[TextBlock(id=1, box=(80, 360, 340, 60), transcription="Påsk\nhos mormor",
                                 date="1984-04")],
                links=[Link(text_id=1, photo_id=1)])
    page.save(page_json_path(work, "page"))
    save_input_dir(work, scans)
    return scans, work, page, img


def test_photo_crop_size_corners_outline_rotation(tmp_path):
    _, _, page, img = _scene(tmp_path)
    red = page.photos[0]
    crop = photo_crop(img, red)
    assert abs(crop.shape[1] - 300) <= 2 and abs(crop.shape[0] - 200) <= 2
    for y, x in ((4, 4), (4, -5), (-5, 4), (-5, -5)):  # corners are photo, not paper
        assert np.abs(crop[y, x].astype(int) - RED).max() < 60

    red.outline = [red.corners[0], red.corners[1], red.corners[2]]  # lower-left half hidden
    hidden = photo_crop(img, red)
    assert (hidden[-5, 4] == 255).all() and np.abs(hidden[4, -5].astype(int) - RED).max() < 60

    red.outline, red.rotation = None, 90
    turned = photo_crop(img, red)
    assert turned.shape[:2] == crop.shape[1::-1]


def test_card_shows_caption_but_not_neighbour(tmp_path):
    _, _, page, img = _scene(tmp_path)
    red = page.photos[0]
    crop = photo_crop(img, red)
    c = card(img, page, red, crop)
    assert c.shape[0] > crop.shape[0] + 40 and c.shape[1] > crop.shape[1]
    margin = (c.shape[1] - crop.shape[1]) // 2
    strip = c[margin + crop.shape[0]:]
    assert (strip.min(axis=2) < 80).any()  # dark handwriting is there
    b, g, r = (strip[..., i].astype(int) for i in range(3))
    assert not ((b - r > 100) & (b - g > 100)).any()  # no blue from the neighbouring print


def test_tag_args_dates():
    assert exif_datetime("1984-12") == "1984:12:01 00:00:00"
    assert exif_datetime("1984") == "1984:01:01 00:00:00"
    assert exif_datetime("1984-12-26") == "1984:12:26 00:00:00"
    assert exif_datetime("jul -84") is None
    assert xmp_date("1984-12") == "1984:12"
    args = tag_args("Påsk", "1984-12")
    assert "-EXIF:DateTimeOriginal=1984:12:01 00:00:00" in args
    assert "-XMP-photoshop:DateCreated=1984:12" in args
    assert "-XMP-dc:Description=Påsk" in args
    assert tag_args(None, None) == []


def test_find_exiftool_uses_bundled_perl_for_k_exe(tmp_path, monkeypatch):
    folder = tmp_path / "exiftool-13.64_64"
    (folder / "exiftool_files").mkdir(parents=True)
    for f in ("exiftool(-k).exe", "exiftool_files/perl.exe", "exiftool_files/exiftool.pl"):
        (folder / f).write_bytes(b"")
    monkeypatch.delenv("ALBUMSPLIT_EXIFTOOL", raising=False)
    assert find_exiftool(str(folder)) == [str(folder / "exiftool_files" / "perl.exe"),
                                          str(folder / "exiftool_files" / "exiftool.pl")]
    assert find_exiftool(str(folder / "exiftool(-k).exe"))[1].endswith("exiftool.pl")
    monkeypatch.setenv("ALBUMSPLIT_EXIFTOOL", str(folder))
    assert find_exiftool()[0].endswith("perl.exe")


def test_export_skips_proposed_and_writes_index(tmp_path):
    scans, work, _, _ = _scene(tmp_path, status="proposed")
    out = tmp_path / "out"
    rep = export_work_dir(work, out, scans)
    assert rep.photos == 0 and rep.skipped == 1 and not out.exists()

    rep = export_work_dir(work, out, scans, include_proposed=True)
    assert rep.pages == 1 and rep.photos == 2 and not rep.failed
    names = sorted(p.name for p in out.iterdir())
    assert names == ["index.csv", "page_1.jpg", "page_1_card.jpg", "page_2.jpg", "page_2_card.jpg"]
    raw = (out / "index.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    rows = list(csv.DictReader(raw.decode("utf-8-sig").splitlines()))
    assert rows[0]["caption"] == "Påsk hos mormor" and rows[0]["date"] == "1984-04"
    assert rows[1]["caption"] == "" and rows[1]["file"] == "page_2.jpg"


def test_export_removes_stale_files(tmp_path):
    scans, work, page, _ = _scene(tmp_path)
    out = tmp_path / "out"
    export_work_dir(work, out, scans, cards=False)
    page.photos = page.photos[:1]
    page.save(page_json_path(work, "page"))
    export_work_dir(work, out, scans, cards=False)
    assert sorted(p.name for p in out.glob("*.jpg")) == ["page_1.jpg"]


@pytest.mark.skipif(find_exiftool() is None, reason="exiftool not installed")
def test_metadata_roundtrip(tmp_path):
    scans, work, page, _ = _scene(tmp_path)
    page.photos[0].caption = "Påsk hos\nmormor i Örebro"
    page.save(page_json_path(work, "page"))
    out = tmp_path / "ut åäö"
    assert main(["export", str(work), "--out", str(out)]) == 0
    tool = find_exiftool()
    for name in ("page_1.jpg", "page_1_card.jpg"):
        tags = read_tags(tool, out / name)
        assert tags["ExifIFD:DateTimeOriginal"] == "1984:04:01 00:00:00"
        assert tags["IFD0:XPComment"] == "Påsk hos\nmormor i Örebro"
        assert tags["XMP-dc:Description"] == "Påsk hos\nmormor i Örebro"
        assert tags["XMP-photoshop:DateCreated"] == "1984:04"
    assert "XMP-dc:Description" not in read_tags(tool, out / "page_2.jpg")
