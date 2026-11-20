from pathlib import Path, PurePosixPath, PureWindowsPath

from albumsplit.model import DateStamp, Link, Page, Photo, TextBlock, page_id_from_path


def test_page_id_is_deterministic_and_safe():
    assert page_id_from_path("IMG_0012.jpg") == "IMG_0012"
    assert page_id_from_path(PurePosixPath("2023/album 1/IMG_0012.jpg")) == "2023__album_1__IMG_0012"
    assert page_id_from_path(PureWindowsPath(r"2023\album 1\IMG_0012.jpg")) == "2023__album_1__IMG_0012"
    assert page_id_from_path("Påsk 1984/Mormor & Morfar.jpg") == "Pask_1984__Mormor_Morfar"
    assert page_id_from_path("#/!.jpg") == "____"  # nothing usable left -> placeholders


def test_json_roundtrip(tmp_path: Path):
    page = Page(
        page_id="p", source="p.jpg", width=100, height=200,
        photos=[Photo(id=1, corners=[(0, 0), (10, 0), (10, 10), (0, 10)], needs_review=True,
                      stamp=DateStamp(corners=[(6, 8), (9, 8), (9, 9), (6, 9)], text="6 5 '95",
                                      date="1995-05-06", date_alts=["1995-05-08"], score=3.2)),
                Photo(id=2, corners=[(5, 5), (30, 5), (30, 30), (5, 30)],
                      outline=[(10, 5), (30, 5), (30, 30), (5, 30), (5, 10), (10, 10)])],
        texts=[TextBlock(id=1, box=(20, 0, 5, 5), transcription="Annandag jul -84", date="1984-12-26")],
        links=[Link(text_id=1, photo_id=1)],
        page_quad=[(0, 0), (99, 0), (99, 199), (0, 199)],
    )
    path = tmp_path / "pages" / "p.json"
    page.save(path)
    loaded = Page.load(path)
    assert loaded == page
    assert loaded.status == "proposed"
    assert not list(path.parent.glob("*.tmp"))


def test_assign_page_ids_disambiguates_formats():
    from albumsplit.model import assign_page_ids
    paths = [PurePosixPath(p) for p in
             ["IMG_0055.tif", "IMG_0055.jpg", "IMG_0055.png", "IMG_0008.png", "a/IMG_0008.tif"]]
    ids = assign_page_ids(paths)
    assert ids[paths[1]] == "IMG_0055"
    assert ids[paths[0]] == "IMG_0055_tif"
    assert ids[paths[2]] == "IMG_0055_png"
    assert ids[paths[3]] == "IMG_0008"
    assert ids[paths[4]] == "a__IMG_0008"
