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


def test_caption_date_edited_roundtrip_and_old_json(tmp_path: Path):
    page = Page(page_id="p", source="p.jpg", width=10, height=10, edited=True,
                photos=[Photo(id=1, corners=[(0, 0), (1, 0), (1, 1), (0, 1)], caption="Påsk", date="1984")])
    page.save(tmp_path / "p.json")
    assert Page.load(tmp_path / "p.json") == page
    old = Page.from_dict({"page_id": "p", "source": "p.jpg", "width": 10, "height": 10,
                          "photos": [{"id": 1, "corners": [[0, 0], [1, 0], [1, 1], [0, 1]]}]})
    assert old.edited is False and old.photos[0].caption is None and old.photos[0].date is None


def test_effective_caption_and_date():
    from albumsplit.model import effective_caption, effective_date
    sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
    ph = Photo(id=1, corners=sq)
    page = Page(page_id="p", source="p.jpg", width=10, height=10, photos=[ph],
                texts=[TextBlock(id=2, box=(0, 0, 1, 1), transcription="Mormor", date="1984-12-26"),
                       TextBlock(id=1, box=(0, 0, 1, 1), transcription="Annandag jul -84"),
                       TextBlock(id=3, box=(0, 0, 1, 1), transcription="not linked", date="1990")],
                links=[Link(2, 1), Link(1, 1)])
    assert effective_caption(page, ph) == "Annandag jul -84 Mormor"  # reading order, one line
    assert effective_date(page, ph) == "1984-12-26"
    ph.stamp = DateStamp(corners=sq, text="98 6 26", date="1998-06-26")
    assert effective_date(page, ph) == "1998-06-26"  # camera stamp first
    ph.caption, ph.date = "Jul hos mormor", "1984-12"
    assert effective_caption(page, ph) == "Jul hos mormor" and effective_date(page, ph) == "1984-12"
    ph.caption, ph.date = "", ""  # cleared on purpose
    assert effective_caption(page, ph) is None and effective_date(page, ph) is None


def test_flatten_caption():
    from albumsplit.model import flatten_caption
    assert flatten_caption("Karin högt\nuppe på\nsvinhus-\ntaket,") == "Karin högt uppe på svinhustaket,"
    assert flatten_caption("Karl-\nJohan") == "Karl-Johan"
    assert flatten_caption("Påsk\n-84") == "Påsk -84"
    assert flatten_caption("  Midsommar  \n\n 1990 ") == "Midsommar 1990"
    # The transcription itself keeps its line breaks; only the derived caption is flattened.
    ph = Photo(id=1, corners=[(0, 0), (1, 0), (1, 1), (0, 1)])
    page = Page(page_id="p", source="p.jpg", width=10, height=10, photos=[ph],
                texts=[TextBlock(id=1, box=(0, 0, 1, 1), transcription="Titta så fint\nguldklimpen!")],
                links=[Link(1, 1)])
    from albumsplit.model import derived_caption
    assert derived_caption(page, ph) == "Titta så fint guldklimpen!"
    assert page.texts[0].transcription == "Titta så fint\nguldklimpen!"
