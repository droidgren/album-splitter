"""Date guesses from the album order: timeline, segments, answers and what they may change."""

from pathlib import Path

import numpy as np
import pytest

from albumsplit import dates
from albumsplit.dates import (AlbumPage, albums, apply_guesses, build_requests, far_from,
                              guess_dates, natural_key, needs_guess, parse_answer)
from albumsplit.model import (DateGuess, DateStamp, Link, Page, Photo, TextBlock, date_source,
                              derived_date, effective_date)


def quad(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _page(name: str, *photo_dates: str | None, folder="Album8", caption=None) -> Page:
    """A page whose photos have these stamp dates (None = undated)."""
    photos = [Photo(id=i, corners=quad(100 * i, 0, 100 * i + 90, 60),
                    stamp=DateStamp(corners=quad(0, 40, 30, 50), text="x", date=d) if d else None)
              for i, d in enumerate(photo_dates, 1)]
    page = Page(page_id=f"{folder}__{name}", source=f"{folder}/{name}.png", width=500,
                height=200, photos=photos)
    if caption:
        page.texts = [TextBlock(id=1, box=(0, 100, 50, 20), transcription=caption)]
        page.links = [Link(text_id=1, photo_id=1)]
    return page


def _ap(page: Page) -> AlbumPage:
    return AlbumPage(Path(f"{page.page_id}.json"), page)


def test_natural_order():
    names = ["IMG_10.png", "IMG_2.png", "IMG_1.png"]
    assert sorted(names, key=natural_key) == ["IMG_1.png", "IMG_2.png", "IMG_10.png"]


def test_derived_date_falls_back_to_the_guess_last():
    page = _page("IMG_0001", None)
    ph = page.photos[0]
    ph.date_guess = DateGuess(date="1995-05", reason="r", model="m")
    assert derived_date(page, ph) == "1995-05" and date_source(page, ph) == "guess"
    page.texts = [TextBlock(id=1, box=(0, 0, 1, 1), date="1995-06-24")]
    page.links = [Link(1, 1)]
    assert derived_date(page, ph) == "1995-06-24" and date_source(page, ph) == "caption"
    ph.date = "1996"
    assert effective_date(page, ph) == "1996" and date_source(page, ph) == "review"
    ph.date = ""
    assert effective_date(page, ph) is None and date_source(page, ph) is None


def test_date_guess_round_trips():
    page = _page("IMG_0001", None)
    page.photos[0].date_guess = DateGuess(date="1995", reason="r", confidence="low", model="m")
    again = Page.from_dict(page.to_dict())
    assert again.photos[0].date_guess == page.photos[0].date_guess


def test_timeline_marks_photos_to_date_and_known_dates():
    pages = [_ap(_page("IMG_0001", "1995-05-06")), _ap(_page("IMG_0002", None, "1995-05-20")),
             _ap(_page("IMG_0003", None, caption="Midsommar\npå landet"))]
    [req] = build_requests("Album8", pages, "m")
    assert [(t.gid, t.page_id, t.photo_id) for t in req.targets] == [
        (1, "Album8__IMG_0002", 1), (2, "Album8__IMG_0003", 1)]
    assert "IMG_0001.png:\n  P1: 1995-05-06 (camera stamp)" in req.text
    assert "P1: G1, date unknown" in req.text
    assert 'caption "Midsommar på landet"' in req.text
    assert req.refs["IMG_0002 P2"] == ("Album8__IMG_0002", 2)
    assert req.refs["IMG_0002.png P2"] == ("Album8__IMG_0002", 2)


def test_digest_ignores_earlier_guesses():
    pages = [_ap(_page("IMG_0001", "1995-05-06")), _ap(_page("IMG_0002", None))]
    [a] = build_requests("Album8", pages, "m")
    pages[1].page.photos[0].date_guess = DateGuess(date="1995", model="m")
    [b] = build_requests("Album8", pages, "m")
    assert a.digest == b.digest


def test_context_pages_do_not_show_earlier_guesses(monkeypatch):
    monkeypatch.setattr(dates, "MAX_GUESSES_PER_REQUEST", 1)
    pages = [_ap(_page("IMG_0001", None)), _ap(_page("IMG_0002", None))]
    pages[0].page.photos[0].date_guess = DateGuess(date="1995", model="m")
    first, second = build_requests("Album8", pages, "m")
    assert "IMG_0001.png (context):\n  P1: date unknown" in second.text
    assert "1995" not in second.text


def test_long_albums_are_split_with_context(monkeypatch):
    monkeypatch.setattr(dates, "MAX_GUESSES_PER_REQUEST", 4)
    monkeypatch.setattr(dates, "CONTEXT_PAGES", 1)
    pages = [_ap(_page(f"IMG_{i:04d}", None, None)) for i in range(1, 6)]
    reqs = build_requests("Album8", pages, "m")
    assert [len(r.core) for r in reqs] == [2, 2, 1]
    assert "IMG_0003.png (context)" in reqs[0].text and "IMG_0004.png" not in reqs[0].text
    assert "IMG_0002.png (context)" in reqs[1].text
    assert all(1 <= t.gid <= 4 for r in reqs for t in r.targets)


def test_parse_and_apply_answer():
    pages = [_ap(_page("IMG_0001", "1995-05-06", "1994-01-07")), _ap(_page("IMG_0002", None, None))]
    [req] = build_requests("Album8", pages, "m")
    data = {"guesses": [{"id": 1, "date": "1995-05", "confidence": "medium", "reason": "between"},
                        {"id": 2, "date": "not a date", "confidence": "high", "reason": "x"},
                        {"id": 9, "date": "1995", "confidence": "low", "reason": "unknown id"}],
            "suspects": [{"ref": "IMG_0001.png P2", "note": "1994 among 1995"},
                         {"ref": "IMG_0002 P1", "note": "undated photo cannot be a suspect"},
                         {"ref": "nope", "note": ""}]}
    ans = parse_answer(req, data, "m")
    assert set(ans.guesses) == {("Album8__IMG_0002", 1)}
    p1, p2 = pages[0].page, pages[1].page
    assert apply_guesses(p1, ans) and apply_guesses(p2, ans)
    assert p2.photos[0].date_guess.date == "1995-05" and p2.photos[1].date_guess is None
    [issue] = p1.issues
    assert issue.kind == "date_out_of_order" and issue.photo_ids == [2] and issue.source == "dates"
    assert p1.photos[1].needs_review and not p2.issues
    assert not apply_guesses(p1, ans)  # the same answer again changes nothing


def test_guess_never_replaces_a_known_date():
    page = _page("IMG_0001", "1995-05-06", None)
    page.photos[1].date = "1996"
    assert not needs_guess(page, page.photos[0]) and not needs_guess(page, page.photos[1])
    page.photos[0].date_guess = DateGuess(date="1990", model="m")  # stale
    key = {(page.page_id, 1): DateGuess(date="1999", model="m"),
           (page.page_id, 2): DateGuess(date="1999", model="m")}
    apply_guesses(page, dates.DatesAnswer(key, {}))
    assert page.photos[0].date_guess is None and page.photos[1].date_guess is None


def test_llm_page_answer_keeps_date_issues():
    from albumsplit.llm import LlmResult, apply
    from albumsplit.model import Issue
    page = _page("IMG_0001", "1995-05-06")
    page.issues = [Issue(kind="date_out_of_order", photo_ids=[1], source="dates")]
    apply(page, LlmResult(texts=[], issues=[], model="m", cached=False))
    assert [i.source for i in page.issues] == ["dates"]


class FakeLinker:
    model = tag = "m"
    effort = None

    def __init__(self, answer):
        self.answer, self.calls = answer, 0

    def _call(self, page_id, content, system, schema):
        self.calls += 1
        return self.answer


def test_pass_skips_approved_pages_and_uses_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(dates, "load_bgr", lambda p: np.full((200, 500, 3), 128, np.uint8))
    pages = [_page("IMG_0001", "1995-05-06"), _page("IMG_0002", None), _page("IMG_0003", None)]
    pages[2].status = "approved"
    for p in pages:
        p.save(tmp_path / "pages" / f"{p.page_id}.json")
    _page("IMG_0001", None, folder="Other").save(tmp_path / "pages" / "Other__IMG_0001.json")
    assert list(albums(tmp_path)) == ["Album8", "Other"]
    # G1 = IMG_0002 P1, G2 = IMG_0003 P1 (approved: context only, never changed)
    answer = {"guesses": [{"id": 1, "date": "1995-05", "confidence": "high", "reason": "r"},
                          {"id": 2, "date": "1995-06", "confidence": "high", "reason": "r"}],
              "suspects": []}
    llm = FakeLinker(answer)
    rep = guess_dates(llm, tmp_path, tmp_path, log=lambda m: None, only_albums={"Album8"})
    assert llm.calls == 1 and rep.called == 1 and rep.pages_updated == 1
    p2 = Page.load(tmp_path / "pages" / "Album8__IMG_0002.json")
    p3 = Page.load(tmp_path / "pages" / "Album8__IMG_0003.json")
    assert p2.photos[0].date_guess.date == "1995-05" and p3.photos[0].date_guess is None
    rep = guess_dates(llm, tmp_path, tmp_path, log=lambda m: None, only_albums={"Album8"})
    assert llm.calls == 1 and rep.cached == 1 and rep.pages_updated == 0


@pytest.mark.parametrize("folder,expect", [("", "_top"), ("2023/album 1", "2023__album_1")])
def test_cache_key_slug(folder, expect):
    pages = [_ap(_page("IMG_0001", None, folder=folder))]
    for ap in pages:
        ap.page.source = f"{folder}/IMG_0001.png".lstrip("/")
    [req] = build_requests(folder, pages, "m")
    assert req.key.startswith(expect + "__")


def test_far_from():
    assert far_from("1994-01-07", ["2002-03-01", "2002-05"])
    assert not far_from("2001-07-04", ["2001-06-28", "2001-06-06"])
    assert not far_from("2002", ["2002-06-01"])  # a year overlapping the range
    assert far_from("1999-01", [])


def test_suspect_close_to_its_neighbours_is_dropped():
    pages = [_ap(_page("IMG_0001", "2001-06-28")), _ap(_page("IMG_0002", "2001-07-04", "1994-01-07")),
             _ap(_page("IMG_0003", "2001-07-10"))]
    [req] = build_requests("Album8", pages, "m")
    data = {"guesses": [], "suspects": [{"ref": "IMG_0002 P1", "note": "a week off"},
                                        {"ref": "IMG_0002 P2", "note": "years off"}]}
    ans = parse_answer(req, data, "m", {ap.page.page_id: ap.page for ap in pages})
    assert ans.suspects == {"Album8__IMG_0002": [(2, "years off")]}
