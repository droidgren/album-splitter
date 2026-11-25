"""Claude's page answer: checking the OCR's date stamp readings."""

import numpy as np

from albumsplit.llm import LlmLinker, LlmResult, StampRead, _validate_stamps, apply
from albumsplit.model import DateStamp, Page, Photo


def quad(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _page(*stamps: DateStamp | None) -> Page:
    photos = [Photo(id=i, corners=quad(100 * i, 0, 100 * i + 90, 60), stamp=st)
              for i, st in enumerate(stamps, 1)]
    return Page(page_id="p", source="p.png", width=500, height=200, photos=photos)


def _stamp(text, date, alts=()):
    return DateStamp(corners=quad(0, 40, 30, 50), text=text, date=date, date_alts=list(alts))


def _apply(page, *reads):
    apply(page, LlmResult(texts=[], issues=[], model="m", cached=False, stamps=list(reads)))


def test_agreeing_stamp_is_kept_and_not_flagged():
    page = _page(_stamp("6 5 '95", "1995-05-06"))
    _apply(page, StampRead(1, True, "6 5 '95", "1995-05-06"))
    ph = page.photos[0]
    assert ph.stamp.date == "1995-05-06" and not ph.needs_review and not page.issues
    assert ph.stamp.reader == "llm:m"


def test_disagreement_takes_claudes_date_and_flags_the_photo():
    page = _page(_stamp("'99 9 28", "1999-09-28"), _stamp("98 8 9", "1998-08-09"))
    _apply(page, StampRead(1, True, "'99 7 20", "1999-07-20"),
           StampRead(2, True, "98 8 9", "1998-08-09"))
    p1, p2 = page.photos
    assert p1.stamp.date == "1999-07-20" and "1999-09-28" in p1.stamp.date_alts
    assert p1.stamp.text == "'99 7 20" and p1.needs_review
    assert not p2.needs_review
    [issue] = page.issues
    assert issue.kind == "stamp_mismatch" and issue.photo_ids == [1] and issue.source == "llm"
    assert "1999-09-28" in issue.note and "1999-07-20" in issue.note


def test_undated_ocr_agrees_with_one_of_its_near_ties():
    page = _page(_stamp("96 2 26", None, ["1996-06-26", "1996-02-26"]))
    _apply(page, StampRead(1, True, "96 6 26", "1996-06-26"))
    ph = page.photos[0]
    assert ph.stamp.date == "1996-06-26" and not ph.needs_review and not page.issues


def test_undated_ocr_and_a_new_date_from_claude_is_flagged():
    page = _page(_stamp("96 4 2", None, ["1996-04-02"]))
    _apply(page, StampRead(1, True, "96 4 8", "1996-04-08"))
    ph = page.photos[0]
    assert ph.stamp.date == "1996-04-08" and ph.needs_review


def test_claude_unsure_keeps_the_ocr_date_flagged():
    page = _page(_stamp("'99 1 18", "1999-01-18"))
    _apply(page, StampRead(1, True, "'99 ? 18", None))
    ph = page.photos[0]
    assert ph.stamp.date == "1999-01-18" and ph.stamp.reader == "ocr" and ph.needs_review
    assert "kept the OCR date" in page.issues[0].note


def test_neither_can_read_it():
    page = _page(_stamp("2? 5 '92", None))
    _apply(page, StampRead(1, True, "2? 5 '92", None))
    assert page.photos[0].stamp.date is None and not page.photos[0].needs_review and not page.issues


def test_reapplying_replaces_the_old_stamp_issues():
    page = _page(_stamp("'99 9 28", "1999-09-28"))
    _apply(page, StampRead(1, True, "'99 7 20", "1999-07-20"))
    _apply(page, StampRead(1, True, "'99 7 20", "1999-07-20"))  # L pressed again: now agrees
    assert not page.issues


def test_stamp_answers_only_for_photos_that_were_asked():
    page = _page(_stamp("6 5 '95", "1995-05-06"), None)
    data = {"stamps": [{"photo_id": 1, "found": True, "text": "6 5 '95", "date": "1995-05-06"},
                       {"photo_id": 1, "found": True, "text": "x", "date": "2000-01-01"},
                       {"photo_id": 2, "found": True, "text": "1 1 '01", "date": "2001-01-01"},
                       {"photo_id": 9, "found": True, "text": "1 1 '01", "date": "2001-01-01"}]}
    assert _validate_stamps(page, data) == [StampRead(1, True, "6 5 '95", "1995-05-06")]
    bad = {"stamps": [{"photo_id": 1, "found": True, "text": "31 2 '95", "date": "1995-02-31"}]}
    assert _validate_stamps(page, bad)[0].date is None


def test_request_carries_a_closeup_of_each_stamp():
    linker = LlmLinker.__new__(LlmLinker)  # no network: prepare() only builds the request
    linker.model, linker.effort = "m", None
    page = _page(_stamp("6 5 '95", "1995-05-06"), None, _stamp("98 8 9", None, ["1998-08-09"]))
    img = np.full((200, 500, 3), 200, np.uint8)
    req = linker.prepare(page, img)
    labels = [c["text"] for c in req.content if c["type"] == "text"]
    assert "S1 (date stamp on P1):" in labels and "S3 (date stamp on P3):" in labels
    assert not any(lb.startswith("S2") for lb in labels)
    prompt = labels[-1]
    assert 'OCR read "6 5 \'95" = 1995-05-06' in prompt and "P2: no date stamp found" in prompt
    assert "Stamp close-ups: S1, S3" in prompt


def test_stamp_request_without_a_box_sends_corner_closeups():
    """Shrunk to page size a small stamp is unreadable: with no stamp box, the four corners go
    along at full resolution, the top ones turned 180° (a stamp there is upside down)."""
    from albumsplit.llm import stamp_corners
    img = np.full((1200, 1600, 3), 120, np.uint8)
    corners = quad(100, 100, 1100, 800)  # landscape, 1000 x 700
    img[110:130, 110:150] = (0, 0, 255)  # a red mark in the top left corner
    crops = dict(stamp_corners(img, corners))
    assert list(crops) == ["bottom right", "top left, turned", "bottom left", "top right, turned"]
    tl = crops["top left, turned"]
    assert tl.shape[:2] == (154, 450)
    ys, xs = np.nonzero(tl[..., 2] > 200)
    assert ys.min() > tl.shape[0] * 0.8 and xs.min() > tl.shape[1] * 0.85  # turned: now bottom right
    # A portrait print is turned to landscape first: its corner crops run along the long edges.
    portrait = dict(stamp_corners(img, quad(100, 100, 800, 1100)))
    assert portrait["bottom right"].shape[1] > portrait["bottom right"].shape[0]
    linker = LlmLinker.__new__(LlmLinker)
    linker.model, linker.effort = "m", None
    page = Page(page_id="p", source="p.png", width=1600, height=1200,
                photos=[Photo(id=1, corners=corners)])
    content, _ = linker.prepare_stamp(page, page.photos[0], img)
    assert sum(c["type"] == "image" for c in content) == 5  # the photo + 4 corners


def test_marked_stamp_takes_claudes_reading_unflagged():
    """A box marked in review has no OCR reading to disagree with."""
    st = DateStamp(corners=quad(0, 40, 30, 50), manual=True)
    page = _page(st)
    _apply(page, StampRead(1, True, "8 5 '93", "1993-05-08"))
    ph = page.photos[0]
    assert ph.stamp.date == "1993-05-08" and ph.stamp.text == "8 5 '93"
    assert ph.stamp.reader == "llm:m" and not ph.needs_review and not page.issues
    linker = LlmLinker.__new__(LlmLinker)
    linker.model, linker.effort = "m", None
    req = linker.prepare(_page(DateStamp(corners=quad(0, 40, 30, 50), manual=True)),
                         np.full((200, 500, 3), 200, np.uint8))
    assert "P1: date stamp S1, marked in review, not read yet" in req.content[-1]["text"]


def test_effort_is_sent_and_keys_the_cache():
    """Answers at another effort are other answers; the model's default keeps the old keys."""
    page = _page(_stamp("6 5 '95", "1995-05-06"))
    img = np.full((200, 500, 3), 200, np.uint8)
    linker = LlmLinker.__new__(LlmLinker)
    linker.model, linker.effort = "m", None
    plain = linker.prepare(page, img)
    assert "effort" not in linker.params(plain.content)["output_config"]
    linker.effort = "low"
    low = linker.prepare(page, img)
    assert linker.params(low.content)["output_config"]["effort"] == "low"
    assert linker.tag == "m@low" and low.digest != plain.digest


def test_cache_keeps_token_usage_beside_the_answer(tmp_path):
    import json
    from types import SimpleNamespace

    from albumsplit.llm import parse_message, read_cache, write_cache
    msg = SimpleNamespace(stop_reason="end_turn", content=[SimpleNamespace(type="text", text='{"a": 1}')],
                          usage=SimpleNamespace(input_tokens=5000, output_tokens=900,
                                                cache_read_input_tokens=0))
    data = parse_message("p", msg)
    write_cache(tmp_path / "p.json", "h", "m", data, "low")
    stored = json.loads((tmp_path / "p.json").read_text(encoding="utf-8"))
    assert stored["usage"] == {"input_tokens": 5000, "output_tokens": 900,
                               "cache_read_input_tokens": 0}
    assert stored["effort"] == "low" and read_cache(tmp_path / "p.json", "h") == {"a": 1}
