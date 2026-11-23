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
    linker.model = "m"
    page = _page(_stamp("6 5 '95", "1995-05-06"), None, _stamp("98 8 9", None, ["1998-08-09"]))
    img = np.full((200, 500, 3), 200, np.uint8)
    req = linker.prepare(page, img)
    labels = [c["text"] for c in req.content if c["type"] == "text"]
    assert "S1 (date stamp on P1):" in labels and "S3 (date stamp on P3):" in labels
    assert not any(lb.startswith("S2") for lb in labels)
    prompt = labels[-1]
    assert 'OCR read "6 5 \'95" = 1995-05-06' in prompt and "P2: no date stamp found" in prompt
    assert "Stamp close-ups: S1, S3" in prompt
