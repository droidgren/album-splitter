"""Claude Code as the Claude backend: request folders, answers, the `claude -p` call."""

import json
import subprocess

import numpy as np
import pytest

from albumsplit import claude_code
from albumsplit.claude_code import ClaudeCodeLinker, extract_json, requests_dir
from albumsplit.llm import SCHEMA, _encode, _image
from albumsplit.model import Page, Photo, TextBlock

ANSWER = {"texts": [{"id": 1, "is_caption": True, "photo_ids": [1], "transcription": "Påsk 95",
                     "date": "1995-04-16"}], "stamps": [], "issues": []}


def _page():
    return Page(page_id="IMG_1", source="IMG_1.png", width=400, height=300,
                photos=[Photo(id=1, corners=[(10, 10), (200, 10), (200, 150), (10, 150)])],
                texts=[TextBlock(id=1, box=(220, 20, 160, 40))])


def _img():
    img = np.full((300, 400, 3), 255, np.uint8)
    img[10:150, 10:200] = (40, 90, 160)
    return img


def _content():
    jpg = _encode(np.zeros((8, 8, 3), np.uint8))
    return [{"type": "text", "text": "Page overview:"}, _image(jpg),
            {"type": "text", "text": "T1:"}, _image(jpg), {"type": "text", "text": "Answer."}]


def _linker(tmp_path, mode="export", **kw):
    return ClaudeCodeLinker(tmp_path, "sonnet", mode=mode, claude=kw.pop("claude", None), **kw)


def _folder(tmp_path):
    dirs = [d for d in requests_dir(tmp_path).iterdir() if d.is_dir()]
    assert len(dirs) == 1
    return dirs[0]


def test_export_writes_the_request_and_waits(tmp_path):
    llm = _linker(tmp_path)
    assert llm._call("IMG_1", _content()) is None
    assert llm.pending == 1
    d = _folder(tmp_path)
    md = (d / "request.md").read_text(encoding="utf-8")
    assert md.index("Page overview:") < md.index("read 01.jpg") < md.index("T1:") < md.index("read 02.jpg")
    assert "You read scanned pages" in md  # the system prompt goes along
    assert (d / "01.jpg").read_bytes()[:2] == b"\xff\xd8"
    assert json.loads((d / "schema.json").read_text()) == SCHEMA
    assert (requests_dir(tmp_path) / "README.md").exists()
    assert llm.model == "claude-code/sonnet" and llm.tag == "claude-code/sonnet"


def test_an_answer_in_the_folder_is_used(tmp_path):
    llm = _linker(tmp_path)
    llm._call("IMG_1", _content())
    (_folder(tmp_path) / "answer.json").write_text("```json\n" + json.dumps(ANSWER) + "\n```",
                                                    encoding="utf-8")
    assert llm._call("IMG_1", _content()) == ANSWER


def test_an_answer_without_the_required_keys_is_refused(tmp_path):
    llm = _linker(tmp_path)
    llm._call("IMG_1", _content())
    (_folder(tmp_path) / "answer.json").write_text('{"texts": []}', encoding="utf-8")
    assert llm._call("IMG_1", _content()) is None


def test_other_content_gets_another_folder(tmp_path):
    llm = _linker(tmp_path)
    llm._call("IMG_1", _content())
    llm._call("IMG_1", _content()[:3])
    assert len([d for d in requests_dir(tmp_path).iterdir() if d.is_dir()]) == 2


def _fake_run(monkeypatch, stdout, returncode=0, seen=None):
    def run(cmd, **kw):
        if seen is not None:
            seen.update(cmd=cmd, **kw)
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")
    monkeypatch.setattr(claude_code.subprocess, "run", run)


def test_run_calls_claude_without_the_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    seen: dict = {}
    _fake_run(monkeypatch, json.dumps({"type": "result", "is_error": False, "result": "",
                                       "structured_output": ANSWER, "total_cost_usd": 0.1}),
              seen=seen)
    llm = _linker(tmp_path, mode="run", claude=__file__)
    assert llm._call("IMG_1", _content()) == ANSWER
    d = _folder(tmp_path)
    assert seen["cwd"] == d and "ANTHROPIC_API_KEY" not in seen["env"]
    cmd = seen["cmd"]
    assert cmd[cmd.index("--model") + 1] == "sonnet"
    assert cmd[cmd.index("--tools") + 1] == "Read"
    assert json.loads(cmd[cmd.index("--json-schema") + 1]) == SCHEMA
    assert json.loads((d / "answer.json").read_text(encoding="utf-8")) == ANSWER
    assert json.loads((d / "run.json").read_text())["total_cost_usd"] == 0.1
    # Answered once: the next call reads the folder.
    _fake_run(monkeypatch, "", returncode=1)
    assert llm._call("IMG_1", _content()) == ANSWER


def test_run_takes_json_from_the_text_answer(tmp_path, monkeypatch):
    text = "Here it is:\n```json\n" + json.dumps(ANSWER) + "\n```"
    _fake_run(monkeypatch, json.dumps([{"type": "system"},
                                       {"type": "result", "is_error": False, "result": text}]))
    assert _linker(tmp_path, mode="run", claude=__file__)._call("IMG_1", _content()) == ANSWER


@pytest.mark.parametrize("stdout,code", [
    ("not json", 0),
    (json.dumps({"type": "result", "is_error": True, "result": "usage limit reached"}), 1),
    (json.dumps({"type": "result", "is_error": False, "result": "I cannot see the image."}), 0),
])
def test_run_failures_give_none(tmp_path, monkeypatch, stdout, code):
    _fake_run(monkeypatch, stdout, returncode=code)
    llm = _linker(tmp_path, mode="run", claude=__file__)
    assert llm._call("IMG_1", _content()) is None
    assert not (_folder(tmp_path) / "answer.json").exists()


def test_timeout_gives_none(tmp_path, monkeypatch):
    def run(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, kw["timeout"])
    monkeypatch.setattr(claude_code.subprocess, "run", run)
    assert _linker(tmp_path, mode="run", claude=__file__, timeout=1)._call("IMG_1", _content()) is None


def test_extract_json():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('Answer: {"a": 1} done') == {"a": 1}
    assert extract_json("```\n{\"a\": 1}\n```") == {"a": 1}
    assert extract_json("no json") is None


def test_page_read_end_to_end(tmp_path, monkeypatch):
    """claude_page with the Claude Code backend: answer applied, cached apart from the API's."""
    from albumsplit.pipeline import claude_page, page_json_path
    scans = tmp_path / "scans"
    scans.mkdir()
    import cv2
    cv2.imwrite(str(scans / "IMG_1.png"), _img())
    work = tmp_path / "work"
    path = page_json_path(work, "IMG_1")
    path.parent.mkdir(parents=True)
    _page().save(path)
    _fake_run(monkeypatch, json.dumps({"type": "result", "is_error": False,
                                       "structured_output": ANSWER}))
    llm = ClaudeCodeLinker(work, "opus", mode="run", claude=__file__)
    res = claude_page(llm, work, scans, path, debug=False, cache_dir=claude_code.llm_cache_dir(work))
    assert res.status == "called"
    page = Page.load(path)
    assert page.texts[0].transcription == "Påsk 95" and page.linker == "llm:claude-code/opus"
    assert (work / "claude-code" / "llm" / "IMG_1.json").exists()
    assert not (work / "llm").exists()
    res = claude_page(llm, work, scans, path, debug=False, cache_dir=claude_code.llm_cache_dir(work))
    assert res.status == "cached"


def _setup_page(tmp_path, page):
    import cv2
    from albumsplit.pipeline import page_json_path
    scans = tmp_path / "scans"
    scans.mkdir()
    cv2.imwrite(str(scans / "IMG_1.png"), _img())
    work = tmp_path / "work"
    path = page_json_path(work, "IMG_1")
    path.parent.mkdir(parents=True)
    page.save(path)
    return scans, work, path


def test_stamps_step_reads_photos_without_a_stamp_box(tmp_path, monkeypatch):
    from albumsplit.llm import STAMP_SCHEMA
    from albumsplit.model import DateStamp, Photo
    from albumsplit.pipeline import stamps_page
    page = _page()
    page.photos.append(Photo(id=2, corners=[(210, 160), (390, 160), (390, 290), (210, 290)],
                             stamp=DateStamp(corners=[(300, 270), (380, 270), (380, 285), (300, 285)],
                                             text="6 5 '95", date="1995-05-06")))
    page.photos.append(Photo(id=3, corners=[(10, 160), (200, 160), (200, 290), (10, 290)],
                             date="1990"))  # reviewer date: left alone
    scans, work, path = _setup_page(tmp_path, page)
    seen = []

    def run(cmd, **kw):
        seen.append(json.loads(cmd[cmd.index("--json-schema") + 1]))
        out = {"found": True, "text": "'94 7 14", "date": "1994-07-14", "corner": "bottom left"}
        return subprocess.CompletedProcess(cmd, 0, stderr="", stdout=json.dumps(
            {"type": "result", "is_error": False, "structured_output": out}))
    monkeypatch.setattr(claude_code.subprocess, "run", run)
    llm = ClaudeCodeLinker(work, "opus", mode="run", claude=__file__)
    res = stamps_page(llm, work, scans, path, debug=False, cache_dir=claude_code.llm_cache_dir(work))
    assert (res.asked, res.found, res.unplaced, res.failed) == (1, 1, 0, 0)
    assert seen == [STAMP_SCHEMA]
    p1, p2, p3 = Page.load(path).photos
    assert p1.stamp.date == "1994-07-14" and p1.stamp.reader == "llm:claude-code/opus"
    assert p2.stamp.date == "1995-05-06" and p3.stamp is None
    # Done once: a second run has nothing left to ask.
    assert stamps_page(llm, work, scans, path, debug=False).asked == 0


def test_stamps_step_flags_a_stamp_outside_the_corners(tmp_path, monkeypatch):
    from albumsplit.pipeline import stamps_page
    scans, work, path = _setup_page(tmp_path, _page())
    _fake_run(monkeypatch, json.dumps({"type": "result", "is_error": False, "structured_output": {
        "found": True, "text": "'94 7 14", "date": "1994-07-14", "corner": "elsewhere"}}))
    llm = ClaudeCodeLinker(work, "opus", mode="run", claude=__file__)
    for _ in range(2):  # the issue is not added twice
        res = stamps_page(llm, work, scans, path, debug=False)
    assert res.unplaced == 1
    page = Page.load(path)
    assert page.photos[0].stamp is None and page.photos[0].needs_review
    assert [i.source for i in page.issues] == ["stamps"]


def test_stamps_step_reads_an_undated_stamp_box_again(tmp_path, monkeypatch):
    from albumsplit.model import DateStamp
    from albumsplit.pipeline import stamps_page
    page = _page()
    box = [(150, 130), (195, 130), (195, 145), (150, 145)]
    page.photos[0].stamp = DateStamp(corners=box, text="'?9 7 20")  # the OCR could not date it
    scans, work, path = _setup_page(tmp_path, page)
    _fake_run(monkeypatch, json.dumps({"type": "result", "is_error": False, "structured_output": {
        "found": True, "text": "'99 7 20", "date": "1999-07-20", "corner": ""}}))
    llm = ClaudeCodeLinker(work, "opus", mode="run", claude=__file__)
    res = stamps_page(llm, work, scans, path, debug=False)
    assert (res.asked, res.found) == (1, 1)
    st = Page.load(path).photos[0].stamp
    assert st.date == "1999-07-20" and st.corners == box and st.reader == "llm:claude-code/opus"
