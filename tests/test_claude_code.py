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
