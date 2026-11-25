import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from albumsplit.model import Link, Page, Photo, TextBlock  # noqa: E402
from albumsplit.pipeline import page_json_path, process_page  # noqa: E402
from albumsplit.review import create_app  # noqa: E402
from albumsplit.review.geometry import text_outline  # noqa: E402
from albumsplit.review.server import preview_path  # noqa: E402

W, H = 4000, 3000


def quad(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


@pytest.fixture
def env(tmp_path: Path):
    scans, work = tmp_path / "scans", tmp_path / "work"
    scans.mkdir()
    img = np.full((H, W, 3), 245, np.uint8)
    cv2.rectangle(img, (200, 200), (1800, 1400), (60, 90, 120), -1)
    cv2.imwrite(str(scans / "p1.jpg"), img)
    page = Page(page_id="p1", source="p1.jpg", width=W, height=H,
                photos=[Photo(id=1, corners=quad(200, 200, 1800, 1400),
                              outline=[(200, 200), (1800, 200), (1800, 1400), (200, 1400)],
                              needs_review=True),
                        Photo(id=2, corners=quad(2200, 200, 3800, 1400))],
                texts=[TextBlock(id=1, box=(200, 1450, 1200, 150), transcription="Påsk -84"),
                       TextBlock(id=2, box=(2200, 1450, 1000, 150), transcription="Mormor",
                                 date="1984-04-22")],
                links=[Link(1, 1), Link(2, 2)])
    page.save(page_json_path(work, "p1"))
    # Own config dir: opening a project must not add test folders to the user's recent list.
    return TestClient(create_app(work, scans, config_dir=tmp_path / "cfg")), work, scans


def test_list_and_get(env):
    client, _, _ = env
    pages = client.get("/api/pages").json()
    assert [p["page_id"] for p in pages] == ["p1"]
    assert pages[0]["needs_review"] == 1 and pages[0]["n_unlinked"] == 0
    d = client.get("/api/pages/p1").json()
    assert d["photos"][0]["derived_caption"] == "Påsk -84"
    assert d["photos"][1]["derived_date"] == "1984-04-22"


def test_unknown_or_traversal_id_is_404(env):
    client, _, _ = env
    assert client.get("/api/pages/nope").status_code == 404
    assert client.get("/api/pages/..%2Fp1").status_code == 404
    assert client.put("/api/pages/nope", json={}).status_code == 404


def test_preview_is_downscaled_and_cached(env):
    client, work, _ = env
    r = client.get("/api/pages/p1/image")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    im = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
    assert max(im.shape[:2]) == 3000
    out = preview_path(work, "p1")
    mtime = out.stat().st_mtime_ns
    client.get("/api/pages/p1/image")
    assert out.stat().st_mtime_ns == mtime


def test_put_roundtrip_marks_edited(env):
    client, work, _ = env
    d = client.get("/api/pages/p1").json()
    d["photos"][1]["caption"] = "Mormor på verandan"
    d["status"] = "approved"
    d["width"] = 1  # not editable: kept from disk
    saved = client.put("/api/pages/p1", json=d).json()
    assert saved["edited"] and saved["status"] == "approved" and saved["width"] == W
    page = Page.load(page_json_path(work, "p1"))
    assert page.photos[1].caption == "Mormor på verandan" and page.edited
    # Untouched photo keeps its detector state.
    assert page.photos[0].outline is not None and page.photos[0].needs_review


def test_moved_photo_loses_outline_and_review_flag(env):
    client, _, _ = env
    d = client.get("/api/pages/p1").json()
    d["photos"][0]["corners"][2] = [1850, 1420]
    saved = client.put("/api/pages/p1", json=d).json()
    assert saved["photos"][0]["outline"] is None
    assert saved["photos"][0]["needs_review"] is False


def test_new_text_over_photo_corner_gets_outline(env):
    client, _, _ = env
    d = client.get("/api/pages/p1").json()
    d["texts"].append({"id": 3, "box": [1600, 1000, 500, 600], "transcription": None, "date": None,
                       "outline": None})
    saved = client.put("/api/pages/p1", json=d).json()
    ol = saved["texts"][2]["outline"]
    assert ol is not None
    # The covered photo corner (1600..1800, 1000..1400) is outside the outline polygon.
    c = cv2.pointPolygonTest(np.array(ol, np.float32), (1700, 1200), False)
    assert c < 0
    assert cv2.pointPolygonTest(np.array(ol, np.float32), (2000, 1500), False) > 0
    # Unchanged texts are left alone.
    assert saved["texts"][0]["outline"] is None


def test_deleting_photo_drops_its_links(env):
    client, _, _ = env
    d = client.get("/api/pages/p1").json()
    d["photos"] = [p for p in d["photos"] if p["id"] != 2]  # the client forgot the link
    saved = client.put("/api/pages/p1", json=d).json()
    assert saved["links"] == [{"text_id": 1, "photo_id": 1}]


def test_bad_body_is_rejected(env):
    client, _, _ = env
    d = client.get("/api/pages/p1").json()
    assert client.put("/api/pages/p1", json={**d, "page_id": "other"}).status_code == 422
    assert client.put("/api/pages/p1", json={**d, "status": "done"}).status_code == 422
    assert client.put("/api/pages/p1", json={"page_id": "p1"}).status_code == 422


def test_text_outline_none_when_clear_of_photos():
    assert text_outline((0, 0, 100, 50), [Photo(id=1, corners=quad(200, 200, 300, 300))]) is None


def test_detect_skips_edited_pages(env):
    _, work, scans = env
    page = Page.load(page_json_path(work, "p1"))
    page.edited = True
    page.save(page_json_path(work, "p1"))
    res = process_page(scans, scans / "p1.jpg", work, debug=False, params=None, page_id="p1")
    assert res.skipped and res.skip_reason == "edited"
    assert json.loads(page_json_path(work, "p1").read_text(encoding="utf-8"))["texts"][0]["transcription"] == "Påsk -84"
    assert os.path.exists(page_json_path(work, "p1"))


# ---------- start page: project, folders, detection ----------

def test_no_project_means_409_and_project_can_be_chosen(tmp_path: Path, env):
    _, work, scans = env
    client = TestClient(create_app(config_dir=tmp_path / "cfg"))
    assert client.get("/api/pages").status_code == 409
    assert client.get("/api/project").json()["work_dir"] is None
    r = client.post("/api/project", json={"work_dir": str(work), "input_dir": str(scans)})
    assert r.status_code == 200, r.text
    info = r.json()
    assert info["stats"]["pages"] == 1 and info["stats"]["scans"] == 1 and info["stats"]["new"] == 0
    assert [p["page_id"] for p in client.get("/api/pages").json()] == ["p1"]
    assert client.get("/api/recent").json()[0]["work_dir"] == str(work.resolve())


def test_project_from_work_dir_alone_and_bad_choices(tmp_path: Path, env):
    _, work, scans = env
    from albumsplit.pipeline import save_input_dir
    save_input_dir(work, scans)
    client = TestClient(create_app(config_dir=tmp_path / "cfg"))
    r = client.post("/api/project", json={"work_dir": str(work)})
    assert r.status_code == 200 and r.json()["input_dir"] == str(scans.resolve())
    # A new work folder needs the scans; the work folder may not live inside the scans.
    assert client.post("/api/project", json={"work_dir": str(tmp_path / "new")}).status_code == 422
    r = client.post("/api/project", json={"work_dir": str(scans / "work"), "input_dir": str(scans)})
    assert r.status_code == 422 and not (scans / "work").exists()
    r = client.post("/api/project", json={"work_dir": str(tmp_path / "new"), "input_dir": str(scans)})
    assert r.status_code == 200 and r.json()["stats"]["new"] == 1


def test_folder_listing(tmp_path: Path, env):
    client, work, scans = env
    from albumsplit.pipeline import save_input_dir
    save_input_dir(work, scans)
    d = client.get("/api/fs", params={"path": str(tmp_path)}).json()
    assert {x["name"] for x in d["dirs"]} == {"scans", "work"}
    assert client.get("/api/fs", params={"path": str(scans)}).json()["n_images"] == 1
    w = client.get("/api/fs", params={"path": str(work)}).json()
    assert w["is_work_dir"] and w["input_dir"] == str(scans)
    assert client.get("/api/fs", params={"path": str(tmp_path / "nope")}).status_code == 404


def test_detect_job_runs_in_background(tmp_path: Path):
    scans, work = tmp_path / "scans", tmp_path / "work"
    scans.mkdir()
    img = np.full((900, 1200, 3), 245, np.uint8)
    cv2.rectangle(img, (100, 100), (600, 500), (60, 90, 120), -1)
    cv2.imwrite(str(scans / "a.jpg"), img)
    client = TestClient(create_app(config_dir=tmp_path / "cfg"))
    client.post("/api/project", json={"work_dir": str(work), "input_dir": str(scans)})
    assert client.post("/api/detect", json={}).status_code == 200
    import time
    for _ in range(300):
        st = client.get("/api/detect").json()
        if st["state"] not in ("starting", "running"):
            break
        time.sleep(0.1)
    assert st["state"] == "finished" and st["detected"] == 1, st
    assert (work / "pages" / "a.json").exists()
    assert client.get("/api/project").json()["stats"]["new"] == 0
    # Again: nothing new to do.
    client.post("/api/detect", json={})
    for _ in range(100):
        st = client.get("/api/detect").json()
        if st["state"] == "finished":
            break
        time.sleep(0.1)
    assert st["skipped"] == 1 and st["detected"] == 0


# ---------- Claude, with a fake client ----------

class FakeLinker:
    calls: list = []

    def __init__(self, model):
        self.model = model

    def link(self, page, img, cache_dir, *, force=False):
        from albumsplit.llm import LlmResult, TextAnswer
        FakeLinker.calls.append(("link", page.page_id, force))
        return LlmResult(texts=[TextAnswer(id=1, is_caption=True, photo_ids=[2],
                                           transcription="Påsk\n-84", date="1984-04-22")],
                         issues=[], model=self.model, cached=False)

    def _call(self, page_id, content, system, schema):  # the date-guess request
        FakeLinker.calls.append(("dates", page_id, None))
        return {"guesses": [{"id": 1, "date": "1984-04", "confidence": "medium",
                             "reason": "same page as Mormor 1984-04-22"}], "suspects": []}

    def read_stamp(self, page, photo, img, cache_dir, *, force=False):
        from albumsplit.llm import StampAnswer
        FakeLinker.calls.append(("stamp", photo.id, force))
        return StampAnswer(found=True, text="22 4 '84", date="1984-04-22", model=self.model)


def test_ask_claude_about_page(env, monkeypatch):
    client, work, _ = env
    monkeypatch.setattr("albumsplit.llm.LlmLinker", FakeLinker)
    FakeLinker.calls = []
    r = client.post("/api/pages/p1/llm")
    assert r.status_code == 200, r.text
    d = r.json()
    assert ("link", "p1", True) in FakeLinker.calls  # always a fresh answer from review
    assert {"text_id": 1, "photo_id": 2} in d["links"] and d["llm_result"]["transcribed"] == 2
    assert d["photos"][1]["derived_caption"] == "Påsk -84 Mormor"
    page = Page.load(page_json_path(work, "p1"))
    assert page.edited and page.linker.startswith("llm:")


def test_guess_dates_setting_and_job(env, monkeypatch):
    client, work, _ = env
    monkeypatch.setattr("albumsplit.llm.LlmLinker", FakeLinker)
    FakeLinker.calls = []
    assert client.get("/api/project").json()["guess_dates"] is False
    assert client.post("/api/settings", json={"guess_dates": True}).json()["guess_dates"] is True
    r = client.post("/api/dates")
    assert r.status_code == 200, r.text
    for _ in range(100):
        st = client.get("/api/detect").json()
        if st["state"] not in ("starting", "running"):
            break
        time.sleep(0.05)
    assert st["state"] == "finished" and st["dates_only"], st
    assert [c[0] for c in FakeLinker.calls] == ["dates"]
    d = client.get("/api/pages/p1").json()
    p1, p2 = d["photos"]
    assert p1["date_guess"]["date"] == "1984-04" and p1["date_source"] == "guess"
    assert p1["derived_date"] == "1984-04" and p2["date_source"] == "caption"


def test_ask_claude_for_stamp(env, monkeypatch):
    client, work, _ = env
    monkeypatch.setattr("albumsplit.llm.LlmLinker", FakeLinker)
    # Photo 1 has no stamp box: the date becomes the photo's date.
    d = client.post("/api/pages/p1/photos/1/stamp").json()
    assert d["stamp_result"]["found"] and d["photos"][0]["date"] == "1984-04-22"
    # With a stamp box, the OCR reading moves to the alternatives.
    from albumsplit.model import DateStamp
    page = Page.load(page_json_path(work, "p1"))
    page.photos[1].stamp = DateStamp(corners=quad(3600, 1300, 3780, 1380), text="22 4 '81", date="1981-04-22")
    page.save(page_json_path(work, "p1"))
    d = client.post("/api/pages/p1/photos/2/stamp").json()
    st = d["photos"][1]["stamp"]
    assert st["date"] == "1984-04-22" and st["date_alts"] == ["1981-04-22"] and st["reader"].startswith("llm:")
    assert d["photos"][1]["derived_date"] == "1984-04-22"
    assert client.post("/api/pages/p1/photos/9/stamp").status_code == 404


def test_mark_stamp_box(env):
    client, work, _ = env
    # The box is kept as the photo's stamp, unread, for Claude.
    box = quad(3400, 1250, 3750, 1350)
    r = client.post("/api/pages/p1/photos/2/stamp_box", json={"box": box})
    assert r.status_code == 200, r.text
    st = r.json()["photos"][1]["stamp"]
    assert st["manual"] and st["date"] is None
    assert [tuple(map(round, c)) for c in st["corners"]] == box  # bottom half: reads as drawn
    page = Page.load(page_json_path(work, "p1"))
    assert page.edited and page.photos[1].stamp.manual
    assert client.post("/api/pages/p1/photos/2/stamp_box", json={"box": box[:3]}).status_code == 422
    assert client.post("/api/pages/p1/photos/9/stamp_box", json={"box": box}).status_code == 404


def test_claude_unavailable_is_503(env, monkeypatch):
    client, _, _ = env
    from albumsplit.llm import LlmUnavailable

    def broken(model):
        raise LlmUnavailable("no Anthropic credentials found: set ANTHROPIC_API_KEY")
    monkeypatch.setattr("albumsplit.llm.LlmLinker", broken)
    r = client.post("/api/pages/p1/llm")
    assert r.status_code == 503 and "ANTHROPIC_API_KEY" in r.json()["detail"]


def test_stamp_zoom_is_upright_and_legible():
    from albumsplit.llm import stamp_zoom
    img = np.zeros((500, 500, 3), np.uint8)
    # A stamp running up the image (print rotated 90°): reading direction bottom -> top.
    z = stamp_zoom(img, [(100, 300), (100, 200), (130, 200), (130, 300)])
    assert z.shape[1] > z.shape[0] and z.shape[0] >= 160


def test_redetect_all_overwrites_reviewed_pages(tmp_path: Path):
    import time
    scans, work = tmp_path / "scans", tmp_path / "work"
    scans.mkdir()
    img = np.full((900, 1200, 3), 245, np.uint8)
    cv2.rectangle(img, (100, 100), (600, 500), (60, 90, 120), -1)
    cv2.imwrite(str(scans / "a.jpg"), img)
    client = TestClient(create_app(config_dir=tmp_path / "cfg"))
    client.post("/api/project", json={"work_dir": str(work), "input_dir": str(scans)})

    def run(body):
        assert client.post("/api/detect", json=body).status_code == 200
        for _ in range(300):
            st = client.get("/api/detect").json()
            if st["state"] not in ("starting", "running"):
                return st
            time.sleep(0.1)
        raise AssertionError("detection did not finish")

    run({})
    path = work / "pages" / "a.json"
    page = Page.load(path)
    page.status, page.edited, page.photos = "approved", True, []
    page.save(path)
    st = run({"redo": True})  # reviewed pages are kept...
    assert st["skipped"] == 1 and Page.load(path).status == "approved"
    st = run({"force": True})  # ...unless everything is re-detected
    assert st["detected"] == 1, st
    page = Page.load(path)
    assert page.status == "proposed" and not page.edited and page.photos


# ---------- rotation and export ----------

def test_rotation_roundtrip_and_snapped(env):
    client, work, _ = env
    d = client.get("/api/pages/p1").json()
    assert d["photos"][0]["rotation"] == 0
    d["photos"][0]["rotation"] = 270
    d["photos"][1]["rotation"] = 100  # not a quarter turn: snapped to 90
    client.put("/api/pages/p1", json=d)
    page = Page.load(page_json_path(work, "p1"))
    assert [p.rotation for p in page.photos] == [270, 90]


def _wait_export(client):
    import time
    for _ in range(300):
        st = client.get("/api/export").json()
        if st["state"] not in ("starting", "running"):
            return st
        time.sleep(0.1)
    raise AssertionError(st)


def test_export_job_from_start_page(env, tmp_path):
    client, work, scans = env
    proj = client.get("/api/project").json()
    assert proj["export_dir"] == str(work / "export") and "found" in proj["exiftool"]
    # Not inside the scans.
    r = client.post("/api/export", json={"out_dir": str(scans / "out"), "metadata": False})
    assert r.status_code == 422
    out = tmp_path / "ut"
    # Nothing approved yet: runs, exports nothing.
    assert client.post("/api/export", json={"out_dir": str(out), "metadata": False}).status_code == 200
    st = _wait_export(client)
    assert st["state"] == "finished" and st["total"] == 0 and st["photos"] == 0
    # With the not-approved pages, and a rotated photo.
    d = client.get("/api/pages/p1").json()
    d["photos"][1]["rotation"] = 90
    client.put("/api/pages/p1", json=d)
    client.post("/api/export", json={"out_dir": str(out), "all": True, "cards": False, "metadata": False})
    st = _wait_export(client)
    assert st["state"] == "finished" and st["done"] == 1 and st["photos"] == 2, st
    assert sorted(p.name for p in out.iterdir()) == ["index.csv", "p1_1.jpg", "p1_2.jpg"]
    h, w = cv2.imread(str(out / "p1_2.jpg")).shape[:2]
    assert h > w  # 1600x1200 print turned a quarter
    # The folder is remembered in workdir.json, without losing what else is there.
    assert client.get("/api/project").json()["export_dir"] == str(out.resolve())
    from albumsplit.pipeline import load_input_dir, save_input_dir
    save_input_dir(work, scans)
    assert client.get("/api/project").json()["export_dir"] == str(out.resolve())
    assert load_input_dir(work) == scans


def test_export_needs_exiftool_when_metadata_wanted(env, tmp_path, monkeypatch):
    import albumsplit.exiftool as et
    monkeypatch.setattr(et, "find_exiftool", lambda explicit=None: None)
    client, _, _ = env
    r = client.post("/api/export", json={"out_dir": str(tmp_path / "ut"), "all": True})
    assert r.status_code == 422 and "exiftool" in r.json()["detail"]


# ---------- model setting and Claude batches, with a fake Batch API ----------

def _run_detect(client, body):
    import time
    assert client.post("/api/detect", json=body).status_code == 200
    for _ in range(300):
        st = client.get("/api/detect").json()
        if st["state"] not in ("starting", "running"):
            return st
        time.sleep(0.1)
    raise AssertionError("detection did not finish")


def _two_scan_project(tmp_path: Path):
    scans, work = tmp_path / "scans", tmp_path / "work"
    scans.mkdir()
    for i, name in enumerate(("a", "b")):  # different scans: identical requests are sent once
        img = np.full((900, 1200, 3), 245, np.uint8)
        cv2.rectangle(img, (100 + 300 * i, 100), (600 + 300 * i, 500), (60, 90, 120), -1)
        cv2.imwrite(str(scans / f"{name}.jpg"), img)
    client = TestClient(create_app(config_dir=tmp_path / "cfg"))
    client.post("/api/project", json={"work_dir": str(work), "input_dir": str(scans)})
    return client, work


class FakeBatches:
    created: list = []
    status = "in_progress"
    answer: dict = {"texts": [], "issues": []}

    def create(self, requests):
        from types import SimpleNamespace as NS
        FakeBatches.created.append(requests)
        return NS(id=f"batch{len(FakeBatches.created)}")

    def retrieve(self, bid):
        from types import SimpleNamespace as NS
        n = len(FakeBatches.created[int(bid[5:]) - 1])
        ended = FakeBatches.status == "ended"
        return NS(processing_status=FakeBatches.status,
                  request_counts=NS(processing=0 if ended else n, succeeded=n if ended else 0,
                                    errored=0, canceled=0, expired=0))

    def results(self, bid):
        from types import SimpleNamespace as NS
        answer = json.dumps(FakeBatches.answer)
        for item in FakeBatches.created[int(bid[5:]) - 1]:
            msg = NS(stop_reason="end_turn", content=[NS(type="text", text=answer)])
            yield NS(custom_id=item["custom_id"], result=NS(type="succeeded", message=msg))


def _fake_batch_linker():
    from types import SimpleNamespace as NS

    from albumsplit.llm import LlmLinker

    class FakeBatchLinker(LlmLinker):
        models: list = []

        def __init__(self, model):  # no network: only the Batch API is used, and it is fake
            self.model = model
            self._anthropic = None
            self.client = NS(messages=NS(batches=FakeBatches()))
            FakeBatchLinker.models.append(model)
    return FakeBatchLinker


def test_model_setting_is_saved_per_project_and_used(tmp_path: Path, env, monkeypatch):
    client, work, scans = env
    client.post("/api/project", json={"work_dir": str(work), "input_dir": str(scans)})
    info = client.get("/api/project").json()
    assert info["llm_model"] == "claude-opus-5-5"
    assert {m["id"] for m in info["llm_models"]} >= {"claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"}
    r = client.post("/api/settings", json={"llm_model": "claude-sonnet-5-5"})
    assert r.status_code == 200 and r.json()["llm_model"] == "claude-sonnet-5-5"
    assert json.loads((work / "workdir.json").read_text(encoding="utf-8"))["llm_model"] == "claude-sonnet-5-5"
    assert client.post("/api/settings", json={"llm_model": "gpt-4"}).status_code == 422
    # Review's Ask Claude uses the chosen model.
    monkeypatch.setattr("albumsplit.llm.LlmLinker", FakeLinker)
    d = client.post("/api/pages/p1/llm").json()
    assert d["llm_result"]["model"] == "claude-sonnet-5-5"


def test_haiku_is_called_without_refusal_fallback():
    from types import SimpleNamespace as NS

    from albumsplit.llm import LlmLinker
    sent = []

    def create(**kw):
        sent.append(kw)
        return NS(stop_reason="end_turn", content=[NS(type="text", text='{"texts": [], "issues": []}')])
    for model, has_fallback in (("claude-haiku-5-5", False), ("claude-opus-5-5", True)):
        llm = LlmLinker.__new__(LlmLinker)
        llm.model, llm._anthropic = model, NS(APIStatusError=OSError, APIConnectionError=OSError)
        llm.client = NS(beta=NS(messages=NS(create=create)))
        assert llm._call("p", [{"type": "text", "text": "x"}]) == {"texts": [], "issues": []}
        assert ("fallbacks" in sent[-1]) == has_fallback and sent[-1]["model"] == model


def test_batch_detect_then_collect(tmp_path: Path, monkeypatch):
    FakeBatches.created, FakeBatches.status = [], "in_progress"
    monkeypatch.setattr("albumsplit.llm.LlmLinker", _fake_batch_linker())
    client, work = _two_scan_project(tmp_path)
    client.post("/api/settings", json={"llm_model": "claude-sonnet-5-5"})

    st = _run_detect(client, {"llm": True, "batch": True})
    assert st["state"] == "finished" and st["detected"] == 2, st
    assert st["queued"] == 2 and st["batches"] == ["batch1"]
    assert len(FakeBatches.created[0]) == 2
    assert FakeBatches.created[0][0]["params"]["model"] == "claude-sonnet-5-5"
    assert client.get("/api/project").json()["pending"] == {"batches": 1, "pages": 2}
    # Pages are saved right away with heuristic links, so review can start.
    assert Page.load(work / "pages" / "a.json").linker == "heuristic"

    b = client.get("/api/batches").json()
    assert b[0]["status"] == "in_progress" and b[0]["pages"] == 2 and b[0]["processing"] == 2

    # Page b is edited in review while the batch runs: its answer must not overwrite that.
    pb = Page.load(work / "pages" / "b.json")
    pb.edited = True
    pb.save(work / "pages" / "b.json")

    FakeBatches.status = "ended"
    assert client.get("/api/batches").json()[0]["status"] == "ended"
    r = client.post("/api/batches/collect").json()
    assert r["updated"] == 1 and r["not_updated"] == 1, r
    assert any("edited in review" in line for line in r["log"])
    assert Page.load(work / "pages" / "a.json").linker == "llm:claude-sonnet-5-5"
    assert Page.load(work / "pages" / "b.json").linker == "heuristic"
    assert r["pending"] == {"batches": 0, "pages": 0}
    assert client.get("/api/batches").json() == []


def test_cancelled_batch_detect_still_submits_queued_pages(tmp_path: Path, monkeypatch):
    FakeBatches.created, FakeBatches.status = [], "in_progress"
    monkeypatch.setattr("albumsplit.llm.LlmLinker", _fake_batch_linker())
    client, work = _two_scan_project(tmp_path)
    import albumsplit.llm as llm_mod
    from albumsplit.review.jobs import DetectJob
    job = DetectJob(input_dir=tmp_path / "scans", work_dir=work, llm=llm_mod.LlmLinker("claude-opus-5-5"),
                    batch=True)
    orig = job._detect_all

    def one_then_cancel(images, ids, queue):
        orig(images[:1], ids, queue)
        job.cancel()
        orig(images[1:], ids, queue)
    job._detect_all = one_then_cancel
    job._run()
    assert job.state == "cancelled" and job.queued == 1 and job.batches == ["batch1"]


def test_edited_only_for_real_changes(env):
    client, work, _ = env
    d = client.get("/api/pages/p1").json()
    assert client.put("/api/pages/p1", json=d).json()["edited"] is False  # nothing changed
    d["status"] = "approved"
    assert client.put("/api/pages/p1", json=d).json()["edited"] is False  # approving is not editing
    d["status"] = "proposed"
    assert client.put("/api/pages/p1", json=d).json()["edited"] is False  # nor is un-approving (U)
    d["links"] = d["links"] + [{"text_id": 1, "photo_id": 2}]
    assert client.put("/api/pages/p1", json=d).json()["edited"] is True   # a real change
    assert Page.load(page_json_path(work, "p1")).edited


# ---------- only ask Claude again (keep frames, boxes and links) ----------

def test_apply_keep_links():
    from albumsplit.llm import LlmResult, TextAnswer, apply
    page = Page(page_id="p", source="p.jpg", width=10, height=10,
                photos=[Photo(id=1, corners=quad(0, 0, 4, 4)), Photo(id=2, corners=quad(5, 0, 9, 4))],
                texts=[TextBlock(id=1, box=(0, 5, 4, 2), transcription="old"),
                       TextBlock(id=2, box=(5, 5, 4, 2))],
                links=[Link(1, 1)])
    res = LlmResult(texts=[TextAnswer(1, True, [2], "Påsk -84", "1984-04-22"),
                           TextAnswer(2, True, [1, 2], "Mormor", None)],
                    issues=[], model="m", cached=False)
    apply(page, res, keep_links=True)
    # T1 keeps the person's link to P1; T2 had none and gets Claude's.
    assert sorted((lk.text_id, lk.photo_id) for lk in page.links) == [(1, 1), (2, 1), (2, 2)]
    assert page.texts[0].transcription == "Påsk -84" and page.texts[0].date == "1984-04-22"
    assert page.texts[1].transcription == "Mormor" and page.linker == "llm:m"


class FakeAllTexts(FakeLinker):
    """Answers for every text on the page: each one about photo 2."""

    def link(self, page, img, cache_dir, *, force=False):
        from albumsplit.llm import LlmResult, TextAnswer
        FakeLinker.calls.append(("link", page.page_id, force))
        return LlmResult(texts=[TextAnswer(t.id, True, [2], f"text {t.id}", None) for t in page.texts],
                         issues=[], model=self.model, cached=False)


def _claude_only_setup(env):
    client, work, scans = env
    client.post("/api/project", json={"work_dir": str(work), "input_dir": str(scans)})
    path = page_json_path(work, "p1")
    page = Page.load(path)
    page.texts.append(TextBlock(id=3, box=(2200, 1650, 900, 120)))  # a box drawn in review
    page.edited = True
    page.save(path)
    # A second, approved page must be left alone.
    other = Page.load(path)
    other.page_id, other.status = "p2", "approved"
    other.save(page_json_path(work, "p2"))
    return client, work, page


def test_claude_only_now_keeps_boxes_and_links(env, monkeypatch):
    monkeypatch.setattr("albumsplit.llm.LlmLinker", FakeAllTexts)
    FakeLinker.calls = []
    client, work, before = _claude_only_setup(env)
    st = _run_detect(client, {"claude_only": True, "batch": False})
    assert st["state"] == "finished" and st["claude_only"], st
    assert st["total"] == 1 and st["detected"] == 1, st          # only the unapproved page
    assert [c[1] for c in FakeLinker.calls] == ["p1"]
    after = Page.load(page_json_path(work, "p1"))
    assert [t.box for t in after.texts] == [t.box for t in before.texts]   # no new detection
    assert [p.corners for p in after.photos] == [p.corners for p in before.photos]
    # T1 -> P1 and T2 -> P2 were on the page and stay; the new T3 gets Claude's link.
    assert sorted((lk.text_id, lk.photo_id) for lk in after.links) == [(1, 1), (2, 2), (3, 2)]
    assert [t.transcription for t in after.texts] == ["text 1", "text 2", "text 3"]
    assert after.edited and after.linker.startswith("llm:")
    assert Page.load(page_json_path(work, "p2")).linker == "heuristic"


def test_claude_only_batch_updates_edited_pages(env, monkeypatch):
    FakeBatches.created, FakeBatches.status = [], "in_progress"
    monkeypatch.setattr(FakeBatches, "answer", {"texts": [
        {"id": 1, "is_caption": True, "photo_ids": [2], "transcription": "Påsk", "date": None},
        {"id": 3, "is_caption": True, "photo_ids": [2], "transcription": "Ny", "date": "1990"}],
        "issues": []})
    monkeypatch.setattr("albumsplit.llm.LlmLinker", _fake_batch_linker())
    client, work, _ = _claude_only_setup(env)
    st = _run_detect(client, {"claude_only": True, "batch": True})
    assert st["queued"] == 1 and st["batches"] == ["batch1"], st
    assert Page.load(page_json_path(work, "p1")).linker == "heuristic"  # not yet
    FakeBatches.status = "ended"
    r = client.post("/api/batches/collect").json()
    assert r["updated"] == 1, r  # the page is edited, but this batch keeps links: applied
    after = Page.load(page_json_path(work, "p1"))
    assert sorted((lk.text_id, lk.photo_id) for lk in after.links) == [(1, 1), (2, 2), (3, 2)]
    assert after.texts[0].transcription == "Påsk" and after.texts[2].date == "1990"


def test_version_endpoint(tmp_path):
    client = TestClient(create_app(config_dir=tmp_path / "cfg"))
    v = client.get("/api/version").json()
    assert v["text"].startswith(v["version"])
