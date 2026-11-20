import json
import os
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
    return TestClient(create_app(work, scans)), work, scans


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
