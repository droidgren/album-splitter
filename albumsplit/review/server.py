"""FastAPI app behind the start and review pages.

One project at a time: a scan folder (read only) and a work folder holding the page JSON.
It can be given on the command line or chosen on the start page.
"""

from __future__ import annotations

import json
import os
import string
import sys
import threading
import time
from pathlib import Path

import cv2
from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from ..imageio import IMAGE_EXTS, downscale, find_images, load_bgr
from ..llm import DEFAULT_MODEL
from ..model import Page, assign_page_ids, derived_caption, derived_date
from ..pipeline import load_input_dir, page_json_path, save_input_dir
from .geometry import normalize
from .jobs import DetectJob

STATIC = Path(__file__).parent / "static"
PREVIEW_LONG_SIDE = 3000
PREVIEW_QUALITY = 88
MAX_RECENT = 10


def preview_path(work_dir: Path, page_id: str) -> Path:
    return work_dir / "preview" / f"{page_id}.jpg"


def default_config_dir() -> Path:
    return Path.home() / ".albumsplit"


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def create_app(work_dir: Path | None = None, input_dir: Path | None = None, *,
               llm_model: str = DEFAULT_MODEL, config_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="albumsplit", docs_url=None, redoc_url=None)
    config_dir = config_dir or default_config_dir()
    lock = threading.Lock()          # page JSON writes
    llm_lock = threading.Lock()      # creating the Claude client
    preview_locks: dict[str, threading.Lock] = {}
    st: dict = {"work": None, "input": None, "job": None, "llm": None}

    # ---------- project ----------

    def recent_path() -> Path:
        return config_dir / "recent.json"

    def load_recent() -> list[dict]:
        try:
            return json.loads(recent_path().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []

    def remember(work: Path, inp: Path) -> None:
        items = [r for r in load_recent() if r.get("work_dir") != str(work)]
        items.insert(0, {"work_dir": str(work), "input_dir": str(inp), "opened": time.time()})
        try:
            recent_path().parent.mkdir(parents=True, exist_ok=True)
            recent_path().write_text(json.dumps(items[:MAX_RECENT], indent=2), encoding="utf-8")
        except OSError:
            pass  # a read-only home (e.g. in Docker) only loses the recent list

    def set_project(work: Path, inp: Path | None) -> None:
        work = work.expanduser().resolve()
        if inp is None:
            inp = load_input_dir(work) if work.is_dir() else None
            if inp is None:
                raise HTTPException(422, "Choose the folder with the scans too: this work "
                                         "folder doesn't say where they are.")
        inp = inp.expanduser().resolve()
        if not inp.is_dir():
            raise HTTPException(422, f"Scan folder not found: {inp}")
        if work == inp or _inside(work, inp):
            raise HTTPException(422, "The work folder must not be inside the scan folder "
                                     "(the scans are never written to).")
        if work.exists() and not work.is_dir():
            raise HTTPException(422, f"Not a folder: {work}")
        job = st["job"]
        if job is not None and job.running and (job.work_dir != work):
            raise HTTPException(409, "Detection is running for another project.")
        work.mkdir(parents=True, exist_ok=True)
        if load_input_dir(work) != inp:
            save_input_dir(work, inp)
        st["work"], st["input"] = work, inp
        preview_locks.clear()
        remember(work, inp)

    if work_dir is not None:
        wd = Path(work_dir).resolve()
        st["work"], st["input"] = wd, Path(input_dir).resolve() if input_dir else load_input_dir(wd)

    def project() -> tuple[Path, Path]:
        if st["work"] is None or st["input"] is None:
            raise HTTPException(409, "No project chosen")
        return st["work"], st["input"]

    def stats(work: Path, inp: Path) -> dict:
        counts = {"pages": 0, "approved": 0, "edited": 0, "needs_review": 0}
        for path in (work / "pages").glob("*.json"):
            try:
                pg = Page.load(path)
            except Exception:
                continue
            counts["pages"] += 1
            counts["approved"] += pg.status == "approved"
            counts["edited"] += pg.edited and pg.status != "approved"
            counts["needs_review"] += pg.status != "approved" and (
                bool(pg.issues) or any(p.needs_review for p in pg.photos))
        try:
            images = find_images(inp)
            ids = assign_page_ids([p.relative_to(inp) for p in images])
            counts["scans"] = len(images)
            counts["new"] = sum(1 for pid in ids.values() if not page_json_path(work, pid).exists())
        except (OSError, ValueError) as e:
            counts["scans"] = counts["new"] = None
            counts["scan_error"] = str(e)
        return counts

    def project_info() -> dict:
        job = st["job"]
        out = {"work_dir": None, "input_dir": None, "stats": None,
               "detect": job.status() if job else None, "llm_model": llm_model,
               "api_key": bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))}
        if st["work"] is not None and st["input"] is not None:
            out.update(work_dir=str(st["work"]), input_dir=str(st["input"]),
                       stats=stats(st["work"], st["input"]))
        return out

    @app.get("/api/project")
    def get_project() -> dict:
        return project_info()

    @app.post("/api/project")
    def post_project(body: dict = Body(...)) -> dict:
        work = (body.get("work_dir") or "").strip()
        inp = (body.get("input_dir") or "").strip()
        if not work:
            raise HTTPException(422, "Choose a work folder")
        set_project(Path(work), Path(inp) if inp else None)
        return project_info()

    @app.get("/api/recent")
    def get_recent() -> list[dict]:
        return [r for r in load_recent() if Path(r.get("work_dir", "")).is_dir()]

    @app.get("/api/fs")
    def list_dir(path: str = "") -> dict:
        """Sub-folders of a folder, so the browser can pick server-side paths."""
        if not path:
            if sys.platform == "win32":
                drives = [f"{d}:\\" for d in string.ascii_uppercase if os.path.isdir(f"{d}:\\")]
                return {"path": "", "parent": None, "dirs": [{"name": d, "path": d} for d in drives],
                        "n_images": 0, "is_work_dir": False}
            path = "/"
        p = Path(path).expanduser()
        if not p.is_dir():
            raise HTTPException(404, f"Not a folder: {path}")
        p = p.resolve()
        dirs, n_images = [], 0
        try:
            for e in os.scandir(p):
                if e.name.startswith((".", "$")):
                    continue
                try:
                    if e.is_dir():
                        dirs.append({"name": e.name, "path": str(Path(e.path))})
                    elif os.path.splitext(e.name)[1].lower() in IMAGE_EXTS:
                        n_images += 1
                except OSError:
                    continue
        except PermissionError:
            raise HTTPException(403, f"No access to {p}") from None
        dirs.sort(key=lambda d: d["name"].lower())
        parent = None if p.parent == p else str(p.parent)
        if parent is None and sys.platform == "win32":
            parent = ""  # drive root -> list of drives
        try:
            recorded = load_input_dir(p)
        except (OSError, ValueError, KeyError):
            recorded = None
        return {"path": str(p), "parent": parent, "dirs": dirs, "n_images": n_images,
                "is_work_dir": (p / "pages").is_dir() or recorded is not None,
                "input_dir": str(recorded) if recorded else None}

    # ---------- detection ----------

    def linker():
        with llm_lock:
            if st["llm"] is None:
                from ..llm import LlmLinker, LlmUnavailable
                try:
                    st["llm"] = LlmLinker(llm_model)
                except LlmUnavailable as e:
                    raise HTTPException(503, str(e)) from e
            return st["llm"]

    @app.post("/api/detect")
    def start_detect(body: dict = Body(default={})) -> dict:
        work, inp = project()
        job = st["job"]
        if job is not None and job.running:
            raise HTTPException(409, "Detection is already running")
        llm = linker() if body.get("llm") else None
        st["job"] = DetectJob(input_dir=inp, work_dir=work, redo=bool(body.get("redo")),
                              force=bool(body.get("force")), llm=llm)
        st["job"].start()
        return project_info()

    @app.get("/api/detect")
    def detect_status() -> dict | None:
        return st["job"].status() if st["job"] else None

    @app.post("/api/detect/cancel")
    def cancel_detect() -> dict | None:
        if st["job"]:
            st["job"].cancel()
        return detect_status()

    # ---------- pages ----------

    def json_path(page_id: str) -> Path:
        work, _ = project()
        # Only ids of existing page files: nothing else (e.g. "../x") maps to a path.
        pages_dir = work / "pages"
        path = page_json_path(work, page_id)
        if path.parent != pages_dir or path.name != f"{page_id}.json" or not path.is_file():
            raise HTTPException(404, f"no page {page_id!r}")
        return path

    def with_derived(page: Page, **extra) -> dict:
        d = page.to_dict()
        for pd, ph in zip(d["photos"], page.photos):
            pd["derived_caption"] = derived_caption(page, ph)
            pd["derived_date"] = derived_date(page, ph)
        d.update(extra)
        return d

    def load_scan(page: Page):
        _, inp = project()
        src = inp / page.source
        if not src.is_file():
            raise HTTPException(404, f"scan not found: {src}")
        return src

    @app.get("/", include_in_schema=False)
    def index() -> RedirectResponse:
        return RedirectResponse("/static/start.html")

    @app.get("/review", include_in_schema=False)
    def review() -> RedirectResponse:
        return RedirectResponse("/static/index.html")

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/api/pages")
    def list_pages() -> list[dict]:
        work, _ = project()
        out = []
        for path in sorted((work / "pages").glob("*.json")):
            try:
                pg = Page.load(path)
            except Exception:  # half-written or foreign file: skip, don't break the list
                continue
            linked = {lk.text_id for lk in pg.links}
            out.append({
                "page_id": pg.page_id,
                "status": pg.status,
                "edited": pg.edited,
                "n_photos": len(pg.photos),
                "n_texts": len(pg.texts),
                "n_unlinked": sum(1 for t in pg.texts if t.id not in linked),
                "needs_review": sum(1 for p in pg.photos if p.needs_review),
                "issues": len(pg.issues),
            })
        return out

    @app.get("/api/pages/{page_id}")
    def get_page(page_id: str) -> dict:
        return with_derived(Page.load(json_path(page_id)))

    @app.put("/api/pages/{page_id}")
    def put_page(page_id: str, body: dict = Body(...)) -> dict:
        path = json_path(page_id)
        try:
            new = Page.from_dict(body)
        except (KeyError, TypeError, ValueError) as e:
            raise HTTPException(422, f"bad page: {e}") from e
        if new.page_id != page_id:
            raise HTTPException(422, "page_id does not match the URL")
        if new.status not in ("proposed", "approved"):
            raise HTTPException(422, f"bad status {new.status!r}")
        with lock:
            old = Page.load(path)
            # Fields the UI cannot change come from disk.
            new.source, new.width, new.height = old.source, old.width, old.height
            new.detector_version, new.linker = old.detector_version, old.linker
            new.edited = True
            normalize(new, old)
            new.save(path)
        return with_derived(new)

    @app.post("/api/pages/{page_id}/llm")
    def llm_page(page_id: str) -> dict:
        """Ask Claude again to link and read the captions of this page as it is now."""
        from ..llm import apply
        path = json_path(page_id)
        llm = linker()
        page = Page.load(path)
        img = load_bgr(load_scan(page))
        res = llm.link(page, img, project()[0] / "llm", force=True)
        if res is None:
            raise HTTPException(502, "Claude gave no usable answer (details in the server log)")
        with lock:
            apply(page, res)
            page.edited = True
            page.save(path)
        return with_derived(page, llm_result={
            "transcribed": sum(1 for t in page.texts if t.transcription),
            "issues": len(res.issues)})

    @app.post("/api/pages/{page_id}/photos/{photo_id}/stamp")
    def llm_stamp(page_id: str, photo_id: int) -> dict:
        """Ask Claude to read the camera date stamp of one photo."""
        path = json_path(page_id)
        llm = linker()
        page = Page.load(path)
        photo = next((p for p in page.photos if p.id == photo_id), None)
        if photo is None:
            raise HTTPException(404, f"no photo {photo_id} on {page_id}")
        img = load_bgr(load_scan(page))
        ans = llm.read_stamp(page, photo, img, project()[0] / "llm", force=True)
        if ans is None:
            raise HTTPException(502, "Claude gave no usable answer (details in the server log)")
        if ans.found:
            if photo.stamp is not None:
                old = [d for d in [photo.stamp.date, *photo.stamp.date_alts] if d and d != ans.date]
                photo.stamp.text = ans.text or photo.stamp.text
                photo.stamp.date = ans.date
                photo.stamp.date_alts = list(dict.fromkeys(old))
                photo.stamp.reader = f"llm:{ans.model}"
            if ans.date and (photo.stamp is None or photo.date is not None):
                photo.date = ans.date  # no stamp box to hold it, or replace a typed date
            with lock:
                page.edited = True
                page.save(path)
        return with_derived(page, stamp_result={"found": ans.found, "text": ans.text, "date": ans.date})

    @app.get("/api/pages/{page_id}/image")
    def page_image(page_id: str) -> FileResponse:
        page = Page.load(json_path(page_id))
        src = load_scan(page)
        out = preview_path(project()[0], page_id)
        lk = preview_locks.setdefault(page_id, threading.Lock())
        with lk:
            if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
                small, _ = downscale(load_bgr(src), PREVIEW_LONG_SIDE)
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(".tmp.jpg")
                cv2.imwrite(str(tmp), small, [cv2.IMWRITE_JPEG_QUALITY, PREVIEW_QUALITY])
                tmp.replace(out)
        return FileResponse(out, media_type="image/jpeg", headers={"Cache-Control": "no-cache"})

    return app
