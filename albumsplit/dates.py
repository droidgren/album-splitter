"""Optional date guesses from the album order (`--guess-dates`).

Albums are scanned in order (IMG_0001 first), and an album is kept roughly in the order the
photos were taken. So the camera stamps and caption dates on the neighbouring pages narrow down
when an undated photo was taken. Claude gets a text timeline of the album (every page and
photo with what is known about its date, plus the captions) and a small image of each photo to
date, and returns a guess per photo with its precision, confidence and reason. In the same
answer it names known dates that clearly break the order (a camera clock that was never set, a
misread stamp), which become `date_out_of_order` issues.

An album is the folder of the scans; pages are ordered by a natural sort of the file name.
A guess is stored as `photos[].date_guess` and used by `model.derived_date` only when the photo
has no stamp date and no dated caption. Approved pages give context but are never changed.
Answers are cached under `<work_dir>/llm/dates/`, keyed by a hash of everything sent.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable

import numpy as np

from .imageio import downscale, load_bgr
from .llm import (LlmLinker, _encode, _image, normalize_date, read_cache, warp_quad,
                  write_cache)
from .model import (DateGuess, Issue, Page, Photo, caption_date, effective_caption,
                    effective_date, page_id_from_path)

DATES_PROMPT_VERSION = 3
THUMB_LONG_SIDE = 512
MAX_GUESSES_PER_REQUEST = 40  # images per request stay well below the API limit
MAX_CORE_PAGES = 80           # pages a request works on ...
CONTEXT_PAGES = 25            # ... plus this many before and after, for context only
CAPTION_CHARS = 240
CONFIDENCES = ("low", "medium", "high")
NEIGHBOUR_PAGES = 3   # a suspect date is checked against known dates this many pages around it
SUSPECT_SLACK_DAYS = 120  # ... and only kept if it lies this far outside their range
ISSUE_SOURCE = "dates"

DATES_SYSTEM_PROMPT = """\
You date photos in a Swedish family photo album. The album pages were scanned in the order they
appear in the album, and the album is kept roughly in the order the photos were taken: photos on
a page were usually taken before those on later pages. The order is not strict. A page may span a
few weeks, a print was sometimes added later, and a camera's clock can be wrong.

You get a timeline of the album pages in order. For each photo it gives what is known: the date
the camera printed into the photo (stamp), a date stated by its handwritten caption, or a date a
reviewer set, plus the caption text. Photos marked G<n> have no known date; a small image of each
follows the timeline. Pages marked (context) are only there to show what comes before and after.

For each G<n>, guess when it was taken:
- date: YYYY-MM-DD, YYYY-MM or YYYY. Be only as precise as the evidence. Give a day only when
  the photo clearly belongs to a dated event (same page and same scene as a dated photo, or a
  caption naming the day). Give a month when the neighbouring dates and the scene agree on it.
  Otherwise give a year. null if there is no real clue (the known dates on both sides are years
  apart and nothing in the picture or the captions narrows it down).
- Use the nearest known dates before and after it, the captions on its own and nearby pages
  (an event, a holiday, a place, a child's age or birthday), and what the picture shows: snow,
  spring flowers, summer clothes, autumn leaves, a Christmas tree, a midsummer pole, a birthday
  cake, a graduation.
- A small image may show a camera date stamp that the stamp reader missed. The digits are hard
  to read at that size: trust them for the year and month, and give the day only when the
  neighbouring dates or the captions agree with it.
- confidence: high (almost surely right at that precision), medium (likely), low (rough).
- reason: one short sentence in English with the evidence, e.g. "Between stamps 1995-05-06
  (IMG_0011 P2) and 1995-05-20 (IMG_0013 P1), same garden as IMG_0012 P1".

Also list in "suspects" known dates on pages that are not (context) that are far from where the
album order puts them: at least about half a year off from the dates around them, e.g. a stamp
from 1994 among pages from 2002 (camera clock not set, or misread) or a stamp that contradicts the
photo's own caption by years. Dates a few days or weeks out of order are normal: do not list them.
ref: the photo as written in the timeline, e.g. "IMG_0012.png P2". note: short, in English, with
the dates that disagree. An empty list is the normal answer."""

DATES_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "guesses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "date": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "confidence": {"type": "string", "enum": list(CONFIDENCES)},
                    "reason": {"type": "string"},
                },
                "required": ["id", "date", "confidence", "reason"],
                "additionalProperties": False,
            },
        },
        "suspects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"ref": {"type": "string"}, "note": {"type": "string"}},
                "required": ["ref", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["guesses", "suspects"],
    "additionalProperties": False,
}


# ---------- albums ----------

_DIGITS = re.compile(r"(\d+)")


def natural_key(name: str) -> tuple:
    """'IMG_2' before 'IMG_10'."""
    return tuple(int(p) if p.isdigit() else p.lower() for p in _DIGITS.split(name))


@dataclass
class AlbumPage:
    path: Path  # page JSON
    page: Page

    @property
    def name(self) -> str:
        """Scan file name, as the timeline shows it."""
        return PurePosixPath(self.page.source).name


def album_of(page: Page) -> str:
    """Folder of the scan relative to the input dir; "" for the top folder."""
    a = PurePosixPath(page.source).parent.as_posix()
    return "" if a == "." else a


def albums(work_dir: Path) -> dict[str, list[AlbumPage]]:
    """Page files grouped by scan folder, each in album order."""
    out: dict[str, list[AlbumPage]] = {}
    for path in sorted((work_dir / "pages").glob("*.json")):
        try:
            page = Page.load(path)
        except (OSError, ValueError, KeyError):
            continue
        out.setdefault(album_of(page), []).append(AlbumPage(path, page))
    for pages in out.values():
        pages.sort(key=lambda ap: (natural_key(ap.name), ap.page.page_id))
    return dict(sorted(out.items()))


def needs_guess(page: Page, photo: Photo) -> bool:
    """No reviewer date (or one cleared on purpose), no stamp date and no dated caption."""
    return (photo.date is None and not (photo.stamp and photo.stamp.date)
            and caption_date(page, photo) is None)


# ---------- requests ----------

@dataclass
class Target:
    gid: int  # G<n> in the request
    page_id: str
    photo_id: int


@dataclass
class DatesRequest:
    album: str
    key: str     # cache file stem
    core: list[str]  # page ids this request works on
    text: str
    targets: list[Target]
    digest: str
    refs: dict[str, tuple[str, int]] = field(default_factory=dict)  # "IMG_1.png P2" -> photo
    # (page_id, photo_id) -> known dates on the pages around it, to sanity-check suspects
    neighbours: dict[tuple[str, int], list[str]] = field(default_factory=dict)


def _one_line(s: str, n: int = CAPTION_CHARS) -> str:
    s = " ".join(s.split()).replace('"', "'")
    return s if len(s) <= n else s[: n - 1] + "…"


def _photo_line(page: Page, ph: Photo, gid: int | None) -> str:
    if gid is not None:
        what = f"G{gid}, date unknown"
    elif needs_guess(page, ph):  # on a context page; an earlier guess is no evidence
        what = "date unknown"
    else:
        date = effective_date(page, ph)
        if ph.date is not None:
            what = f"{date} (set by reviewer)" if date else "no date (reviewer)"
        elif ph.stamp and ph.stamp.date:
            alts = [d for d in ph.stamp.date_alts if d]
            what = f"{date} (camera stamp" + (f", could also read {', '.join(alts)}" if alts else "") + ")"
        else:
            what = f"{date} (caption)"
    cap = effective_caption(page, ph)
    return f"  P{ph.id}: {what}" + (f' | caption "{_one_line(cap)}"' if cap else "")


def _page_lines(ap: AlbumPage, context: bool, gids: dict[tuple[str, int], int]) -> list[str]:
    page = ap.page
    lines = [f"{ap.name}{' (context)' if context else ''}:"]
    if not page.photos:
        lines.append("  (no photos)")
    for ph in page.photos:
        lines.append(_photo_line(page, ph, gids.get((page.page_id, ph.id))))
    linked = {lk.text_id for lk in page.links}
    loose = [_one_line(t.transcription) for t in page.texts
             if t.id not in linked and t.transcription and t.transcription.strip()]
    if loose:
        lines.append("  text on the page, not linked to a photo: "
                     + "; ".join(f'"{s}"' for s in loose))
    return lines


def _segments(pages: list[AlbumPage]) -> list[tuple[int, int]]:
    """Split an album into [start, end) runs of pages, each with few enough photos to date."""
    out, start, n = [], 0, 0
    for i, ap in enumerate(pages):
        k = sum(needs_guess(ap.page, ph) for ph in ap.page.photos)
        if i > start and (n + k > MAX_GUESSES_PER_REQUEST or i - start >= MAX_CORE_PAGES):
            out.append((start, i))
            start, n = i, 0
        n += k
    if pages:
        out.append((start, len(pages)))
    return out


def build_requests(album: str, pages: list[AlbumPage], model: str) -> list[DatesRequest]:
    """The requests for one album: one per segment with photos to date or dates to check."""
    slug = (page_id_from_path(PurePosixPath(album) / "x").rsplit("__", 1)[0]
            if album not in ("", ".") else "_top")
    out = []
    for start, end in _segments(pages):
        core = pages[start:end]
        targets = [Target(0, ap.page.page_id, ph.id) for ap in core for ph in ap.page.photos
                   if needs_guess(ap.page, ph)]
        known = sum(1 for ap in core for ph in ap.page.photos if effective_date(ap.page, ph)
                    and not needs_guess(ap.page, ph))
        if not targets and known < 3:  # nothing to date and too little to check the order
            continue
        for i, t in enumerate(targets, 1):
            t.gid = i
        gids = {(t.page_id, t.photo_id): t.gid for t in targets}
        lo, hi = max(0, start - CONTEXT_PAGES), min(len(pages), end + CONTEXT_PAGES)
        lines: list[str] = []
        refs: dict[str, tuple[str, int]] = {}
        for i in range(lo, hi):
            ap = pages[i]
            lines += _page_lines(ap, not start <= i < end, gids)
            if start <= i < end:
                for ph in ap.page.photos:
                    refs[f"{Path(ap.name).stem} P{ph.id}"] = (ap.page.page_id, ph.id)
                    refs[f"{ap.name} P{ph.id}"] = (ap.page.page_id, ph.id)
        text = "\n".join(lines)
        h = hashlib.sha256()
        h.update(f"{model}|dates{DATES_PROMPT_VERSION}|{THUMB_LONG_SIDE}|".encode())
        h.update(json.dumps(DATES_SCHEMA, sort_keys=True).encode())
        h.update(text.encode())
        by_id = {ap.page.page_id: ap.page for ap in core}
        for t in targets:  # what the thumbnail shows, without loading the scan
            pg = by_id[t.page_id]
            ph = next(p for p in pg.photos if p.id == t.photo_id)
            h.update(json.dumps([t.gid, pg.source, ph.corners, ph.outline, ph.rotation]).encode())
        out.append(DatesRequest(album=album, key=f"{slug}__{core[0].page.page_id}",
                                core=[ap.page.page_id for ap in core], text=text,
                                targets=targets, digest=h.hexdigest(), refs=refs,
                                neighbours=_neighbours(pages, start, end)))
    return out


def _neighbours(pages: list[AlbumPage], start: int, end: int) -> dict[tuple[str, int], list[str]]:
    """Known dates of the other photos within NEIGHBOUR_PAGES pages, per photo of the core."""
    known = [[(ph.id, None if needs_guess(ap.page, ph) else effective_date(ap.page, ph))
              for ph in ap.page.photos] for ap in pages]  # earlier guesses do not count
    out = {}
    for i in range(start, end):
        lo, hi = max(0, i - NEIGHBOUR_PAGES), min(len(pages), i + NEIGHBOUR_PAGES + 1)
        for pid, _ in known[i]:
            out[(pages[i].page.page_id, pid)] = [d for j in range(lo, hi) for q, d in known[j]
                                                 if d and not (j == i and q == pid)]
    return out


def _day(d: str, end: bool = False) -> dt.date:
    """First (or last) day of a YYYY / YYYY-MM / YYYY-MM-DD date."""
    if len(d) == 10:
        return dt.date.fromisoformat(d)
    y, m = int(d[:4]), int(d[5:7]) if len(d) == 7 else (12 if end else 1)
    if not end:
        return dt.date(y, m, 1)
    return dt.date(y + (m == 12), m % 12 + 1, 1) - dt.timedelta(days=1)


def far_from(date: str, others: list[str], slack_days: int = SUSPECT_SLACK_DAYS) -> bool:
    """Whether `date` lies more than `slack_days` outside the range of `others` (True if
    there are none to compare with: then Claude's judgement stands)."""
    if not others:
        return True
    lo = min(_day(d) for d in others) - dt.timedelta(days=slack_days)
    hi = max(_day(d, end=True) for d in others) + dt.timedelta(days=slack_days)
    return _day(date, end=True) < lo or _day(date) > hi


def thumbnail(img_bgr: np.ndarray, photo: Photo) -> np.ndarray:
    crop = warp_quad(img_bgr, photo.corners)
    k = (photo.rotation // 90) % 4
    if k:
        crop = np.ascontiguousarray(np.rot90(crop, -k))
    return downscale(crop, THUMB_LONG_SIDE)[0]


def request_content(req: DatesRequest, pages: dict[str, Page], input_dir: Path) -> list[dict]:
    content: list[dict] = [{"type": "text", "text": "Album timeline, in album order:\n" + req.text}]
    images: dict[str, np.ndarray] = {}
    for t in req.targets:
        pg = pages[t.page_id]
        if t.page_id not in images:
            images = {t.page_id: load_bgr(input_dir / pg.source)}  # one scan in memory at a time
        ph = next(p for p in pg.photos if p.id == t.photo_id)
        content += [{"type": "text", "text": f"G{t.gid} ({PurePosixPath(pg.source).name} P{ph.id}):"},
                    _image(_encode(thumbnail(images[t.page_id], ph)))]
    gs = ", ".join(f"G{t.gid}" for t in req.targets) or "(none)"
    content.append({"type": "text", "text": f"Photos to date: {gs}\nGuess a date for each of them, "
                                            "then list any suspect known dates."})
    return content


# ---------- answers ----------

@dataclass
class DatesAnswer:
    guesses: dict[tuple[str, int], DateGuess]  # (page_id, photo_id) -> guess; missing = no clue
    suspects: dict[str, list[tuple[int, str]]]  # page_id -> [(photo_id, note)]


def parse_answer(req: DatesRequest, data: dict, model: str,
                 pages: dict[str, Page] | None = None) -> DatesAnswer:
    """Claude's answer, checked. A suspect whose date lies close to the known dates on the
    pages around it is dropped (an album is only roughly in order); `pages` gives the dates."""
    by_gid = {t.gid: t for t in req.targets}
    guesses: dict[tuple[str, int], DateGuess] = {}
    for item in data.get("guesses", []):
        t = by_gid.get(item.get("id"))
        date = normalize_date(item.get("date"))
        if t is None or date is None or (t.page_id, t.photo_id) in guesses:
            continue
        conf = item.get("confidence") if item.get("confidence") in CONFIDENCES else "low"
        guesses[(t.page_id, t.photo_id)] = DateGuess(
            date=date, reason=(item.get("reason") or "").strip(), confidence=conf, model=model)
    suspects: dict[str, list[tuple[int, str]]] = {}
    for item in data.get("suspects", []):
        ref = " ".join(str(item.get("ref") or "").split())
        hit = req.refs.get(ref)
        if hit is None:
            continue
        page_id, pid = hit
        page = (pages or {}).get(page_id)
        ph = next((p for p in page.photos if p.id == pid), None) if page else None
        date = effective_date(page, ph) if ph else None
        if date and not far_from(date, req.neighbours.get((page_id, pid), [])):
            continue
        suspects.setdefault(page_id, []).append((pid, (item.get("note") or "").strip()))
    return DatesAnswer(guesses, suspects)


def apply_guesses(page: Page, answer: DatesAnswer) -> bool:
    """Set the guesses and order issues of one page. Returns whether anything changed.
    Photos that do not need a guess lose a stale one; reviewer fields are never touched."""
    before = json.dumps(page.to_dict(), sort_keys=True)
    for ph in page.photos:
        ph.date_guess = answer.guesses.get((page.page_id, ph.id)) if needs_guess(page, ph) else None
    issues = []
    photos = {p.id: p for p in page.photos}
    for pid, note in answer.suspects.get(page.page_id, []):
        ph = photos.get(pid)
        if ph is None or needs_guess(page, ph):
            continue
        ph.needs_review = True
        issues.append(Issue(kind="date_out_of_order", photo_ids=[pid], source=ISSUE_SOURCE,
                            note=f"P{pid} date {effective_date(page, ph)} breaks the album order"
                                 + (f": {note}" if note else "")))
    page.issues = [i for i in page.issues if i.source != ISSUE_SOURCE] + issues
    return json.dumps(page.to_dict(), sort_keys=True) != before


# ---------- the pass ----------

@dataclass
class DatesReport:
    requests: int = 0
    called: int = 0
    cached: int = 0
    failed: int = 0
    pages_updated: int = 0
    guessed: int = 0     # photos with a guess after the pass (on the pages worked on)
    undated: int = 0     # photos that still have no date
    suspects: int = 0
    cancelled: bool = False


def guess_dates(linker: LlmLinker, work_dir: Path, input_dir: Path, *, force: bool = False,
                log: Callable[[str], None] = print, stop: Callable[[], bool] = lambda: False,
                page_lock=None, on_progress: Callable[[int, int], None] | None = None,
                only_albums: set[str] | None = None) -> DatesReport:
    """Guess dates for every album in the work dir (or `only_albums`) with direct API calls.
    `force` asks again even where an answer is cached."""
    rep = DatesReport()
    cache = work_dir / "llm" / "dates"
    work: list[tuple[DatesRequest, list[AlbumPage]]] = []
    for album, pages in albums(work_dir).items():
        if only_albums is not None and album not in only_albums:
            continue
        work += [(r, pages) for r in build_requests(album, pages, linker.tag)]
    rep.requests = len(work)
    for n, (req, pages) in enumerate(work):
        if on_progress:
            on_progress(n, len(work))
        if stop():
            rep.cancelled = True
            log("Cancelled.")
            break
        t0 = time.perf_counter()
        by_id = {ap.page.page_id: ap for ap in pages}
        path = cache / f"{req.key}.json"
        data = None if force else read_cache(path, req.digest)
        was_cached = data is not None
        if was_cached:
            rep.cached += 1
        else:
            try:
                content = request_content(req, {k: v.page for k, v in by_id.items()}, input_dir)
            except (OSError, ValueError) as e:
                rep.failed += 1
                log(f"{req.key}: cannot read a scan ({e}), dates not guessed")
                continue
            data = linker._call(req.key, content, DATES_SYSTEM_PROMPT, DATES_SCHEMA)
            if data is None:
                rep.failed += 1
                continue
            write_cache(path, req.digest, linker.model, data, linker.effort)
            rep.called += 1
        ans = parse_answer(req, data, linker.model, {k: v.page for k, v in by_id.items()})
        updated = 0
        for pid in req.core:
            ap = by_id[pid]
            with page_lock or nullcontext():
                try:
                    now = Page.load(ap.path)  # it may have been edited in review meanwhile
                except (OSError, ValueError, KeyError):
                    continue
                if now.status == "approved":
                    continue
                if apply_guesses(now, ans):
                    now.save(ap.path)
                    updated += 1
            rep.guessed += sum(1 for ph in now.photos if ph.date_guess is not None)
            rep.undated += sum(1 for ph in now.photos if effective_date(now, ph) is None)
        n_sus = sum(len(v) for v in ans.suspects.values())
        rep.pages_updated += updated
        rep.suspects += n_sus
        where = req.album or "(top folder)"
        log(f"{where} {by_id[req.core[0]].name} to {by_id[req.core[-1]].name}: "
            f"{len(ans.guesses)} of {len(req.targets)} photos dated"
            + (f", {n_sus} dates out of order" if n_sus else "")
            + (" (cached)" if was_cached else "")
            + f", {updated} pages updated ({time.perf_counter() - t0:.1f}s)")
    if on_progress and not rep.cancelled:
        on_progress(len(work), len(work))
    return rep
