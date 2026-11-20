"""Optional LLM linking: Claude reads the handwritten captions and links them to photos.

Geometry always comes from OpenCV. Claude gets the downscaled page with numbered boxes
plus a full-resolution crop of each text block, and returns per text block: the photo ids
it belongs to, a transcription and a normalized date. Answers are cached per page under
`<work_dir>/llm/` keyed by a hash of everything sent, so re-runs don't pay twice.
"""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .debug import photo_color
from .imageio import downscale
from .model import Issue, Link, Page, Photo, Point

DEFAULT_MODEL = "claude-opus-5-5"
# Models offered on the start page: (id, label, note). Prices per million input/output tokens.
MODELS = [
    ("claude-opus-5-5", "Claude Opus 5.5", "default: reads handwriting well ($4 / $20)"),
    ("claude-sonnet-5-5", "Claude Sonnet 5.5", "half the price of Opus ($2 / $10)"),
    ("claude-haiku-5-5", "Claude Haiku 5.5", "cheapest, may misread hard handwriting ($0.10 / $0.50)"),
    ("claude-fable-5-2", "Claude Fable 5.2", "most capable, twice the price of Opus ($8 / $40)"),
]
# Models with a server-side refusal fallback (`fallbacks="default"`); Haiku has none.
FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5",
                   "claude-fable-5-2", "claude-fable-5-1"}
ISSUE_KINDS = ("missed_caption", "missed_photo", "merged_photos", "bad_frame", "other")
PROMPT_VERSION = 2  # bump when the prompt or schema changes, invalidates the cache
PAGE_LONG_SIDE = 1568
CROP_LONG_SIDE = 1200
JPEG_QUALITY = 88

SYSTEM_PROMPT = """\
You read scanned pages from Swedish family photo albums. Each page holds a few photo prints
on paper, with handwritten captions (mostly Swedish) in the free space beside, below or above
the prints.

You get:
1. An overview of the page. Each photo is outlined in a colour and labelled P<n> in its
   middle. Each candidate text block found by a detector is outlined in black and labelled
   T<n> on a yellow tag.
2. A close-up of each text block, labelled T<n>.
3. A list of the photos, with the date the camera printed into the photo where one was read.

For every text block, decide:
- is_caption: true if it is handwriting or lettering that describes photos. False for
  stickers, drawings, arrows, decorations, stains, punch holes and other non-text.
- photo_ids: the photos the caption belongs to. Usually the nearest photo, the one the text
  sits beside or under. A caption between two photos may describe both. A heading over
  the whole page (a place, an event, a year) applies to all photos on the page. Empty if
  is_caption is false.
- transcription: the text exactly as written, keeping the original spelling, Swedish letters
  (å ä ö) and line breaks (as \\n). Use [?] for a word you cannot read. Empty if is_caption is
  false.
- date: the date the caption states or implies, as YYYY-MM-DD, or YYYY-MM or YYYY when only
  that much follows. Resolve Swedish holidays and names of days, e.g. "Annandag jul -84" ->
  1984-12-26, "Midsommarafton 1990" -> 1990-06-22, "Påsk 95" -> 1995-04-16 (Easter Sunday),
  "Sommaren -87" -> 1987. Two-digit years like -84 or '84 are 1984; use
  20xx only if 19xx is implausible next to the camera dates. null if the text implies no date.

Return every text block exactly once, using the ids shown.

Also list in "issues" clear problems with the detected boxes, so a person can fix them:
- missed_caption: handwriting on the paper that no T box covers. photo_ids: the photo it
  belongs to. note: where it is and, if readable, what it says.
- missed_photo: a print with no P outline.
- merged_photos: one P outline that covers two or more separate prints.
- bad_frame: a P outline that clearly cuts off part of its print or takes in a lot of paper
  or a neighbouring print.
- other: anything else that would make the crop or caption wrong. Explain in note.
Write notes in English and keep them short. Report only what you can actually see. An
empty list is the normal answer for a well-detected page."""

SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "texts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "is_caption": {"type": "boolean"},
                    "photo_ids": {"type": "array", "items": {"type": "integer"}},
                    "transcription": {"type": "string"},
                    "date": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["id", "is_caption", "photo_ids", "transcription", "date"],
                "additionalProperties": False,
            },
        },
        "issues": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": list(ISSUE_KINDS)},
                    "photo_ids": {"type": "array", "items": {"type": "integer"}},
                    "note": {"type": "string"},
                },
                "required": ["kind", "photo_ids", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["texts", "issues"],
    "additionalProperties": False,
}


STAMP_PROMPT_VERSION = 1
STAMP_SYSTEM_PROMPT = """\
You read the date that a film camera printed into a photo: small orange or red
seven-segment digits, usually near a corner, such as 6 5 '95, '93 7 14 or 98 6 26. The
order of day, month and year varies between cameras. The year has two digits, between
'75 and '15. The print may lie rotated, so the stamp may run up or down the image.

Answer:
- found: true only if you can see such a camera-printed date in the photo. Handwriting,
  signs, clocks or calendars in the scene are not date stamps.
- text: the stamp exactly as printed, with the spaces and apostrophe, e.g. "6 5 '95".
  Use ? for a digit you cannot read. Empty if not found.
- date: the date as YYYY-MM-DD, or null if you cannot read it with confidence. When the
  field order is ambiguous (e.g. 6 5 '95), choose the order that matches the camera dates
  of other photos on the same page if they are given, otherwise day month year."""

STAMP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "found": {"type": "boolean"},
        "text": {"type": "string"},
        "date": {"anyOf": [{"type": "string"}, {"type": "null"}]},
    },
    "required": ["found", "text", "date"],
    "additionalProperties": False,
}


@dataclass
class StampAnswer:
    found: bool
    text: str
    date: str | None
    model: str


class LlmUnavailable(RuntimeError):
    """The SDK is missing or no credentials work: raised once, before any page is processed."""


@dataclass
class TextAnswer:
    id: int
    is_caption: bool
    photo_ids: list[int]
    transcription: str
    date: str | None


@dataclass
class LlmResult:
    texts: list[TextAnswer]
    issues: list[Issue]
    model: str
    cached: bool


_PARTIAL_DATE = re.compile(r"^(\d{4})(?:-(\d{2}))?$")


def normalize_date(s: str | None) -> str | None:
    """Keep YYYY-MM-DD, YYYY-MM or YYYY if it is a real date, else None."""
    if not s:
        return None
    s = s.strip()
    try:
        return dt.date.fromisoformat(s).isoformat() if len(s) == 10 else _partial(s)
    except ValueError:
        return None


def _partial(s: str) -> str | None:
    m = _PARTIAL_DATE.match(s)
    if not m:
        return None
    if m.group(2) and not 1 <= int(m.group(2)) <= 12:
        return None
    return s


class LlmLinker:
    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise LlmUnavailable("--llm needs the 'anthropic' package: pip install anthropic") from e
        self._anthropic = anthropic
        self.model = model
        try:
            self.client = anthropic.Anthropic()
            # Free call that checks credentials and the model id before the batch starts.
            self.client.models.retrieve(model)
        except anthropic.AuthenticationError as e:
            raise LlmUnavailable("Anthropic credentials rejected: check ANTHROPIC_API_KEY") from e
        except anthropic.NotFoundError as e:
            raise LlmUnavailable(f"Unknown model {model!r}") from e
        except anthropic.APIConnectionError as e:
            raise LlmUnavailable(f"Cannot reach the Anthropic API: {e}") from e
        except (anthropic.AnthropicError, TypeError) as e:  # TypeError: no credentials at all
            raise LlmUnavailable("no Anthropic credentials found: set ANTHROPIC_API_KEY") from e

    def link(self, page: Page, img_bgr: np.ndarray, cache_dir: Path, *, force: bool = False
             ) -> LlmResult | None:
        """Ask Claude about this page. None (with a warning) if no usable answer came back.
        `force` asks again even if an answer for exactly this request is cached."""
        # Called even for pages without text boxes: Claude may spot captions the detector missed.
        req = self.prepare(page, img_bgr)
        cache_path = cache_dir / f"{page.page_id}.json"
        data = None if force else read_cache(cache_path, req.digest)
        cached = data is not None
        if data is None:
            data = self._call(page.page_id, req.content)
            if data is None:
                return None
            write_cache(cache_path, req.digest, self.model, data)
        return self.result(page, data, cached=cached)

    def cached(self, page: Page, img_bgr: np.ndarray, cache_dir: Path
               ) -> tuple[LlmResult | None, Request]:
        """The cached answer for this page as it is now (no API call), plus the request that
        would be sent if there is none."""
        req = self.prepare(page, img_bgr)
        data = read_cache(cache_dir / f"{page.page_id}.json", req.digest)
        return (self.result(page, data, cached=True) if data is not None else None), req

    def result(self, page: Page, data: dict, *, cached: bool) -> LlmResult:
        return LlmResult(texts=_validate(page, data), issues=_validate_issues(page, data),
                         model=self.model, cached=cached)

    def prepare(self, page: Page, img_bgr: np.ndarray) -> Request:
        h = hashlib.sha256()
        h.update(f"{self.model}|{PROMPT_VERSION}|".encode())
        h.update(json.dumps(SCHEMA, sort_keys=True).encode())
        overview = _encode(render_overview(page, img_bgr))
        h.update(overview)
        content: list[dict] = [{"type": "text", "text": "Page overview:"}, _image(overview)]
        for t in page.texts:
            crop = _encode(text_crop(t, img_bgr, page.photos))
            h.update(crop)
            content += [{"type": "text", "text": f"T{t.id}:"}, _image(crop)]
        lines = []
        for ph in page.photos:
            stamp = ph.stamp.date if ph.stamp and ph.stamp.date else None
            lines.append(f"P{ph.id}: camera date {stamp}" if stamp else f"P{ph.id}: no camera date")
        ids = ", ".join(f"T{t.id}" for t in page.texts) or "(none detected)"
        prompt = ("Photos:\n" + ("\n".join(lines) or "(none detected)")
                  + f"\n\nText blocks: {ids}\nAnswer for every text block, then list any issues.")
        h.update(prompt.encode())
        content.append({"type": "text", "text": prompt})
        return Request(page_id=page.page_id, content=content, digest=h.hexdigest())

    def params(self, content: list[dict], system: str = SYSTEM_PROMPT,
               schema: dict[str, Any] = SCHEMA) -> dict:
        """Messages API parameters shared by the direct call and the Batch API."""
        return {
            "model": self.model,
            "max_tokens": 16000,
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": content}],
            "output_config": {"format": {"type": "json_schema", "schema": schema}},
        }

    def read_stamp(self, page: Page, photo: Photo, img_bgr: np.ndarray, cache_dir: Path, *,
                   force: bool = False) -> StampAnswer | None:
        """Ask Claude to read the date the camera printed into one photo. None (with a
        warning) if no usable answer came back."""
        content, digest = self.prepare_stamp(page, photo, img_bgr)
        cache_path = cache_dir / "stamps" / f"{page.page_id}__P{photo.id}.json"
        data = None if force else read_cache(cache_path, digest)
        if data is None:
            data = self._call(page.page_id, content, STAMP_SYSTEM_PROMPT, STAMP_SCHEMA)
            if data is None:
                return None
            write_cache(cache_path, digest, self.model, data)
        found = bool(data.get("found"))
        return StampAnswer(found=found, text=(data.get("text") or "").strip() if found else "",
                           date=normalize_date(data.get("date")) if found else None,
                           model=self.model)

    def prepare_stamp(self, page: Page, photo: Photo, img_bgr: np.ndarray
                      ) -> tuple[list[dict], str]:
        """Content blocks for a stamp read, plus a hash of everything sent (the cache key)."""
        h = hashlib.sha256()
        h.update(f"{self.model}|stamp{STAMP_PROMPT_VERSION}|".encode())
        h.update(json.dumps(STAMP_SCHEMA, sort_keys=True).encode())
        whole = _encode(downscale(warp_quad(img_bgr, photo.corners), PAGE_LONG_SIDE)[0])
        h.update(whole)
        content: list[dict] = [{"type": "text", "text": "The photo:"}, _image(whole)]
        st = photo.stamp
        if st is not None:
            zoom = _encode(stamp_zoom(img_bgr, st.corners))
            h.update(zoom)
            content += [{"type": "text", "text": "Close-up of where a detector found the stamp, "
                                                 "turned to reading direction:"}, _image(zoom)]
        others = sorted({p.stamp.date for p in page.photos
                         if p.id != photo.id and p.stamp and p.stamp.date})
        lines = []
        if st is not None and st.text:
            lines.append(f'An OCR read of the stamp: "{st.text}" (may be wrong).')
        if others:
            lines.append("Camera dates read on other photos of this album page: " + ", ".join(others))
        prompt = "\n".join(lines + ["Read the date stamp."])
        h.update(prompt.encode())
        content.append({"type": "text", "text": prompt})
        return content, h.hexdigest()

    def _call(self, page_id: str, content: list[dict], system: str = SYSTEM_PROMPT,
              schema: dict[str, Any] = SCHEMA) -> dict | None:
        a = self._anthropic
        # Server-side refusal fallback, where the model has one; the Batch API does not accept
        # it, so only here.
        extra = ({"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
                 if self.model in FALLBACK_MODELS else {})
        try:
            resp = self.client.beta.messages.create(**self.params(content, system, schema), **extra)
        except (a.APIStatusError, a.APIConnectionError) as e:
            _warn(page_id, f"API error: {e}")
            return None
        return parse_message(page_id, resp)


@dataclass
class Request:
    page_id: str
    content: list[dict]
    digest: str  # sha256 of everything sent; cache key and Batch API custom_id


def parse_message(page_id: str, msg) -> dict | None:
    """The JSON answer from a Messages API response, or None (with a warning)."""
    if msg.stop_reason != "end_turn":
        details = getattr(msg, "stop_details", None)
        _warn(page_id, f"stopped with {msg.stop_reason}"
              + (f" ({details.category})" if details is not None else ""))
        return None
    text = next((b.text for b in msg.content if b.type == "text"), None)
    try:
        return json.loads(text) if text else None
    except json.JSONDecodeError:
        _warn(page_id, "answer was not valid JSON")
        return None


def write_cache(path: Path, digest: str, model: str, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"hash": digest, "model": model, "response": data},
                              indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _validate(page: Page, data: dict) -> list[TextAnswer]:
    text_ids = {t.id for t in page.texts}
    photo_ids = {p.id for p in page.photos}
    out: dict[int, TextAnswer] = {}
    for item in data.get("texts", []):
        tid = item.get("id")
        if tid not in text_ids or tid in out:
            continue
        is_caption = bool(item.get("is_caption"))
        pids = sorted({p for p in item.get("photo_ids", []) if p in photo_ids}) if is_caption else []
        out[tid] = TextAnswer(id=tid, is_caption=is_caption, photo_ids=pids,
                              transcription=(item.get("transcription") or "").strip(),
                              date=normalize_date(item.get("date")) if is_caption else None)
    return [out[t] for t in sorted(out)]


def _validate_issues(page: Page, data: dict) -> list[Issue]:
    photo_ids = {p.id for p in page.photos}
    out: list[Issue] = []
    for item in data.get("issues", []):
        kind = item.get("kind")
        if kind not in ISSUE_KINDS:
            continue
        pids = sorted({p for p in item.get("photo_ids", []) if p in photo_ids})
        out.append(Issue(kind=kind, note=(item.get("note") or "").strip(), photo_ids=pids))
    return out


def apply(page: Page, result: LlmResult, *, keep_links: bool = False) -> None:
    """Replace the heuristic links and fill in transcription/date for texts Claude answered.
    Record the reported issues and flag the photos they concern for review.

    With `keep_links`, links already on the page stay as they are (a person may have set
    them) and Claude's links are only used for texts that have none yet.
    """
    page.issues = [i for i in page.issues if i.source != "llm"] + result.issues
    photos = {p.id: p for p in page.photos}
    for issue in result.issues:
        for pid in issue.photo_ids:
            photos[pid].needs_review = True
    answered = {a.id: a for a in result.texts}
    linked = {lk.text_id for lk in page.links} if keep_links else set()
    page.links = [lk for lk in page.links if lk.text_id in linked or lk.text_id not in answered]
    for t in page.texts:
        a = answered.get(t.id)
        if a is None:
            continue
        if t.id not in linked:
            page.links.extend(Link(text_id=t.id, photo_id=pid) for pid in a.photo_ids)
        t.transcription = a.transcription or None
        t.date = a.date
    page.links.sort(key=lambda lk: (lk.text_id, lk.photo_id))
    page.linker = f"llm:{result.model}"


def render_overview(page: Page, img_bgr: np.ndarray) -> np.ndarray:
    """Page with photo outlines + P<n> and text boxes + T<n>. No links, so Claude decides alone."""
    small, s = downscale(img_bgr, PAGE_LONG_SIDE)
    canvas = small.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    def pt(p) -> tuple[int, int]:
        return round(p[0] * s), round(p[1] * s)

    for ph in page.photos:
        color = photo_color(ph.id)
        shape = np.array([pt(p) for p in (ph.outline or ph.corners)], np.int32)
        cv2.polylines(canvas, [shape], True, color, 4, cv2.LINE_AA)
        cx, cy = pt(ph.center())
        _tag(canvas, f"P{ph.id}", (cx, cy), color, (255, 255, 255), font, 1.1, centered=True)

    for t in page.texts:
        x0, y0, x1, y1 = t.bbox()
        cv2.rectangle(canvas, pt((x0, y0)), pt((x1, y1)), (0, 0, 0), 3, cv2.LINE_AA)
        _tag(canvas, f"T{t.id}", pt((x0, y0)), (0, 230, 255), (0, 0, 0), font, 0.7)
    return canvas


def _tag(canvas: np.ndarray, label: str, at: tuple[int, int], bg, fg, font: int, scale: float,
         centered: bool = False) -> None:
    (tw, th), base = cv2.getTextSize(label, font, scale, 2)
    x, y = at
    if centered:
        x, y = x - tw // 2, y + th // 2
    else:  # just above the box's top-left corner, kept inside the image
        y = max(th + 4, y - 4)
    x = int(np.clip(x, 2, canvas.shape[1] - tw - 4))
    cv2.rectangle(canvas, (x - 3, y - th - 4), (x + tw + 3, y + base + 2), bg, cv2.FILLED)
    cv2.putText(canvas, label, (x, y), font, scale, fg, 2, cv2.LINE_AA)


def text_crop(t, img_bgr: np.ndarray, photos=()) -> np.ndarray:
    """Text block crop for Claude: `text_crop_full`, downscaled."""
    small, _ = downscale(text_crop_full(t, img_bgr, photos), CROP_LONG_SIDE)
    return small


def text_crop_full(t, img_bgr: np.ndarray, photos=()) -> np.ndarray:
    """Full-resolution crop of a text block, white outside its paper outline and wherever
    a photo reaches into the margin around the box."""
    H, W = img_bgr.shape[:2]
    x, y, w, h = t.box
    m = max(8, round(0.04 * max(w, h)))
    x0, y0, x1, y1 = max(0, x - m), max(0, y - m), min(W, x + w + m), min(H, y + h + m)
    crop = img_bgr[y0:y1, x0:x1].copy()

    def local(poly) -> np.ndarray:
        return np.array([(round(px - x0), round(py - y0)) for px, py in poly], np.int32)

    if t.outline:
        mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(mask, [local(t.outline)], 255)
        crop[mask == 0] = 255
    if photos:
        on_photo = np.zeros(crop.shape[:2], np.uint8)
        for ph in photos:
            cv2.fillPoly(on_photo, [local(ph.outline or ph.corners)], 255)
        on_photo[y - y0:y - y0 + h, x - x0:x - x0 + w] = 0  # the box itself stays as detected
        crop[on_photo > 0] = 255
    return crop


def quad_size(quad: list[Point]) -> tuple[int, int]:
    """Output (width, height) of `warp_quad` for a quad given as TL, TR, BR, BL."""
    q = np.array(quad, np.float32)
    w = int(round(max(np.linalg.norm(q[1] - q[0]), np.linalg.norm(q[2] - q[3]))))
    h = int(round(max(np.linalg.norm(q[3] - q[0]), np.linalg.norm(q[2] - q[1]))))
    return max(w, 1), max(h, 1)


def quad_transform(quad: list[Point]) -> tuple[np.ndarray, tuple[int, int]]:
    """Perspective matrix page -> `warp_quad` crop, and the crop's (width, height)."""
    w, h = quad_size(quad)
    dst = np.array([(0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1)], np.float32)
    return cv2.getPerspectiveTransform(np.array(quad, np.float32), dst), (w, h)


def warp_quad(img_bgr: np.ndarray, quad: list[Point], flags: int = cv2.INTER_AREA) -> np.ndarray:
    """Perspective-corrected crop of a quad given as TL, TR, BR, BL."""
    M, (w, h) = quad_transform(quad)
    return cv2.warpPerspective(img_bgr, M, (w, h), flags=flags, borderMode=cv2.BORDER_REPLICATE)


def stamp_zoom(img_bgr: np.ndarray, corners: list[Point], grow_w: float = 1.6,
               grow_h: float = 2.4, min_h: int = 160) -> np.ndarray:
    """The stamp quad with some surroundings, upright and enlarged so the digits are legible."""
    q = np.array(corners, np.float32)
    c = q.mean(axis=0)
    u = ((q[1] - q[0]) + (q[2] - q[3])) / 2 * grow_w / 2  # half-width along the text
    v = ((q[3] - q[0]) + (q[2] - q[1])) / 2 * grow_h / 2  # half-height down the text
    big = [tuple(c - u - v), tuple(c + u - v), tuple(c + u + v), tuple(c - u + v)]
    crop = warp_quad(img_bgr, big)
    if crop.shape[0] < min_h:
        f = min_h / crop.shape[0]
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return crop


def _encode(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def _image(jpeg: bytes) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg",
                                        "data": base64.standard_b64encode(jpeg).decode("ascii")}}


def read_cache(path: Path, digest: str) -> dict | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return d.get("response") if d.get("hash") == digest else None


def _warn(page_id: str, msg: str) -> None:
    print(f"  {page_id}: llm {msg}; keeping heuristic links", file=sys.stderr)
