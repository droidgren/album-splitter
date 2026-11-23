"""Page data model and JSON (de)serialization."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePath
from typing import Literal

Point = tuple[float, float]
Status = Literal["proposed", "approved"]

SCHEMA_VERSION = 1


@dataclass
class DateStamp:
    """Date burnt into the print by the camera (orange seven-segment digits)."""
    corners: list[Point]  # TL, TR, BR, BL in the stamp's reading direction (full-res coords)
    text: str | None = None  # raw read, '?' for undecodable digits, e.g. "6 5 '95"
    date: str | None = None  # ISO date if the read parses as a valid date
    date_alts: list[str] = field(default_factory=list)  # nearly as likely readings, for review
    score: float = 0.0
    reader: str = "ocr"  # or "llm:<model>" once Claude has read it in review

    @classmethod
    def from_dict(cls, d: dict) -> DateStamp:
        return cls(corners=[tuple(c) for c in d["corners"]], text=d.get("text"),
                   date=d.get("date"), date_alts=list(d.get("date_alts", [])),
                   score=d.get("score", 0.0), reader=d.get("reader", "ocr"))


@dataclass
class DateGuess:
    """Date Claude guessed for a photo with no stamp and no dated caption, from the dates and
    captions of the neighbouring pages of an album scanned in order (`dates.py`)."""
    date: str  # YYYY, YYYY-MM or YYYY-MM-DD: only as precise as the evidence
    reason: str = ""
    confidence: str = "medium"  # low | medium | high
    model: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> DateGuess:
        return cls(date=d["date"], reason=d.get("reason", ""),
                   confidence=d.get("confidence", "medium"), model=d.get("model", ""))


@dataclass
class Photo:
    id: int
    corners: list[Point]  # TL, TR, BR, BL in full-resolution pixel coordinates
    needs_review: bool = False
    # Visible part when another print lies on top of this one (polygon, full-res coords).
    # `corners` is still the whole print; export should blank out what lies outside this.
    outline: list[Point] | None = None
    stamp: DateStamp | None = None
    # Set in review. None means derived from the linked texts and the stamp at export time
    # (see effective_caption / effective_date).
    caption: str | None = None
    date: str | None = None
    # Clockwise turn (0/90/180/270) applied at export, for a print mounted sideways.
    rotation: int = 0
    # Claude's guess from the album order; used only when nothing better dates the photo.
    date_guess: DateGuess | None = None

    def bbox(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounding box as (x0, y0, x1, y1)."""
        xs = [p[0] for p in self.corners]
        ys = [p[1] for p in self.corners]
        return min(xs), min(ys), max(xs), max(ys)

    def center(self) -> Point:
        xs = [p[0] for p in self.corners]
        ys = [p[1] for p in self.corners]
        return sum(xs) / len(xs), sum(ys) / len(ys)


@dataclass
class TextBlock:
    id: int
    box: tuple[int, int, int, int]  # x, y, w, h in full-resolution pixel coordinates
    transcription: str | None = None
    date: str | None = None
    # Part of `box` that is paper, when the box runs onto a photo (a caption wrapped around a
    # print's corner). Polygon in full-res coords; export should blank out what lies outside.
    outline: list[Point] | None = None

    def bbox(self) -> tuple[float, float, float, float]:
        x, y, w, h = self.box
        return x, y, x + w, y + h

    def center(self) -> Point:
        x, y, w, h = self.box
        return x + w / 2, y + h / 2


@dataclass
class Link:
    text_id: int
    photo_id: int


IssueKind = Literal["missed_caption", "missed_photo", "merged_photos", "bad_frame", "other",
                    "stamp_mismatch", "date_out_of_order"]


@dataclass
class Issue:
    """A detection problem to fix in review, e.g. reported by the LLM pass."""
    kind: IssueKind
    note: str = ""
    photo_ids: list[int] = field(default_factory=list)  # photos it concerns, if any
    source: str = "llm"


@dataclass
class Page:
    page_id: str
    source: str  # path of the scan relative to the input dir (POSIX style)
    width: int
    height: int
    photos: list[Photo] = field(default_factory=list)
    texts: list[TextBlock] = field(default_factory=list)
    links: list[Link] = field(default_factory=list)
    page_quad: list[Point] | None = None  # page outline if the paper does not fill the frame
    issues: list[Issue] = field(default_factory=list)
    status: Status = "proposed"
    edited: bool = False  # changed in review: re-running detect keeps it unless --force
    detector_version: str = ""
    linker: str = "heuristic"  # or "llm:<model>" when Claude linked and transcribed the texts
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> Page:
        return cls(
            page_id=d["page_id"],
            source=d["source"],
            width=d["width"],
            height=d["height"],
            photos=[
                Photo(id=p["id"], corners=[tuple(c) for c in p["corners"]],
                      needs_review=p.get("needs_review", False),
                      outline=[tuple(c) for c in p["outline"]] if p.get("outline") else None,
                      stamp=DateStamp.from_dict(p["stamp"]) if p.get("stamp") else None,
                      caption=p.get("caption"), date=p.get("date"),
                      rotation=round(float(p.get("rotation") or 0) / 90) * 90 % 360,
                      date_guess=DateGuess.from_dict(p["date_guess"]) if p.get("date_guess") else None)
                for p in d.get("photos", [])
            ],
            texts=[
                TextBlock(id=t["id"], box=tuple(t["box"]),
                          transcription=t.get("transcription"), date=t.get("date"),
                          outline=[tuple(c) for c in t["outline"]] if t.get("outline") else None)
                for t in d.get("texts", [])
            ],
            links=[Link(**lk) for lk in d.get("links", [])],
            page_quad=[tuple(c) for c in d["page_quad"]] if d.get("page_quad") else None,
            issues=[Issue(kind=i["kind"], note=i.get("note", ""),
                          photo_ids=list(i.get("photo_ids", [])), source=i.get("source", "llm"))
                    for i in d.get("issues", [])],
            status=d.get("status", "proposed"),
            edited=d.get("edited", False),
            detector_version=d.get("detector_version", ""),
            linker=d.get("linker", "heuristic"),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> Page:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def linked_texts(self, photo_id: int) -> list[TextBlock]:
        """Texts linked to a photo, in reading order (text ids follow reading order)."""
        ids = {lk.text_id for lk in self.links if lk.photo_id == photo_id}
        return sorted((t for t in self.texts if t.id in ids), key=lambda t: t.id)


_HYPHEN_BREAK = re.compile(r"(?<=[^\W\d_])-\s*\n\s*(?=[a-zåäö])")
_HYPHEN_KEEP = re.compile(r"(?<=\w)-[ \t]*\n\s*(?=\S)")


def flatten_caption(text: str) -> str:
    """One line of running text from a handwritten transcription.

    The line breaks follow the width of the paper, not the meaning, so they become spaces. A word
    split at a line break ("svinhus-\\ntaket") is joined again ("svinhustaket"). A hyphen before
    a capital or a digit ("Karl-\\nJohan", "-\\n84") is kept.
    """
    text = _HYPHEN_BREAK.sub("", text)
    text = _HYPHEN_KEEP.sub("-", text)
    return " ".join(text.split())


def derived_caption(page: Page, photo: Photo) -> str | None:
    """Caption from the linked texts' transcriptions, flattened to one line."""
    parts = [flatten_caption(t.transcription) for t in page.linked_texts(photo.id)
             if t.transcription and t.transcription.strip()]
    return " ".join(parts) or None


DateSource = Literal["review", "stamp", "caption", "guess"]


def caption_date(page: Page, photo: Photo) -> str | None:
    """The first date stated by a linked text."""
    return next((t.date for t in page.linked_texts(photo.id) if t.date), None)


def derived_date(page: Page, photo: Photo) -> str | None:
    """Camera date stamp first, then the first dated linked text, then Claude's guess."""
    if photo.stamp and photo.stamp.date:
        return photo.stamp.date
    return caption_date(page, photo) or (photo.date_guess.date if photo.date_guess else None)


def date_source(page: Page, photo: Photo) -> DateSource | None:
    """Where `effective_date` comes from, or None if the photo has no date."""
    if photo.date is not None:
        return "review" if photo.date else None
    if photo.stamp and photo.stamp.date:
        return "stamp"
    if caption_date(page, photo):
        return "caption"
    return "guess" if photo.date_guess else None


def effective_caption(page: Page, photo: Photo) -> str | None:
    """The reviewer's caption if set ("" clears it), else the derived one."""
    return derived_caption(page, photo) if photo.caption is None else (photo.caption or None)


def effective_date(page: Page, photo: Photo) -> str | None:
    return derived_date(page, photo) if photo.date is None else (photo.date or None)


_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _ascii(s: str) -> str:
    """Strip accents: 'Påsk Örebro' -> 'Pask Orebro'."""
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")


def page_id_from_path(rel_path: str | PurePath) -> str:
    """Deterministic page id from a path relative to the input dir.

    `2023/album 1/IMG_0012.jpg` -> `2023__album_1__IMG_0012`
    """
    p = PurePath(rel_path)
    parts = [*p.parts[:-1], p.stem]
    return "__".join(_UNSAFE.sub("_", _ascii(part)).strip("_") or "_" for part in parts)


# When one scan exists in several formats (IMG_0055.jpg + IMG_0055.tif), the first
# extension in this order keeps the plain id so existing work JSON stays attached to it.
_EXT_PRIORITY = (".jpg", ".jpeg", ".tif", ".tiff", ".png")


def assign_page_ids(rel_paths: list[PurePath]) -> dict[PurePath, str]:
    """Page ids for a set of files, made unique when files differ only by extension.

    `IMG_0055.jpg` -> `IMG_0055`, `IMG_0055.tif` -> `IMG_0055_tif`. Files that share a
    base id get the extension appended, except the one with the highest-priority extension.
    Raises ValueError if ids still collide.
    """
    groups: dict[str, list[PurePath]] = {}
    for rp in rel_paths:
        groups.setdefault(page_id_from_path(rp), []).append(rp)

    def rank(rp: PurePath) -> tuple[int, str]:
        ext = rp.suffix.lower()
        return (_EXT_PRIORITY.index(ext) if ext in _EXT_PRIORITY else len(_EXT_PRIORITY), ext)

    ids: dict[PurePath, str] = {}
    owner: dict[str, PurePath] = {}
    for base, members in groups.items():
        members.sort(key=rank)
        for i, rp in enumerate(members):
            pid = base if i == 0 else f"{base}_{_UNSAFE.sub('_', rp.suffix.lower().lstrip('.'))}"
            if pid in owner:
                raise ValueError(f"Page id collision: {rp} and {owner[pid]} -> {pid}")
            owner[pid] = rp
            ids[rp] = pid
    return ids
