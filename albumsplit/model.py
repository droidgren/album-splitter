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

    @classmethod
    def from_dict(cls, d: dict) -> DateStamp:
        return cls(corners=[tuple(c) for c in d["corners"]], text=d.get("text"),
                   date=d.get("date"), date_alts=list(d.get("date_alts", [])),
                   score=d.get("score", 0.0))


@dataclass
class Photo:
    id: int
    corners: list[Point]  # TL, TR, BR, BL in full-resolution pixel coordinates
    needs_review: bool = False
    # Visible part when another print lies on top of this one (polygon, full-res coords).
    # `corners` is still the whole print; export should blank out what lies outside this.
    outline: list[Point] | None = None
    stamp: DateStamp | None = None

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
    status: Status = "proposed"
    detector_version: str = ""
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
                      stamp=DateStamp.from_dict(p["stamp"]) if p.get("stamp") else None)
                for p in d.get("photos", [])
            ],
            texts=[
                TextBlock(id=t["id"], box=tuple(t["box"]),
                          transcription=t.get("transcription"), date=t.get("date"))
                for t in d.get("texts", [])
            ],
            links=[Link(**lk) for lk in d.get("links", [])],
            page_quad=[tuple(c) for c in d["page_quad"]] if d.get("page_quad") else None,
            status=d.get("status", "proposed"),
            detector_version=d.get("detector_version", ""),
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
