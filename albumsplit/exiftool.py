"""Write caption and date metadata to exported JPEGs with exiftool.

Captions are Swedish (åäö) and EXIF ImageDescription is officially ASCII, so the caption goes
to several tags that readers look in: EXIF ImageDescription (UTF-8, read by most tools),
XPTitle/XPComment (Windows Explorer), UserComment, XMP dc:description/dc:title and IPTC
Caption-Abstract. Everything is passed through a UTF-8 argfile, never the Windows command line.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ENV_VAR = "ALBUMSPLIT_EXIFTOOL"
_REPO = Path(__file__).resolve().parents[1]


class ExiftoolError(RuntimeError):
    pass


def _from_folder(folder: Path) -> list[str] | None:
    """Command for an unpacked Windows exiftool folder, or a folder holding `exiftool`."""
    for name in ("exiftool.exe", "exiftool"):
        if (folder / name).is_file():
            return [str(folder / name)]
    # The Windows zip ships `exiftool(-k).exe`, which waits for a key press when done.
    # Run its bundled Perl directly instead.
    perl, script = folder / "exiftool_files" / "perl.exe", folder / "exiftool_files" / "exiftool.pl"
    if perl.is_file() and script.is_file():
        return [str(perl), str(script)]
    return None


def _from_path(p: Path) -> list[str] | None:
    if p.is_dir():
        return _from_folder(p)
    if p.is_file():
        if p.name.lower() == "exiftool(-k).exe":
            return _from_folder(p.parent)
        if p.suffix.lower() == ".pl":
            bundled = p.parent / "perl.exe"
            return [str(bundled) if bundled.is_file() else "perl", str(p)]
        return [str(p)]
    return None


def find_exiftool(explicit: str | None = None) -> list[str] | None:
    """Command prefix to run exiftool, or None if it cannot be found.

    Looks at `explicit` (file or folder), $ALBUMSPLIT_EXIFTOOL, `exiftool` on PATH, then an
    unpacked `exiftool-*` folder in the project's (or the running) virtualenv.
    """
    for given in (explicit, os.environ.get(ENV_VAR)):
        if given:
            return _from_path(Path(given))
    on_path = shutil.which("exiftool")
    if on_path:
        return [on_path]
    for venv in dict.fromkeys([_REPO / ".venv", Path(sys.prefix)]):
        for folder in sorted(venv.glob("exiftool-*"), reverse=True):  # newest version first
            cmd = _from_folder(folder)
            if cmd:
                return cmd
    return None


def version(cmd: list[str]) -> str:
    res = subprocess.run([*cmd, "-ver"], capture_output=True, text=True, timeout=60)
    if res.returncode != 0:
        raise ExiftoolError(res.stderr.strip() or f"exit code {res.returncode}")
    return res.stdout.strip()


_DATE = re.compile(r"^(\d{4})(?:-(\d{2}))?(?:-(\d{2}))?$")


def exif_datetime(date: str) -> str | None:
    """EXIF date-time for an ISO date; a partial date is filled with 01 ('1984-12' -> '1984:12:01 00:00:00')."""
    m = _DATE.match(date.strip())
    if not m:
        return None
    y, mo, d = m.group(1), m.group(2) or "01", m.group(3) or "01"
    return f"{y}:{mo}:{d} 00:00:00"


def xmp_date(date: str) -> str | None:
    """XMP date keeping the known precision ('1984-12' -> '1984:12', exiftool's input form)."""
    m = _DATE.match(date.strip())
    return ":".join(g for g in m.groups() if g) if m else None


def tag_args(caption: str | None, date: str | None) -> list[str]:
    """exiftool tag assignments for one file (without the file name)."""
    args: list[str] = []
    if caption:
        for tag in ("EXIF:ImageDescription", "EXIF:XPTitle", "EXIF:XPComment", "EXIF:UserComment",
                    "XMP-dc:Description", "XMP-dc:Title", "IPTC:Caption-Abstract"):
            args.append(f"-{tag}={caption}")
        args.append("-IPTC:CodedCharacterSet=UTF8")
    if date:
        full = exif_datetime(date)
        if full:
            args += [f"-EXIF:DateTimeOriginal={full}", f"-EXIF:CreateDate={full}"]
            part = xmp_date(date)
            args += [f"-XMP-photoshop:DateCreated={part}", f"-XMP-exif:DateTimeOriginal={part}"]
    return args


def _argfile_line(arg: str) -> str:
    """One argfile line. Values with line breaks use exiftool's #[CSTR] C-string form."""
    if "\n" in arg or "\r" in arg or "\\" in arg:
        esc = (arg.replace("\\", "\\\\").replace("\r\n", "\n").replace("\r", "\n")
               .replace("\n", "\\n"))
        return "#[CSTR]" + esc
    return arg


def write_tags(cmd: list[str], jobs: list[tuple[Path, list[str]]]) -> None:
    """Tag several files in one exiftool run. Files with no tags are left alone."""
    jobs = [(path, args) for path, args in jobs if args]
    if not jobs:
        return
    lines: list[str] = []
    for path, args in jobs:
        lines += ["-charset", "utf8", "-charset", "filename=utf8", "-overwrite_original", "-m", "-q",
                  *args, str(path), "-execute"]
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".args", delete=False,
                                     newline="\n") as f:
        f.write("\n".join(_argfile_line(a) for a in lines) + "\n")
        argfile = f.name
    try:
        res = subprocess.run([*cmd, "-charset", "filename=utf8", "-@", argfile],
                             capture_output=True, timeout=600)
    finally:
        os.unlink(argfile)
    err = res.stderr.decode("utf-8", "replace").strip()
    if res.returncode != 0 or "Error" in err:
        raise ExiftoolError(err or f"exiftool exit code {res.returncode}")


def read_tags(cmd: list[str], path: Path) -> dict:
    """All tags of a file as exiftool's JSON (group-prefixed), for tests and checks."""
    import json
    res = subprocess.run([*cmd, "-j", "-G1", "-charset", "filename=utf8", "-@", "-"],
                         input=str(path).encode("utf-8"), capture_output=True, timeout=60)
    if res.returncode != 0:
        raise ExiftoolError(res.stderr.decode("utf-8", "replace").strip())
    return json.loads(res.stdout.decode("utf-8"))[0]
