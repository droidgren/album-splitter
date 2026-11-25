"""Which build of albumsplit is running: the package version plus the git commit.

The commit comes from $ALBUMSPLIT_COMMIT / $ALBUMSPLIT_COMMIT_DATE (set by the Docker build,
which has no .git) or else from the git checkout the package runs from. A checkout with
uncommitted changes to tracked files is marked "dirty".
"""

from __future__ import annotations

import os
import subprocess
from functools import cache
from pathlib import Path

from . import __version__

REPO = Path(__file__).resolve().parent.parent


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                             timeout=3, check=True)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


@cache
def build_info() -> dict:
    """{"version", "commit", "date", "dirty"}; commit/date are None when unknown."""
    commit = os.environ.get("ALBUMSPLIT_COMMIT", "").strip()[:7] or None
    date = os.environ.get("ALBUMSPLIT_COMMIT_DATE", "").strip() or None
    dirty = False
    if commit is None and (REPO / ".git").exists():
        commit = _git("rev-parse", "--short=7", "HEAD")
        if commit:
            date = _git("show", "-s", "--format=%cs", "HEAD")
            dirty = bool(_git("status", "--porcelain", "--untracked-files=no"))
    return {"version": __version__, "commit": commit, "date": date, "dirty": dirty}


def version_string() -> str:
    """E.g. "0.1.0+a90198b (2026-11-24)", "0.1.0+a90198b.dirty (...)" or plain "0.1.0"."""
    info = build_info()
    s = info["version"]
    if info["commit"]:
        s += f"+{info['commit']}" + (".dirty" if info["dirty"] else "")
    if info["date"]:
        s += f" ({info['date']})"
    return s
