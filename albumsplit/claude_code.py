"""Claude Code as the Claude backend: the same requests as `llm.py`, answered through the
subscription instead of the API key.

`ClaudeCodeLinker` replaces only `LlmLinker._call`, so page reads, stamp reads and date
guesses (`dates.py`) all work unchanged, with the normal cache layout, kept apart from the
API's under `<work_dir>/claude-code/llm/`.
Each request is written out as a folder `<work_dir>/claude-code/requests/<hash>/`:

- `request.md`: the instructions (system prompt) and the request, images as `NN.jpg`
- `schema.json`: the JSON the answer must match
- `answer.json`: the answer, written by `claude -p` (mode "run") or by a person's Claude Code
  session (mode "export": the folders are only written, a later run picks the answers up)

`claude -p` runs with only the Read tool, no MCP servers or slash commands, and without
`ANTHROPIC_API_KEY` in its environment, so it uses the Claude Code login (the subscription).
"""

from __future__ import annotations

import base64
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Literal

from .llm import EFFORTS, SCHEMA, SYSTEM_PROMPT, LlmLinker, LlmUnavailable, _warn

MODELS = ("opus", "sonnet", "haiku", "fable")
DEFAULT_MODEL = "opus"
DEFAULT_TIMEOUT = 600
Mode = Literal["run", "export"]

RUN_PROMPT = ("Answer the request in request.md in this folder. Read request.md and every image "
              "it names with the Read tool, then give the answer as JSON only.")

README = """\
# Requests from albumsplit for Claude Code

Every folder here is one request (a scanned album page, or a stretch of an album to date).
To answer one, in a Claude Code session:

1. Read `request.md` in the folder and follow its instructions. Look at every image it names
   (`NN.jpg`, with the Read tool).
2. Write the answer as JSON matching `schema.json` to `answer.json` in the same folder.
   Nothing else: do not change any other file.

Folders that already have an `answer.json` are done. Afterwards run the albumsplit command
again (`python -m albumsplit claude-code <work_dir>`): it reads the answers and applies them.
"""


def requests_dir(work_dir: Path) -> Path:
    return work_dir / "claude-code" / "requests"


def llm_cache_dir(work_dir: Path) -> Path:
    """Where the answers are cached like `llm/`, so they never replace the API's answers."""
    return work_dir / "claude-code" / "llm"


def find_claude(explicit: str | None = None) -> str:
    """The Claude Code executable: `explicit`, `$ALBUMSPLIT_CLAUDE`, PATH, then the binary
    that comes with the VS Code extension."""
    for cand in (explicit, os.environ.get("ALBUMSPLIT_CLAUDE")):
        if cand:
            if Path(cand).is_file() or shutil.which(cand):
                return shutil.which(cand) or cand
            raise LlmUnavailable(f"Claude Code not found at {cand}")
    if found := shutil.which("claude"):
        return found
    exe = "claude.exe" if sys.platform == "win32" else "claude"
    hits = []
    for ext_dir in (".vscode", ".vscode-insiders", ".cursor"):
        pattern = str(Path.home() / ext_dir / "extensions" / "anthropic.claude-code-*"
                      / "resources" / "native-binary" / exe)
        hits += glob.glob(pattern)
    if hits:
        return max(hits, key=os.path.getmtime)  # the newest extension version
    raise LlmUnavailable("Claude Code not found: install it (https://claude.com/claude-code), "
                         "or give its path with --claude or $ALBUMSPLIT_CLAUDE")


class ClaudeCodeLinker(LlmLinker):
    """An `LlmLinker` whose calls go to Claude Code. `model` is a Claude Code model alias;
    answers are cached as model `claude-code/<alias>`, apart from the API's."""

    def __init__(self, work_dir: Path, model: str = DEFAULT_MODEL, *, mode: Mode = "run",
                 effort: str | None = None, claude: str | None = None,
                 timeout: float = DEFAULT_TIMEOUT) -> None:
        if effort is not None and effort not in EFFORTS:
            raise LlmUnavailable(f"Unknown effort {effort!r}: use one of {', '.join(EFFORTS)}")
        if mode not in ("run", "export"):
            raise ValueError(f"unknown mode {mode!r}")
        self.work_dir = Path(work_dir)
        self.alias = model
        self.model = f"claude-code/{model}"
        self.effort = effort
        self.mode = mode
        self.timeout = timeout
        self.claude = find_claude(claude) if mode == "run" else None
        self.pending = 0  # requests written in export mode that still wait for an answer
        self._lock = threading.Lock()

    def _call(self, page_id: str, content: list[dict], system: str = SYSTEM_PROMPT,
              schema: dict[str, Any] = SCHEMA) -> dict | None:
        folder = self._write_request(page_id, content, system, schema)
        answer = folder / "answer.json"
        if answer.exists():
            return _load_answer(page_id, answer, schema)
        if self.mode == "export":
            with self._lock:
                self.pending += 1
            print(f"  {page_id}: waiting for an answer in {folder}", file=sys.stderr)
            return None
        data = self._run(page_id, folder, schema)
        if data is None:
            return None
        tmp = answer.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(answer)
        return data

    def _write_request(self, key: str, content: list[dict], system: str,
                       schema: dict[str, Any]) -> Path:
        """The request folder for this content (named by a hash of everything in it)."""
        h = hashlib.sha256()
        h.update(f"{self.tag}|".encode())
        h.update(system.encode())
        h.update(json.dumps(schema, sort_keys=True).encode())
        h.update(json.dumps(content, sort_keys=True).encode())
        root = requests_dir(self.work_dir)
        folder = root / f"{_safe(key)}__{h.hexdigest()[:12]}"
        if (folder / "request.md").exists():
            return folder
        folder.mkdir(parents=True, exist_ok=True)
        if not (root / "README.md").exists():
            (root / "README.md").write_text(README, encoding="utf-8")
        parts = []
        n = 0
        for block in content:
            if block["type"] == "text":
                parts.append(block["text"])
            elif block["type"] == "image":
                n += 1
                name = f"{n:02d}.jpg"
                (folder / name).write_bytes(base64.standard_b64decode(block["source"]["data"]))
                parts.append(f"[image: read {name}]")
        (folder / "system.md").write_text(system, encoding="utf-8")
        (folder / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
        md = ("# Instructions\n\n" + system + "\n\n# Request\n\n" + "\n\n".join(parts)
              + "\n\n# Answer\n\nAnswer with JSON only, matching schema.json in this folder. "
                "In a session, write it to answer.json here.\n")
        (folder / "request.md").write_text(md, encoding="utf-8")
        (folder / "meta.json").write_text(json.dumps(
            {"key": key, "model": self.model, "effort": self.effort, "images": n,
             "created": time.time()}, indent=2), encoding="utf-8")
        return folder

    def _run(self, page_id: str, folder: Path, schema: dict[str, Any]) -> dict | None:
        cmd = [self.claude, "-p", RUN_PROMPT, "--model", self.alias,
               "--output-format", "json", "--json-schema", json.dumps(schema),
               "--system-prompt-file", "system.md",
               "--tools", "Read", "--allowedTools", "Read",
               "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence"]
        if self.effort is not None:
            cmd += ["--effort", self.effort]
        env = {k: v for k, v in os.environ.items()
               if k not in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")}  # use the login
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, cwd=folder, env=env, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", timeout=self.timeout,
                                  stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            _warn(page_id, f"Claude Code took more than {self.timeout:.0f} s")
            return None
        except OSError as e:
            _warn(page_id, f"cannot start Claude Code: {e}")
            return None
        (folder / "run.log").write_text(proc.stdout + ("\n--- stderr ---\n" + proc.stderr
                                                       if proc.stderr else ""), encoding="utf-8")
        out = _parse_output(proc.stdout)
        if proc.returncode != 0 or out is None or out.get("is_error"):
            msg = (out or {}).get("result") or proc.stderr.strip() or f"exit code {proc.returncode}"
            _warn(page_id, f"Claude Code failed: {str(msg)[:300]}")
            return None
        run = {k: out[k] for k in ("total_cost_usd", "usage", "num_turns", "duration_ms",
                                   "modelUsage") if k in out}
        run["seconds"] = round(time.perf_counter() - t0, 1)
        (folder / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
        data = out.get("structured_output")
        if not isinstance(data, dict):
            data = extract_json(out.get("result") or "")
        if data is None or not _has_required(data, schema):
            _warn(page_id, "Claude Code's answer was not the expected JSON")
            return None
        return data


def _parse_output(stdout: str) -> dict | None:
    """The result object of `claude -p --output-format json`."""
    try:
        out = json.loads(stdout)
    except json.JSONDecodeError:
        return None
    if isinstance(out, list):  # some versions print every message; the result comes last
        out = next((m for m in reversed(out) if isinstance(m, dict) and m.get("type") == "result"),
                   None)
    return out if isinstance(out, dict) else None


def extract_json(text: str) -> dict | None:
    """A JSON object from an answer, also inside a ```json fence or after some words."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    elif not text.startswith("{") and "{" in text:
        text = text[text.index("{"):text.rindex("}") + 1]
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _has_required(data: dict, schema: dict[str, Any]) -> bool:
    return all(k in data for k in schema.get("required", []))


def _load_answer(page_id: str, path: Path, schema: dict[str, Any]) -> dict | None:
    try:
        data = extract_json(path.read_text(encoding="utf-8"))
    except OSError:
        data = None
    if data is None or not _has_required(data, schema):
        _warn(page_id, f"{path} is not the expected JSON")
        return None
    return data


def _safe(key: str) -> str:
    return re.sub(r"[^\w.-]+", "_", key)[:80]
