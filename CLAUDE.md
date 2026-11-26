# Album Splitter — crop photos from scanned album pages and keep their captions

## Goal
Thousands of scanned photo-album pages (JPG, white paper background, 2–4 prints per page,
handwritten Swedish captions in the free space next to/below the photos).
For each page: detect each photo, detect each handwritten text block, link text to the
right photo, let the user review/fix quickly, then export each photo with its caption.

Sample pages are uncompressed 600 dpi PNG/TIFF scans, kept out of git (too large): put them in
`samples/raw/` (ignored). `samples/raw/SOURCES.txt` lists where each one comes from.

## Architecture
Three stages, each resumable. All state per page is stored as JSON so detection can be
re-run or reviewed later without losing manual edits.

### 1. Detect (batch, CLI)
`python -m albumsplit detect <input_dir> --out <work_dir>`

- **Photos**: OpenCV. Convert to grayscale/HSV, threshold against near-white paper,
  morphological close, find external contours, keep large ones, fit `minAreaRect`
  (photos may be slightly rotated). Store 4 corner points per photo.
  Known hard cases: overlapping/touching prints (may merge into one contour — try
  splitting via approxPolyDP / convexity defects, otherwise leave to manual review),
  prints with white borders (blend into paper).
- **Text blocks**: mask out photo regions (with margin), threshold remaining ink,
  dilate to merge strokes into blocks, filter tiny noise (punch holes on the left edge,
  dust). Store axis-aligned boxes.
- **Date stamps** (`stamp.py`): the orange seven-segment camera date burnt into a print
  (`6 5 '95`, `'93 7 14`, `98 6 26`). It always lies inside the photo, near one end of a long
  edge and parallel to it. Each photo is rectified at full resolution and both long-edge bands
  are searched (the print may lie in any 90° rotation). Warm (a*/b*) contrast is measured
  against a local median, so faint pink stamps on sky survive and warm backgrounds are
  suppressed. Glyphs are read by segment levels, and the line is read as the cheapest valid
  date (years 1975–2015). Stored as `photos[].stamp` with `corners`, raw `text`, `date` and the
  near-tie `date_alts`. Near-ties are settled against the other stamps on the page. Export uses
  `stamp.date` for `DateTimeOriginal` ahead of handwritten dates. `--no-stamps` turns it off.
  Tuned for uncompressed 600 dpi scans (PNG/TIFF): the photo is shrunk with area averaging
  before the warp (`_warp`), the search runs on the raw colour and a 1 px chroma blur and all
  readings of a line vote on the date (`_vote`), italic digits are deslanted per line (`_slant`),
  a stricter threshold is tried only if nothing was found (`mask_levels`), and a date that needs
  a digit unreadable on its own (`max_digit_cost`) stays undated. Misreads are worse than misses.
  The "dotted" LED style (every segment a short separate bar with a wide red glow; Album8 and some
  old samples) is read by `_read_dotted`: bars are found from the lightness they add (top-hat),
  laid out into digit cells, and each segment measured where its bar must be. Per line, the kind
  (dotted or not) whose readings fit better votes (`Candidate.base`, without the date bonus). A
  dotted reading with a bar no cell explains, or a gap with room for a lost digit (padding only
  sits before a one-digit field), stays undated.
- **Linking (heuristic baseline)**: each text block → photo with the smallest edge-to-edge
  distance, preferring horizontal adjacency. Allow many-to-many links.
- **Linking (optional, `--llm`)**: send the downscaled page plus numbered boxes overlaid to
  Claude via the Anthropic API (`ANTHROPIC_API_KEY`). Ask for JSON: text-block → photo ids,
  transcription of each text block, and a normalized date if one is derivable
  (e.g. "Annandag jul -84" → 1984-12-26; date stamps printed in the photo like "98 6 26").
  Geometry always comes from OpenCV; the LLM only does linking + reading.
  The same request checks the date stamps: a close-up of every stamp the OCR found (`S<n>`) goes
  along with the OCR's reading, and Claude reads it again (`llm.apply_stamps`). Claude's date wins
  (the OCR's moves to `date_alts`, `reader = "llm:<model>"`). A disagreement (or Claude unable to
  read a stamp the OCR dated, which keeps the OCR date) flags the photo with a `stamp_mismatch`
  issue. On the raw samples Opus got all 41 OCR-found stamps right; Haiku leaves more wrong dates,
  but flagged. Stamps the OCR did not find at all are not checked.
- **Date guesses (optional, `--guess-dates`, `dates.py`)**: for albums scanned in order
  (IMG_0001 taken first). After the page answers are in, Claude gets a text timeline per album
  (folder of the scans, natural sort): every photo's known date (stamp with near-ties, caption,
  reviewer) and caption, plus a 512 px image of each photo with no stamp date, no dated caption
  and no reviewer date. It returns `photos[].date_guess` (`date` at the precision the evidence
  allows, `confidence`, `reason`) and names known dates that break the order (issue
  `date_out_of_order`, source `"dates"`, photo flagged). `derived_date` uses the guess last.
  Albums are split into requests of at most 40 photos to date (plus 25 context pages each side).
  Always direct calls (cheap), also after a batch. Cached in `llm/dates/`. Approved pages are
  context only. `guess-dates <work_dir>` re-runs it; off for scans not in album order.
- **Claude Code instead of the API key (`claude-code <work_dir>`, `claude_code.py`)**: runs the
  same requests through Claude Code (the user's subscription). `ClaudeCodeLinker` only replaces
  `LlmLinker._call`, so prompts, validation and `apply` are shared. Works on detected pages
  like Claude-only (`pipeline.claude_page`, also used by the start page's Claude-only job:
  links kept, approved or re-framed pages left alone). `--steps` (default all, in order):
  `pages`; `stamps` (`pipeline.stamps_page`): every photo with no stamp date and no reviewer date
  gets a stamp read (`llm.read_stamp`; an undated box gets its zoom, no box the corner
  close-ups). Without a box Claude names the corner
  (`STAMP_SCHEMA.corner`), so the stamp becomes a box over that corner (`llm.corner_quads`,
  `apply_stamp_answer`, also used by review's "Read stamp with Claude"); found elsewhere: photo
  flagged, issue source `"stamps"`; `dates`: `dates.guess_dates`. The stamp prompt gets the
  other stamps of the page as printed (`"24 5 '92" = 1992-05-24`) so it keeps the camera's
  field order. `llm.check_stamps_against_captions` (after every page answer and stamps step)
  flags a stamp date that no linked caption date agrees with (`stamp_mismatch`, source
  `"caption-check"`): on Album8 that caught exactly the dotted-style 9 -> 8 misreads. Each request becomes a folder `claude-code/requests/<key>__<hash>/`
  (`request.md` with the system prompt and `NN.jpg` images, `schema.json`, `answer.json`).
  Mode run: `claude -p` per request (`--json-schema`, `--system-prompt-file`, only the Read
  tool, no MCP, `ANTHROPIC_API_KEY` removed from its env so the login is used); `run.json`
  keeps its token usage. `--export`: only writes the folders, for a Claude Code session to
  answer; the next run applies the answers found. Answers are cached apart from the API's in
  `claude-code/llm/` as model `claude-code/<alias>` (`page.linker = "llm:claude-code/sonnet"`).
  The executable: `--claude`, `$ALBUMSPLIT_CLAUDE`, PATH, then the VS Code extension's binary.
  Start page and review (L, stamp reads) still use the API.

Output: `<work_dir>/pages/<page_id>.json` with `photos[]`, `texts[]`, `links[]`,
`status: "proposed" | "approved"`.

### 2. Review (local web UI)
`python -m albumsplit review <work_dir>` → FastAPI + a single static HTML/JS page (canvas).

- Show page with photo quads, each photo a distinct color; linked text boxes outlined in the
  same color; unlinked text in grey.
- Keyboard first: **Enter** = approve + next, **←/→** = prev/next, **N** = next unreviewed.
- Click a text box, then a photo → toggle link. Drag on empty area → new text box
  (Shift+drag → new photo). Drag corners to adjust (a photo corner moves both its edges, each parallel to itself, so the neighbouring corners follow; Ctrl+drag moves only the corner). Delete key removes selected box.
- Editable transcription/date fields per photo (prefilled if LLM was used).
- Saves to the page JSON on every change.

Implemented in `albumsplit/review/` (vanilla JS, no build step/CDN). Notes:
- `review` without a work dir (or `review.bat` with no args) opens a **start page**
  (`static/start.html`): pick the scan and work folders with a server-side folder browser,
  run detection as a background job (`review/jobs.py`, optional Claude), then start review.
  Recent projects live in `~/.albumsplit/recent.json`. The work folder may not be inside the scans.
- Claude model: chosen per project on the start page (`llm.MODELS`, saved as `llm_model` in
  `workdir.json`; the CLI's `--llm-model` is the default). Used for detection, L and stamp reads.
  Only models in `llm.FALLBACK_MODELS` get the server-side refusal fallback (Haiku has none).
- Effort: chosen next to the model (`llm.EFFORTS`, saved as `llm_effort` in `workdir.json`, CLI
  `--llm-effort`; none = the model's default). Sent as `output_config.effort`; lower effort means
  less thinking, which is most of the output cost. Cache keys use `LlmLinker.tag` (`model@effort`,
  plain model id at the default, so old caches stay valid); batches record their `effort`.
  Cache files keep the response's token `usage`, so a project's real cost can be measured.
- Detect with Claude has two modes: **Batch** (default; Message Batches API, half price; pages are
  saved with heuristic links at once, the start page polls `/api/batches` every minute and applies
  answers when a batch ends) and **Now** (direct calls). Batch results never overwrite pages that
  were approved or edited in review meanwhile; their answers stay in the `llm/` cache.
- **Guess dates from the album order** (start page, per project: `guess_dates` in `workdir.json`):
  runs at the end of a Now detection, or when the last batch is collected (`/api/batches/collect`
  starts a `DetectJob(dates_only=True)`); **Guess dates now** (`POST /api/dates`) runs it alone.
  Review marks a guessed date ("guessed", plus a chip with Claude's reason).
- **Only ask Claude again** (`claude_only`): no detection; every page not approved, edited ones
  included, is sent as it is now. `llm.apply(keep_links=True)` keeps the links on the page and
  only links texts without one; transcriptions/dates are replaced. Uses the cache, so an
  unchanged page costs nothing. Batches carry `keep_links` in `llm/batches.json`.
- Choosing a scan folder **inside** the one recorded in `workdir.json` scopes the project to it
  (`server.resolve_scan_folder`, `model.source_in_scope`): stats, review's page list, detect,
  Claude-only, date guesses and export cover only pages under it; the recorded root and the page
  ids never change. A folder outside it is refused while the work folder has pages (every id would
  change), unless the recorded folder no longer exists (scans moved) or the pages' scans are found
  under the chosen folder rather than the recorded one (`_found_under`, a wrong root recorded).
- Scans are found via `<work_dir>/workdir.json` (written by detect); `--input` overrides it
  (Docker: `review /work --input /scans --host 0.0.0.0`). Previews are cached in `<work_dir>/preview/`.
- **L** / "Ask Claude" re-runs the LLM linking+transcription on the current page as edited
  (always a fresh call). "Read stamp with Claude" per photo sends the rectified photo plus an
  upright zoom of the stamp (`llm.read_stamp`); the answer replaces `stamp.date` (old OCR dates
  move to `date_alts`, `stamp.reader = "llm:<model>"`), or sets `photo.date` if no stamp box.
  Both are cached under `<work_dir>/llm/` and undoable with Ctrl+Z. With no stamp box, the
  stamp request also carries full-resolution close-ups of the photo's four corners
  (`llm.stamp_corners`, top ones turned 180°): shrunk to page size a stamp is only a few pixels.
  Claude says which corner showed it, and the stamp gets a box there (`photo.date` only if it
  was seen elsewhere).
- **S** / "Mark stamp" (or **Alt**+drag on a photo) marks where a missed stamp is
  (`POST .../photos/<id>/stamp_box`). Only the box is saved (`stamp.orient_box` turns it to
  reading direction), as an unread `stamp` with `manual: true`: no OCR, which already searched
  there. "Read stamp with Claude" and the page request (L) send a zoom of it; Claude's reading
  fills it in. × removes a stamp box. (Keeping detection's best unreadable line as a "possible
  stamp" was tried: on the raw samples at most 1 of 42 was a real stamp, so it was dropped.)
- **F** / the ⚑ Flagged button limits navigation (←/→, N, Enter, jump list) to pages flagged for a
  check (not approved, with `issues` or a photo with `needs_review`, as counted on the start page).
  Remembered in `localStorage`.
- **E** / the ✎ button cycles all pages → only edited (`page.edited`, approved ones included) → only
  not edited. Combines with Flagged (both must let a page through); also in `localStorage`.
- `photos[].caption` / `photos[].date` are reviewer overrides; `None` = derive from the linked
  texts and the stamp (`model.effective_caption` / `effective_date`, which export must use).
  Derived captions are flattened to one line (`model.flatten_caption`: line breaks -> spaces,
  "svinhus-\ntaket" -> "svinhustaket"); `texts[].transcription` keeps the line breaks as written.
- Every save sets `page.edited`; detect skips edited and approved pages unless `--force`.
- On save, `review/geometry.normalize` drops dangling links, clears the outline and `needs_review` of
  moved photos, and recomputes `texts[].outline` for new/changed boxes.

### 3. Export (batch, CLI)
`python -m albumsplit export <work_dir> --out <export_dir>` (approved pages only by default)

Per photo:
- `<page>_<n>.jpg` — perspective-corrected clean crop, full resolution, high JPEG quality.
  Write caption to XMP/EXIF `ImageDescription` / `dc:description` and date to
  `DateTimeOriginal` if known (use exiftool or piexif).
- `<page>_<n>_card.jpg` — the photo with the linked handwritten text crop(s) composited
  as a strip underneath on a white background. Do NOT just crop a union bounding box —
  it would include parts of neighbouring photos. A caption wrapped around a print's corner
  has `texts[].outline` (its box minus the photo): blank out everything outside it.
- One `index.csv` per run: file, source page, caption text, date, date source
  (review / stamp / caption / guess).

Implemented in `export.py` + `exiftool.py`. Notes:
- Caption/date come from `model.effective_caption` / `effective_date`. Tags: EXIF ImageDescription,
  XPTitle/XPComment (Windows Explorer), UserComment, XMP dc:description/dc:title, IPTC
  Caption-Abstract (UTF-8); EXIF DateTimeOriginal/CreateDate (partial dates filled with 01) and
  XMP photoshop:DateCreated / exif:DateTimeOriginal at the real precision. Everything goes through
  a UTF-8 argfile, one exiftool run per page. `--no-metadata` skips it.
- exiftool is found via `--exiftool`, `$ALBUMSPLIT_EXIFTOOL`, PATH, then `.venv/exiftool-*/`
  (the Windows zip's `exiftool(-k).exe` waits for a key, so its bundled `perl.exe exiftool.pl`
  is run instead). The Dockerfile installs `libimage-exiftool-perl`.
- `photos[].rotation` (0/90/180/270 clockwise, default 0) turns the crop. Set in review with
  **R** / **Shift+R** (or the ↻ button on the photo card); a triangle on the canvas marks the export's top edge.
- The start page has an Export section (`review/jobs.ExportJob`, `POST /api/export`): export folder
  (remembered as `export_dir` in `workdir.json`, default `<work_dir>/export`), approved-only or all,
  cards and metadata switches. Detection and export never run at the same time.
- Card caption strips keep page scale (shrunk only if wider than the photo) and have their paper
  lifted to white (`export.whiten_paper`). Re-exporting a page deletes its stale `<page>_<n>` files.

## Conventions
- Python 3.11+, English identifiers, type hints, small modules
  (`detect.py`, `link.py`, `llm.py`, `review/`, `export.py`, `model.py`).
- Never modify source scans.
- Deterministic page ids from relative file path. Input may be JPG, PNG or TIFF (8/16-bit,
  uncompressed scans). If one scan exists in several formats, the JPG keeps the plain id and
  the others get the extension appended (`IMG_0055.tif` -> `IMG_0055_tif`).
- Should also run in Docker (target: QNAP NAS); keep a `Dockerfile` working.
  `docker-compose.yml` is the QNAP Container Station application (start page on port 8000,
  volumes `/scans` read-only, `/work`, `/export`, `/config` = `$HOME` for the recent list).
  `.github/workflows/docker.yml` pushes an amd64 image to `ghcr.io/droidgren/album-splitter`
  (public, so `pull_policy: always` needs no login).
- Version (`version.py`): `__version__` plus the git commit and its date, e.g.
  `0.1.0+a90198b (2026-11-24)` (`.dirty` with uncommitted changes). Docker has no `.git`, so the
  workflow passes `ALBUMSPLIT_COMMIT` / `ALBUMSPLIT_COMMIT_DATE` as build args. Shown by
  `albumsplit --version`, the `review` startup line (container log), `/api/version`, and in the
  corner of the start and review pages.

## First milestone
Detection + heuristic linking on the samples, plus a debug image per page with boxes and
links drawn, so detection quality can be judged before the review UI is built.
