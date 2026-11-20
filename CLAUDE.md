# Album Splitter — crop photos from scanned album pages and keep their captions

## Goal
Thousands of scanned photo-album pages (JPG, white paper background, 2–4 prints per page,
handwritten Swedish captions in the free space next to/below the photos).
For each page: detect each photo, detect each handwritten text block, link text to the
right photo, let the user review/fix quickly, then export each photo with its caption.

Sample pages are in `samples/`.

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
- **Linking (heuristic baseline)**: each text block → photo with the smallest edge-to-edge
  distance, preferring horizontal adjacency. Allow many-to-many links.
- **Linking (optional, `--llm`)**: send the downscaled page plus numbered boxes overlaid to
  Claude via the Anthropic API (`ANTHROPIC_API_KEY`). Ask for JSON: text-block → photo ids,
  transcription of each text block, and a normalized date if one is derivable
  (e.g. "Annandag jul -84" → 1984-12-26; date stamps printed in the photo like "98 6 26").
  Geometry always comes from OpenCV; the LLM only does linking + reading.

Output: `<work_dir>/pages/<page_id>.json` with `photos[]`, `texts[]`, `links[]`,
`status: "proposed" | "approved"`.

### 2. Review (local web UI)
`python -m albumsplit review <work_dir>` → FastAPI + a single static HTML/JS page (canvas).

- Show page with photo quads, each photo a distinct color; linked text boxes outlined in the
  same color; unlinked text in grey.
- Keyboard first: **Enter** = approve + next, **←/→** = prev/next, **N** = next unreviewed.
- Click a text box, then a photo → toggle link. Drag on empty area → new text box
  (Shift+drag → new photo). Drag corners to adjust. Delete key removes selected box.
- Editable transcription/date fields per photo (prefilled if LLM was used).
- Saves to the page JSON on every change.

Implemented in `albumsplit/review/` (vanilla JS, no build step/CDN). Notes:
- `review` without a work dir (or `review.bat` with no args) opens a **start page**
  (`static/start.html`): pick the scan and work folders with a server-side folder browser,
  run detection as a background job (`review/jobs.py`, optional Claude), then start review.
  Recent projects live in `~/.albumsplit/recent.json`. The work folder may not be inside the scans.
- Scans are found via `<work_dir>/workdir.json` (written by detect); `--input` overrides it
  (Docker: `review /work --input /scans --host 0.0.0.0`). Previews are cached in `<work_dir>/preview/`.
- **L** / "Ask Claude" re-runs the LLM linking+transcription on the current page as edited
  (always a fresh call). "Read stamp with Claude" per photo sends the rectified photo plus an
  upright zoom of the stamp (`llm.read_stamp`); the answer replaces `stamp.date` (old OCR dates
  move to `date_alts`, `stamp.reader = "llm:<model>"`), or sets `photo.date` if no stamp box.
  Both are cached under `<work_dir>/llm/` and undoable with Ctrl+Z.
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
- One `index.csv` per run: file, source page, caption text, date.

## Conventions
- Python 3.11+, English identifiers, type hints, small modules
  (`detect.py`, `link.py`, `llm.py`, `review/`, `export.py`, `model.py`).
- Never modify source scans.
- Deterministic page ids from relative file path. Input may be JPG, PNG or TIFF (8/16-bit,
  uncompressed scans). If one scan exists in several formats, the JPG keeps the plain id and
  the others get the extension appended (`IMG_0055.tif` -> `IMG_0055_tif`).
- Should also run in Docker (target: QNAP NAS); keep a `Dockerfile` working.

## First milestone
Detection + heuristic linking on `samples/`, plus a debug image per page with boxes and
links drawn, so detection quality can be judged before the review UI is built.
