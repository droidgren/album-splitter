# Album Splitter

Turn scanned photo-album pages into separate photos that keep their handwritten captions.

Each scanned page usually holds 2–4 prints on white paper, with captions written next to or
below them. For every page, Album Splitter:

1. **Detects** each print (also slightly rotated ones), each block of handwriting, and the
   orange date stamp a camera burnt into the print (`6 5 '95`), then links each caption to its photo.
2. Lets you **review** and fix the result in the browser, keyboard first.
3. **Exports** each photo as a straightened full-resolution JPEG. The caption and date are written
   into the file's metadata. Optionally it also writes a "card": the photo with its handwritten
   caption underneath.

Claude (Anthropic API) is optional. It reads the handwriting, links captions on difficult pages,
double-checks date stamps and can guess dates for undated photos from the album order. Without
it, detection and linking run locally with OpenCV.

## Installation

Python 3.11 or newer.

```sh
git clone https://github.com/droidgren/album-splitter.git
cd album-splitter
python -m venv .venv
.venv\Scripts\activate          # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

To write captions and dates into the exported files you also need
[exiftool](https://exiftool.org/). It is looked for on `PATH`, in `$ALBUMSPLIT_EXIFTOOL`, and in
`.venv/exiftool-*/` (unzip the Windows package there). Without it, use `export --no-metadata`.

To use Claude, set `ANTHROPIC_API_KEY`.

## Quick start: the browser

```sh
python -m albumsplit review        # or double-click review.bat on Windows
```

This opens a start page at <http://127.0.0.1:8000/> where you:

1. pick the folder with the scans and a work folder (not inside the scan folder),
2. run detection, with or without Claude (*Batch*: half price, results arrive within hours;
   *Now*: direct calls),
3. review the pages,
4. export the approved pages.

Scans are never modified. Everything Album Splitter knows about a page is stored as JSON in the
work folder, so you can stop at any point and continue later.

### Reviewing a page

| Key / action | Does |
|---|---|
| **Enter** | approve the page and go to the next |
| **←** / **→** | previous / next page |
| **N** | next page not yet reviewed |
| **F** | only visit pages flagged for a check |
| Click a caption, then a photo | link or unlink them |
| Drag on empty paper | new caption box (**Shift**+drag: new photo) |
| Drag a corner | move the two edges that meet there (**Ctrl**+drag: only that corner) |
| **Delete** | remove the selected box |
| **R** / **Shift+R** | rotate the exported photo |
| **S** or **Alt**+drag | mark a date stamp detection missed |
| **L** | ask Claude again about this page |
| **Ctrl+Z** | undo |

Captions and dates can be edited per photo. Every change is saved immediately.

## Command line

The same steps without the browser:

```sh
# 1. Detect photos, captions and date stamps; writes <work>/pages/*.json and debug images
python -m albumsplit detect D:\scans --out D:\work
python -m albumsplit detect D:\scans --out D:\work --llm              # with Claude, direct calls
python -m albumsplit detect D:\scans --out D:\work --llm-batch --guess-dates
python -m albumsplit llm-collect D:\work --wait                         # fetch batch results

# 2. Review
python -m albumsplit review D:\work

# 3. Export approved pages
python -m albumsplit export D:\work --out D:\export
```

Useful options (see `--help` on each command):

- `detect --only "IMG_00*"`: process only some files. `--force` re-detects pages that were
  already edited or approved (these are skipped otherwise). `--no-stamps` skips date stamps.
  `--llm-model` picks the Claude model.
- `guess-dates <work>`: guess dates for undated photos again. Only use this for albums scanned
  in order.
- `export --all`: also export pages that are not approved. `--no-cards` skips the caption cards,
  `--quality 95` sets the JPEG quality.

### Input

JPG, PNG or TIFF (8 or 16 bit). Date stamp reading is tuned for uncompressed 600 dpi scans.

### Output

For each photo, in the export folder:

- `<page>_<n>.jpg`: the straightened crop at full resolution. The caption is written to
  ImageDescription / XMP `dc:description` / IPTC, and the date to `DateTimeOriginal`. Both show up
  in Windows Explorer and most photo libraries.
- `<page>_<n>_card.jpg`: the photo with its handwritten caption as a strip underneath.
- `index.csv`: every exported file with its source page, caption, date and where the date came
  from (review, stamp, caption or guess).

## Docker / QNAP NAS

An image is published at `ghcr.io/droidgren/album-splitter`.
[docker-compose.yml](docker-compose.yml) is ready for QNAP Container Station: change the volume
paths, optionally add your API key, and open `http://<nas-ip>:8000/`. Inside the container, the
scans are `/scans` (read-only), the work folder is `/work` and the export folder is `/export`.

```sh
docker run -d -p 8000:8000 -e ANTHROPIC_API_KEY \
  -v /share/scans:/scans:ro -v /share/work:/work -v /share/export:/export \
  ghcr.io/droidgren/album-splitter
```

## Development

```sh
pip install pytest
pytest
```

The design and implementation notes are in [CLAUDE.md](CLAUDE.md). Sample scans are too large
for git and go in `samples/raw/` (ignored).
