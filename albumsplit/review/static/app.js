"use strict";
// Review UI: one page at a time on a canvas. All coordinates in the page JSON are full-resolution
// scan pixels; the preview image is drawn stretched to page.width x page.height, and the view
// transform maps those coordinates to canvas pixels.

// Same colours as debug.PALETTE (which is BGR there).
const PALETTE = ["#e62828", "#1478e6", "#1eb41e", "#ffa000", "#c828c8", "#00c8c8", "#a06414", "#780078"];
const GREY = "#969696";
const HANDLE = 7;      // handle half-size, canvas px
const DRAG_MIN = 5;    // px before a press becomes a drag
const GRAB_MIN = 12;   // px before a press on an item that isn't selected moves it (a click may wobble)
const SAVE_DELAY = 300;
const DATE_RE = String.raw`\d{4}(-\d{2}(-\d{2})?)?`;  // what export can write as DateTimeOriginal

const $ = (id) => document.getElementById(id);
const canvas = $("canvas"), ctx = canvas.getContext("2d");

const S = {
  pages: [], idx: -1, page: null, img: null,
  view: { k: 1, tx: 0, ty: 0 },
  sel: null,                 // {kind: "photo"|"text", id}
  drag: null,                // current pointer interaction
  undo: [],
  version: 0, savedVersion: 0, saveTimer: null, saving: null, saveError: false,
  space: false,
  busy: false,               // waiting for Claude: edits are blocked, its answer replaces the page
  stampFor: null,            // photo id whose date stamp the next drag marks (S / "Mark stamp")
  flaggedOnly: localStorage.getItem("albumsplit.flaggedOnly") === "1",  // navigate flagged pages only
};

const colorOf = (pid) => PALETTE[(pid - 1) % PALETTE.length];
const photoById = (id) => S.page.photos.find((p) => p.id === id);
const textById = (id) => S.page.texts.find((t) => t.id === id);
const linksOfText = (tid) => S.page.links.filter((l) => l.text_id === tid).map((l) => l.photo_id);
const clone = (o) => JSON.parse(JSON.stringify(o));

// ---------- loading & navigation ----------

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (r.status === 409 && path === "/api/pages") { location.href = "/"; throw new Error("no project"); }
  if (!r.ok) {
    const body = await r.json().catch(() => null);
    throw new Error((body && body.detail) || `${r.status} ${r.statusText}`);
  }
  return r.json();
}

async function init() {
  S.pages = await api("/api/pages");
  if (!S.pages.length) { $("hint").textContent = "No pages in the work dir."; return; }
  const want = decodeURIComponent(location.hash.slice(1));
  let i = S.pages.findIndex((p) => p.page_id === want);
  if (i < 0) i = Math.max(0, S.pages.findIndex((p) => p.status !== "approved" && visible(p)));
  await go(i);
}

// Flagged = what the start page counts as "flagged for a check".
const isFlagged = (p) => p.status !== "approved" && (p.issues > 0 || p.needs_review > 0);
const visible = (p) => !S.flaggedOnly || isFlagged(p);

// Keep the page list entry in step with the page as saved/edited, so the filter follows it.
function syncEntry(page) {
  const entry = S.pages.find((p) => p.page_id === page.page_id);
  if (!entry) return;
  entry.status = page.status;
  entry.issues = page.issues.length;
  entry.needs_review = page.photos.filter((p) => p.needs_review).length;
}

// Next visible page in direction dir (+1/-1), or -1.
function neighbour(dir) {
  for (let j = S.idx + dir; j >= 0 && j < S.pages.length; j += dir) if (visible(S.pages[j])) return j;
  return -1;
}

async function step(dir) {
  const j = neighbour(dir);
  if (j >= 0) return go(j);
  hint(S.flaggedOnly ? "No more flagged pages this way." : dir > 0 ? "Last page." : "First page.");
}

async function toggleFlagged() {
  S.flaggedOnly = !S.flaggedOnly;
  localStorage.setItem("albumsplit.flaggedOnly", S.flaggedOnly ? "1" : "0");
  renderBar();
  if (!S.flaggedOnly || isFlagged(S.pages[S.idx])) return;
  const j = nextUnreviewed();
  if (j >= 0) await go(j); else hint("No flagged pages.");
}

function loadImage(url) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error("image failed to load: " + url));
    im.src = url;
  });
}

const imageUrl = (pid) => `/api/pages/${encodeURIComponent(pid)}/image`;

async function go(i) {
  if (i < 0 || i >= S.pages.length) return;
  await flushSave();
  if (S.saveError && !confirm("The last change could not be saved. Leave this page anyway?")) return;
  const pid = S.pages[i].page_id;
  const [page, img] = await Promise.all([
    api(`/api/pages/${encodeURIComponent(pid)}`),
    loadImage(imageUrl(pid)).catch((e) => { console.error(e); return null; }),
  ]);
  S.idx = i; S.page = page; S.img = img; S.sel = null; S.undo = []; S.drag = null; S.stampFor = null;
  S.version = S.savedVersion = 0;
  history.replaceState(null, "", "#" + encodeURIComponent(pid));
  fit(); renderBar(); renderPanel(); draw();
  if (!img) hint("Scan not found - check --input");
  if (i + 1 < S.pages.length) new Image().src = imageUrl(S.pages[i + 1].page_id);  // prefetch
}

function nextUnreviewed() {
  const n = S.pages.length;
  for (let d = 1; d <= n; d++) {
    const j = (S.idx + d) % n;
    if (S.pages[j].status !== "approved" && visible(S.pages[j])) return j;
  }
  return -1;
}

async function setStatus(status) {
  mutate(() => { S.page.status = status; });
  syncEntry(S.page);
  renderBar();
}

async function approveAndNext() {
  await setStatus("approved");
  await flushSave();
  if (S.saveError) return;  // stay so nothing is lost
  S.saveError = false;
  const j = neighbour(1);
  if (j >= 0) await go(j);
  else hint(S.flaggedOnly ? "Approved. No more flagged pages after this one." : "Last page approved.");
}

// ---------- saving ----------

// Every edit goes through mutate(): snapshot for undo, apply, redraw, save.
function mutate(fn, { delay = 0, panel = true } = {}) {
  if (S.busy) { hint("Wait for Claude's answer…"); return; }
  S.undo.push(clone(S.page));
  if (S.undo.length > 200) S.undo.shift();
  fn();
  changed({ delay, panel });
}

function changed({ delay = 0, panel = true } = {}) {
  S.version++;
  if (panel) renderPanel();
  draw();
  scheduleSave(delay);
}

function scheduleSave(delay) {
  clearTimeout(S.saveTimer);
  setSaveState("saving");
  S.saveTimer = setTimeout(() => { S.saveTimer = null; save(); }, delay);
}

async function save() {
  if (S.saving) { await S.saving; }
  if (S.version === S.savedVersion) { setSaveState(S.saveError ? "error" : "saved"); return; }
  const v = S.version, pid = S.page.page_id, body = JSON.stringify(S.page);
  S.saving = (async () => {
    try {
      const saved = await api(`/api/pages/${encodeURIComponent(pid)}`, {
        method: "PUT", headers: { "Content-Type": "application/json" }, body,
      });
      S.saveError = false;
      if (S.page && S.page.page_id === pid) {
        S.savedVersion = v;
        // Take the server's outlines and derived captions, unless more edits came meanwhile.
        // Not during a drag either: it edits the page live and saves when it ends.
        if (S.version === v && !S.drag) { S.page = saved; renderPanel(); draw(); }
        const entry = S.pages.find((p) => p.page_id === pid);
        if (entry) entry.edited = true;
        syncEntry(saved);
        renderBar();
      }
    } catch (e) {
      console.error(e);
      S.saveError = true;
      hint("Save failed: " + e.message);
    }
  })();
  await S.saving;
  S.saving = null;
  setSaveState(S.saveError ? "error" : S.version === S.savedVersion ? "saved" : "saving");
}

async function flushSave() {
  if (S.saveTimer) { clearTimeout(S.saveTimer); S.saveTimer = null; }
  if (S.page) await save();
}

function setSaveState(st) {
  const el = $("save-state");
  el.className = st === "saved" ? "" : st;
  el.title = { saved: "All changes saved", saving: "Saving…", error: "Save failed" }[st];
}

window.addEventListener("beforeunload", (e) => {
  if (!S.page || S.version === S.savedVersion) return;
  // keepalive lets the request finish after the page is gone.
  fetch(`/api/pages/${encodeURIComponent(S.page.page_id)}`, {
    method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(S.page), keepalive: true,
  });
});

// ---------- geometry ----------

const toPage = (sx, sy) => [(sx - S.view.tx) / S.view.k, (sy - S.view.ty) / S.view.k];
const toScreen = (x, y) => [x * S.view.k + S.view.tx, y * S.view.k + S.view.ty];

function fit() {
  const r = canvas.parentElement.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr);
  if (!S.page) return;
  const k = Math.min(canvas.width / S.page.width, canvas.height / S.page.height) * 0.98;
  S.view = { k, tx: (canvas.width - S.page.width * k) / 2, ty: (canvas.height - S.page.height * k) / 2 };
}

function pointInPoly(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

const centroid = (pts) => [pts.reduce((a, p) => a + p[0], 0) / pts.length,
                           pts.reduce((a, p) => a + p[1], 0) / pts.length];
const textRect = (t) => { const [x, y, w, h] = t.box; return [x, y, x + w, y + h]; };
const textCenter = (t) => { const [x, y, w, h] = t.box; return [x + w / 2, y + h / 2]; };
const textCorners = (t) => { const [x0, y0, x1, y1] = textRect(t); return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]; };

// Texts are smaller and drawn on top, so they win; the smallest hit wins among each kind.
function hitItem(x, y) {
  const texts = S.page.texts.filter((t) => { const [x0, y0, x1, y1] = textRect(t); return x >= x0 && x <= x1 && y >= y0 && y <= y1; });
  if (texts.length) {
    texts.sort((a, b) => a.box[2] * a.box[3] - b.box[2] * b.box[3]);
    return { kind: "text", id: texts[0].id };
  }
  const photos = S.page.photos.filter((p) => pointInPoly(x, y, p.corners));
  if (photos.length) {
    const area = (p) => { const [x0, y0, x1, y1] = bbox(p.corners); return (x1 - x0) * (y1 - y0); };
    photos.sort((a, b) => area(a) - area(b));
    return { kind: "photo", id: photos[0].id };
  }
  return null;
}

// The smallest photo under a point (texts ignored), or null.
function photoAt(x, y) {
  const hits = S.page.photos.filter((p) => pointInPoly(x, y, p.corners));
  const area = (p) => { const [x0, y0, x1, y1] = bbox(p.corners); return (x1 - x0) * (y1 - y0); };
  hits.sort((a, b) => area(a) - area(b));
  return hits.length ? hits[0] : null;
}

function bbox(pts) {
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
}

function selectedCorners() {
  if (!S.sel) return null;
  if (S.sel.kind === "photo") { const p = photoById(S.sel.id); return p && p.corners; }
  const t = textById(S.sel.id); return t && textCorners(t);
}

function hitHandle(sx, sy) {
  const pts = selectedCorners();
  if (!pts) return -1;
  const r = HANDLE * 1.6 * (window.devicePixelRatio || 1);
  return pts.findIndex(([x, y]) => { const [hx, hy] = toScreen(x, y); return Math.abs(hx - sx) <= r && Math.abs(hy - sy) <= r; });
}

function insideSelected(x, y) {
  if (!S.sel) return false;
  if (S.sel.kind === "photo") { const p = photoById(S.sel.id); return p && pointInPoly(x, y, p.corners); }
  const t = textById(S.sel.id); if (!t) return false;
  const [x0, y0, x1, y1] = textRect(t); return x >= x0 && x <= x1 && y >= y0 && y <= y1;
}

const clampX = (x) => Math.max(0, Math.min(S.page.width - 1, x));
const clampY = (y) => Math.max(0, Math.min(S.page.height - 1, y));
const r1 = (v) => Math.round(v * 10) / 10;

// ---------- drawing ----------

function draw() {
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!S.page) return;
  const { k, tx, ty } = S.view, dpr = window.devicePixelRatio || 1;
  ctx.setTransform(k, 0, 0, k, tx, ty);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, S.page.width, S.page.height);
  if (S.img) ctx.drawImage(S.img, 0, 0, S.page.width, S.page.height);
  const px = dpr / k;  // one screen px in page units
  const pg = S.page;

  if (pg.page_quad) poly(pg.page_quad, { stroke: "#0cc", width: px, dash: [6 * px, 4 * px] });

  // Link lines under the boxes.
  for (const l of pg.links) {
    const t = textById(l.text_id), p = photoById(l.photo_id);
    if (!t || !p) continue;
    const [ax, ay] = textCenter(t), [bx, by] = centroid(p.corners);
    ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by);
    ctx.strokeStyle = colorOf(p.id); ctx.lineWidth = 2 * px; ctx.globalAlpha = 0.8; ctx.stroke(); ctx.globalAlpha = 1;
  }

  for (const p of pg.photos) {
    const c = colorOf(p.id), sel = S.sel && S.sel.kind === "photo" && S.sel.id === p.id;
    if (sel) poly(p.corners, { fill: c, alpha: 0.12 });
    if (p.outline) {  // partly covered by another print: whole print dashed, visible part solid
      poly(p.corners, { stroke: c, width: 1.5 * px, dash: [8 * px, 6 * px] });
      poly(p.outline, { stroke: c, width: (sel ? 5 : 3) * px });
    } else {
      poly(p.corners, { stroke: c, width: (sel ? 5 : 3) * px });
    }
    if (p.stamp) {
      poly(p.stamp.corners, { stroke: c, width: 1.5 * px, dash: p.stamp.manual ? [4 * px, 3 * px] : null });
      const [sx0, sy0] = bbox(p.stamp.corners);
      label(`${p.stamp.text || "?"} = ${p.stamp.date || "?"}`, sx0, sy0 - 4 * px, c, 12 * px, "bottom");
    }
    if (p.rotation || sel) upMarker(p, c, px);
    const [cx, cy] = centroid(p.outline || p.corners);
    ctx.beginPath(); ctx.arc(cx, cy, 18 * px, 0, 2 * Math.PI); ctx.fillStyle = c; ctx.fill();
    ctx.fillStyle = "#fff"; ctx.font = `bold ${18 * px}px system-ui`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(`${p.id}${p.needs_review ? "?" : ""}`, cx, cy);
  }

  for (const t of pg.texts) {
    const pids = linksOfText(t.id), c = pids.length ? colorOf(pids[0]) : GREY;
    const sel = S.sel && S.sel.kind === "text" && S.sel.id === t.id;
    const shape = t.outline || textCorners(t);
    if (sel) poly(shape, { fill: c, alpha: 0.15 });
    poly(shape, { stroke: c, width: (sel ? 4 : 2.5) * px, dash: pids.length ? null : [6 * px, 4 * px] });
    const [x0, y0] = textRect(t);
    label(`T${t.id}`, x0, y0 - 3 * px, c, 13 * px, "bottom");
  }

  // Handles of the selection.
  const pts = selectedCorners();
  if (pts) for (const [x, y] of pts) {
    ctx.fillStyle = "#fff"; ctx.strokeStyle = "#000"; ctx.lineWidth = 1.5 * px;
    ctx.fillRect(x - HANDLE * px, y - HANDLE * px, 2 * HANDLE * px, 2 * HANDLE * px);
    ctx.strokeRect(x - HANDLE * px, y - HANDLE * px, 2 * HANDLE * px, 2 * HANDLE * px);
  }

  const d = S.drag;
  if (d && d.mode === "create" && d.active) {
    const [x0, y0, x1, y1] = [Math.min(d.x0, d.x), Math.min(d.y0, d.y), Math.max(d.x0, d.x), Math.max(d.y0, d.y)];
    const color = d.stamp ? colorOf(d.stamp) : d.photo ? "#000" : "#555";
    poly([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], { stroke: color, width: 2 * px, dash: [6 * px, 4 * px] });
  }
}

// Which edge of the print is up in the export: a triangle pointing at it, just inside the frame.
// rotation turns the crop clockwise, so 90 puts the left edge on top, 180 the bottom, 270 the right.
const TOP_EDGE = { 0: [0, 1], 90: [3, 0], 180: [2, 3], 270: [1, 2] };
function upMarker(p, color, px) {
  const [i, j] = TOP_EDGE[p.rotation || 0], a = p.corners[i], b = p.corners[j];
  const [cx, cy] = centroid(p.corners);
  const mx = (a[0] + b[0]) / 2, my = (a[1] + b[1]) / 2;
  const dx = mx - cx, dy = my - cy, len = Math.hypot(dx, dy) || 1;
  const ux = dx / len, uy = dy / len, s = 16 * px;          // unit vector towards the top edge
  const tx = mx - ux * 2 * s, ty = my - uy * 2 * s;          // tip, a little inside the edge
  poly([[tx, ty], [tx - ux * 1.6 * s - uy * s, ty - uy * 1.6 * s + ux * s],
        [tx - ux * 1.6 * s + uy * s, ty - uy * 1.6 * s - ux * s]],
       { fill: color, stroke: "#fff", width: 1.5 * px, alpha: 0.9 });
}

function poly(pts, { stroke, fill, width = 1, dash = null, alpha = 1 }) {
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.closePath();
  ctx.globalAlpha = alpha;
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = width; ctx.setLineDash(dash || []); ctx.stroke(); ctx.setLineDash([]); }
  ctx.globalAlpha = 1;
}

function label(text, x, y, color, size, baseline) {
  ctx.font = `${size}px system-ui`; ctx.textAlign = "left"; ctx.textBaseline = baseline;
  const w = ctx.measureText(text).width, pad = size * 0.25;
  ctx.fillStyle = color;
  ctx.fillRect(x - pad, baseline === "bottom" ? y - size - pad : y - pad, w + 2 * pad, size + 2 * pad);
  ctx.fillStyle = "#fff"; ctx.fillText(text, x, y);
}

// ---------- pointer interaction ----------

canvas.addEventListener("contextmenu", (e) => e.preventDefault());

canvas.addEventListener("pointerdown", (e) => {
  if (!S.page) return;
  if (S.busy && e.button === 0) { hint("Wait for Claude's answer…"); return; }
  canvas.setPointerCapture(e.pointerId);
  const dpr = window.devicePixelRatio || 1, sx = e.offsetX * dpr, sy = e.offsetY * dpr;
  const [x, y] = toPage(sx, sy);
  if (e.button === 1 || e.button === 2 || S.space) {
    S.drag = { mode: "pan", sx, sy, tx: S.view.tx, ty: S.view.ty };
    return;
  }
  if (e.button !== 0) return;
  const h = hitHandle(sx, sy);
  const base = { sx, sy, x0: x, y0: y, x, y, active: false, shift: e.shiftKey, ctrl: e.ctrlKey || e.metaKey, before: clone(S.page) };
  if (S.stampFor !== null || e.altKey) {
    // Mark a date stamp: for the photo picked with S / "Mark stamp", or the one under Alt+drag.
    const p = S.stampFor !== null ? photoById(S.stampFor) : photoAt(x, y);
    if (!p) { hint("Alt+drag inside a photo to mark its date stamp."); S.drag = null; return; }
    S.drag = { ...base, mode: "create", stamp: p.id };
  } else if (h >= 0) S.drag = { ...base, mode: "handle", handle: h };
  else if (e.shiftKey) S.drag = { ...base, mode: "create", photo: true };
  else if (insideSelected(x, y)) S.drag = { ...base, mode: "move" };
  else {
    // Dragging an item that isn't selected moves it; dragging on paper draws a new text box.
    const hit = hitItem(x, y);
    S.drag = hit ? { ...base, mode: "move", grab: hit } : { ...base, mode: "create", photo: false };
  }
});

canvas.addEventListener("pointermove", (e) => {
  const dpr = window.devicePixelRatio || 1, sx = e.offsetX * dpr, sy = e.offsetY * dpr;
  const d = S.drag;
  if (!d) { updateCursor(sx, sy); return; }
  if (d.mode === "pan") {
    S.view.tx = d.tx + sx - d.sx; S.view.ty = d.ty + sy - d.sy; draw(); return;
  }
  if (!d.active && Math.hypot(sx - d.sx, sy - d.sy) < (d.grab ? GRAB_MIN : DRAG_MIN) * dpr) return;
  if (!d.active && d.grab) { S.sel = d.grab; renderPanel(); }
  d.active = true;
  const [x, y] = toPage(sx, sy);
  const dx = x - d.x, dy = y - d.y;
  d.x = x; d.y = y;
  if (d.mode === "handle") dragHandle(d.handle, clampX(x), clampY(y));
  else if (d.mode === "move") moveSelected(dx, dy);
  draw();
});

canvas.addEventListener("pointerup", (e) => {
  const d = S.drag;
  S.drag = null;
  if (!d || d.mode === "pan") return;
  if (!d.active) {
    if (d.stamp) { hint(`Drag a box around the date stamp of photo ${d.stamp} (Esc cancels).`); return; }
    click(d.x0, d.y0, d.ctrl); return;
  }
  if (d.mode === "create") { createBox(d); return; }
  // handle/move: geometry was edited live; record one undo step for the whole drag.
  S.undo.push(d.before);
  changed();
});

function updateCursor(sx, sy) {
  if (!S.page) return;
  const [x, y] = toPage(sx, sy);
  canvas.style.cursor = S.space ? "grab" : S.stampFor !== null ? "crosshair" : hitHandle(sx, sy) >= 0 ? "move"
    : insideSelected(x, y) ? "move" : hitItem(x, y) ? "pointer" : "crosshair";
}

function click(x, y, selectOnly) {
  const hit = hitItem(x, y);
  if (!hit) { select(null); return; }
  if (!selectOnly && S.sel && S.sel.kind !== hit.kind) {
    const tid = hit.kind === "text" ? hit.id : S.sel.id;
    const pid = hit.kind === "photo" ? hit.id : S.sel.id;
    toggleLink(tid, pid);
    return;
  }
  select(hit);
}

function select(sel) {
  S.sel = sel;
  renderPanel(); draw();
  if (!sel) hint("");
  else if (sel.kind === "text") hint(`T${sel.id} selected - click photos to link/unlink`);
  else hint(`Photo ${sel.id} selected - click texts to link/unlink`);
}

function toggleLink(tid, pid) {
  mutate(() => {
    const i = S.page.links.findIndex((l) => l.text_id === tid && l.photo_id === pid);
    if (i >= 0) S.page.links.splice(i, 1);
    else S.page.links.push({ text_id: tid, photo_id: pid });
  });
}

function dragHandle(h, x, y) {
  if (S.sel.kind === "photo") {
    photoById(S.sel.id).corners[h] = [r1(x), r1(y)];
    return;
  }
  const t = textById(S.sel.id);
  let [x0, y0, x1, y1] = textRect(t);
  if (h === 0 || h === 3) x0 = x; else x1 = x;
  if (h === 0 || h === 1) y0 = y; else y1 = y;
  t.box = [Math.round(Math.min(x0, x1)), Math.round(Math.min(y0, y1)),
           Math.max(1, Math.round(Math.abs(x1 - x0))), Math.max(1, Math.round(Math.abs(y1 - y0)))];
  // Corner order may flip when a handle crosses the opposite side; keep dragging the one under the mouse.
  const cs = textCorners(t);
  S.drag.handle = cs.reduce((best, c, i) => (Math.hypot(c[0] - x, c[1] - y) < Math.hypot(cs[best][0] - x, cs[best][1] - y) ? i : best), 0);
}

function moveSelected(dx, dy) {
  if (S.sel.kind === "photo") {
    const p = photoById(S.sel.id);
    p.corners = p.corners.map(([x, y]) => [r1(clampX(x + dx)), r1(clampY(y + dy))]);
    if (p.stamp) p.stamp.corners = p.stamp.corners.map(([x, y]) => [r1(x + dx), r1(y + dy)]);
  } else {
    const t = textById(S.sel.id), d = S.drag;
    d.rx = (d.rx || 0) + dx; d.ry = (d.ry || 0) + dy;  // sub-pixel remainder while dragging
    const ix = Math.trunc(d.rx), iy = Math.trunc(d.ry);
    t.box = [t.box[0] + ix, t.box[1] + iy, t.box[2], t.box[3]];
    d.rx -= ix; d.ry -= iy;
  }
}

function createBox(d) {
  const x0 = clampX(Math.min(d.x0, d.x)), y0 = clampY(Math.min(d.y0, d.y));
  const x1 = clampX(Math.max(d.x0, d.x)), y1 = clampY(Math.max(d.y0, d.y));
  if (x1 - x0 < 4 || y1 - y0 < 4) { draw(); return; }
  if (d.stamp) { markStamp(d.stamp, [[r1(x0), r1(y0)], [r1(x1), r1(y0)], [r1(x1), r1(y1)], [r1(x0), r1(y1)]]); return; }
  mutate(() => {
    if (d.photo) {
      const id = Math.max(0, ...S.page.photos.map((p) => p.id)) + 1;
      S.page.photos.push({ id, corners: [[r1(x0), r1(y0)], [r1(x1), r1(y0)], [r1(x1), r1(y1)], [r1(x0), r1(y1)]],
                           needs_review: false, outline: null, stamp: null, caption: null, date: null, rotation: 0 });
      S.sel = { kind: "photo", id };
    } else {
      const id = Math.max(0, ...S.page.texts.map((t) => t.id)) + 1;
      S.page.texts.push({ id, box: [Math.round(x0), Math.round(y0), Math.round(x1 - x0), Math.round(y1 - y0)],
                          transcription: null, date: null, outline: null });
      S.sel = { kind: "text", id };
    }
  });
  select(S.sel);
}

function rotateSelected(step) {
  if (S.busy) { hint("Wait for Claude's answer…"); return; }
  if (!S.sel || S.sel.kind !== "photo") { hint("Select a photo to rotate (R turns it clockwise, Shift+R back)."); return; }
  const id = S.sel.id;
  mutate(() => { const p = photoById(id); p.rotation = (((p.rotation || 0) + step) % 360 + 360) % 360; });
  const r = photoById(id).rotation;
  hint(`Photo ${id}: ${r ? `turned ${r}° clockwise` : "upright as on the page"} in the export (the triangle marks its top)`);
}

function deleteSelected() {
  if (!S.sel) return;
  const { kind, id } = S.sel;
  mutate(() => {
    if (kind === "photo") {
      S.page.photos = S.page.photos.filter((p) => p.id !== id);
      S.page.links = S.page.links.filter((l) => l.photo_id !== id);
    } else {
      S.page.texts = S.page.texts.filter((t) => t.id !== id);
      S.page.links = S.page.links.filter((l) => l.text_id !== id);
    }
    S.sel = null;
  });
  hint("");
}

function undo() {
  const prev = S.undo.pop();
  if (!prev) { hint("Nothing to undo"); return; }
  S.page = prev;
  if (S.sel && !(S.sel.kind === "photo" ? photoById(S.sel.id) : textById(S.sel.id))) S.sel = null;
  syncEntry(S.page);
  renderBar();
  changed();
}

canvas.addEventListener("wheel", (e) => {
  e.preventDefault();
  const dpr = window.devicePixelRatio || 1, sx = e.offsetX * dpr, sy = e.offsetY * dpr;
  const f = Math.exp(-e.deltaY * 0.0015);
  const [x, y] = toPage(sx, sy);
  S.view.k *= f;
  S.view.tx = sx - x * S.view.k; S.view.ty = sy - y * S.view.k;
  draw();
}, { passive: false });

window.addEventListener("resize", () => {
  // Keep the zoom relative to the fitted view simple: refit.
  fit(); draw();
});

// ---------- keyboard ----------

const typing = (el) => el && (el.tagName === "TEXTAREA" || (el.tagName === "INPUT" && el.type === "text") || el.tagName === "SELECT");

document.addEventListener("keydown", (e) => {
  if (typing(e.target)) {
    if (e.key === "Escape" || (e.key === "Enter" && e.target.tagName === "INPUT")) { e.target.blur(); e.preventDefault(); }
    return;
  }
  if (e.key === " ") { S.space = true; canvas.style.cursor = "grab"; e.preventDefault(); return; }
  if (e.key === "Alt") { e.preventDefault(); return; }  // (Alt+drag marks a stamp; keep the browser menu shut)
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { undo(); e.preventDefault(); return; }
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  switch (e.key) {
    case "Enter": approveAndNext(); break;
    case "ArrowRight": step(1); break;
    case "ArrowLeft": step(-1); break;
    case "n": case "N": {
      const j = nextUnreviewed();
      if (j >= 0) go(j); else hint(S.flaggedOnly ? "No other flagged pages." : "All pages approved.");
      break;
    }
    case "f": case "F": toggleFlagged(); break;
    case "u": case "U": setStatus("proposed"); break;
    case "l": case "L": askPage(); break;
    case "s": case "S": startStampMode(); break;
    case "r": rotateSelected(90); break;
    case "R": rotateSelected(-90); break;
    case "Delete": case "Backspace": deleteSelected(); break;
    case "Escape": if (S.stampFor !== null) { S.stampFor = null; hint(""); renderPanel(); draw(); } else select(null); break;
    case "0": fit(); draw(); break;
    default: return;
  }
  e.preventDefault();
});

document.addEventListener("keyup", (e) => { if (e.key === " ") { S.space = false; canvas.style.cursor = "crosshair"; } });

// ---------- bar & panel ----------

function hint(msg) { $("hint").textContent = msg; }

function renderBar() {
  const sel = $("jump");
  if (sel.options.length !== S.pages.length) {
    sel.innerHTML = "";
    for (const p of S.pages) sel.add(new Option(p.page_id, p.page_id));
  }
  S.pages.forEach((p, i) => {
    const opt = sel.options[i];
    opt.textContent = `${p.status === "approved" ? "✓" : p.edited ? "✎" : "·"}${isFlagged(p) ? " ⚑" : ""} ${p.page_id}`;
    opt.hidden = i !== S.idx && !visible(p);
  });
  sel.selectedIndex = S.idx;
  const flagged = S.pages.filter(isFlagged);
  if (S.flaggedOnly) {
    const k = flagged.indexOf(S.pages[S.idx]);
    $("pos").textContent = `${k >= 0 ? k + 1 : "–"} / ${flagged.length} flagged`;
  } else {
    $("pos").textContent = `${S.idx + 1} / ${S.pages.length}`;
  }
  const fb = $("flagged");
  fb.textContent = `⚑ Flagged (${flagged.length})`;
  fb.classList.toggle("on", S.flaggedOnly);
  fb.disabled = !S.flaggedOnly && !flagged.length;
  const b = $("status-badge");
  b.textContent = S.page.status; b.className = S.page.status;
  const done = S.pages.filter((p) => p.status === "approved").length;
  $("progress").textContent = `${done} approved`;
}

// A clicked button keeps the focus, and Enter would then press it as well as approve.
document.addEventListener("click", (e) => { if (e.target.closest("button")) e.target.closest("button").blur(); });

$("jump").addEventListener("change", (e) => { go(e.target.selectedIndex); e.target.blur(); });
$("prev").addEventListener("click", () => step(-1));
$("next").addEventListener("click", () => step(1));
$("flagged").addEventListener("click", () => toggleFlagged());
$("approve").addEventListener("click", () => approveAndNext());
$("ask").addEventListener("click", () => askPage());

// ---------- Claude ----------

// Run one request to Claude for the current page. The server saves the answer; the page
// then becomes the answer, with the previous state kept for Ctrl+Z.
async function withClaude(label, url, done, body = null) {
  if (S.busy || !S.page) return;
  await flushSave();
  if (S.saveError) return;
  const pid = S.page.page_id;
  S.busy = true; setBusy(label);
  try {
    const saved = await api(url, body === null ? { method: "POST" }
      : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (S.page && S.page.page_id === pid) {
      S.undo.push(clone(S.page));
      S.page = saved;
      S.version++; S.savedVersion = S.version;
      const entry = S.pages.find((p) => p.page_id === pid);
      if (entry) entry.edited = true;
      syncEntry(saved);
      renderBar();
      hint(done(saved));
    }
  } catch (e) {
    hint("Claude: " + e.message);
  } finally {
    S.busy = false; setBusy(null);
    renderPanel(); draw();
  }
}

function setBusy(label) {
  $("busy").hidden = !label;
  $("busy").textContent = label || "";
  $("ask").disabled = !!label;
  document.querySelectorAll("button.ask").forEach((b) => { b.disabled = !!label; });
}

// "claude-sonnet-5-5" -> "Claude Sonnet 5.5"
const modelName = (id) => !id ? "Claude" : id.replace(/^claude-/, "Claude ")
  .replace(/-(\d+)-(\d+)$/, " $1.$2").replace(/-(\d+)$/, " $1")
  .replace(/ (\w)/, (m, c) => " " + c.toUpperCase());

function askPage() {
  const pid = encodeURIComponent(S.page.page_id);
  withClaude("Claude is reading the captions…", `/api/pages/${pid}/llm`, (pg) => {
    const r = pg.llm_result;
    return `${modelName(r.model)}: ${r.transcribed} caption${r.transcribed === 1 ? "" : "s"} read`
      + (r.issues ? `, ${r.issues} issue${r.issues === 1 ? "" : "s"} reported` : "") + " (Ctrl+Z undoes)";
  });
}

function askStamp(photoId) {
  const pid = encodeURIComponent(S.page.page_id);
  withClaude(`Claude is reading the date stamp of photo ${photoId}…`, `/api/pages/${pid}/photos/${photoId}/stamp`, (pg) => {
    const r = pg.stamp_result;
    if (!r.found) return `${modelName(r.model)} sees no date stamp on photo ${photoId}.`;
    return `Photo ${photoId}: stamp "${r.text}" = ${r.date || "date unclear"} (${modelName(r.model)})`;
  });
}

function startStampMode(photoId) {
  if (S.busy) { hint("Wait for Claude's answer…"); return; }
  const id = photoId ?? (S.sel && S.sel.kind === "photo" ? S.sel.id : null);
  if (id === null) { hint("Select a photo first (or Alt+drag inside a photo) to mark its date stamp."); return; }
  S.stampFor = id;
  select({ kind: "photo", id });
  hint(`Drag a box around the date stamp of photo ${id} (Esc cancels).`);
}

// The box becomes the photo's stamp, unread: Claude reads it (its button, or L for the page).
function markStamp(photoId, box) {
  S.stampFor = null;
  const pid = encodeURIComponent(S.page.page_id);
  withClaude(`Marking the date stamp of photo ${photoId}…`, `/api/pages/${pid}/photos/${photoId}/stamp_box`,
    () => `Photo ${photoId}: stamp marked - "Read stamp with Claude" reads it (Ctrl+Z undoes)`, { box });
}

function removeStamp(photoId) {
  mutate(() => { photoById(photoId).stamp = null; });
  hint(`Photo ${photoId}: date stamp removed (Ctrl+Z undoes)`);
}

function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k.startsWith("on")) n.addEventListener(k.slice(2), v);
    else if (k === "class") n.className = v;
    else if (k === "style") n.style.cssText = v;
    else if (v !== null && v !== undefined && v !== false) n.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids.flat()) if (c !== null && c !== undefined) n.append(c);
  return n;
}

// Fit multi-line captions without scrolling (a long line wraps, so allow one extra row).
const rowsFor = (s) => Math.min(8, Math.max(2,
  s.split("\n").reduce((n, line) => n + Math.max(1, Math.ceil(line.length / 42)), 0)));

// An editable field over a nullable override: shows the derived value (greyed) while the
// override is null; typing sets it; ↺ clears it again.
function overrideField({ key, multiline, value, derived, onSet, placeholder, pattern }) {
  const isDerived = value === null || value === undefined;
  const attrs = { "data-key": key, class: isDerived ? "derived" : "", placeholder: placeholder || "",
    oninput: (e) => { e.target.className = ""; onSet(e.target.value); } };
  if (pattern) attrs.pattern = pattern;
  const text = isDerived ? (derived || "") : value;
  const input = multiline ? el("textarea", { ...attrs, rows: rowsFor(text) }) : el("input", { ...attrs, type: "text" });
  input.value = text;
  const reset = el("button", { class: "reset", title: "Back to the value from the captions/stamp", disabled: isDerived,
    onclick: () => onSet(null, true) }, "↺");
  return el("div", { class: "row" }, input, reset);
}

function renderPanel() {
  const panel = $("panel");
  // Rebuilding would drop the focus and caret of a field being typed in; restore them.
  const f = document.activeElement, fkey = f && panel.contains(f) && f.dataset.key;
  const caret = fkey ? [f.selectionStart, f.selectionEnd] : null;
  panel.innerHTML = "";
  if (!S.page) return;
  const pg = S.page;

  if (pg.issues.length) {
    panel.append(el("h3", {}, "Issues"));
    pg.issues.forEach((iss, i) => panel.append(el("div", { class: "issue" },
      el("button", { title: "Dismiss", onclick: () => mutate(() => pg.issues.splice(i, 1)) }, "×"),
      el("b", {}, iss.kind.replaceAll("_", " ")),
      iss.photo_ids.length ? ` (photo ${iss.photo_ids.join(", ")})` : "", iss.note ? ": " + iss.note : "")));
  }

  panel.append(el("h3", {}, `Photos (${pg.photos.length})`));
  if (!pg.photos.length) panel.append(el("div", { class: "muted" }, "None. Shift+drag on the page to add one."));
  for (const p of pg.photos) {
    const sel = S.sel && S.sel.kind === "photo" && S.sel.id === p.id;
    const tids = pg.links.filter((l) => l.photo_id === p.id).map((l) => "T" + l.text_id);
    const card = el("div", { class: "card" + (sel ? " selected" : ""), style: `border-left-color:${colorOf(p.id)}` },
      el("div", { class: "head", onclick: () => select({ kind: "photo", id: p.id }) },
        el("b", {}, `Photo ${p.id}`),
        p.needs_review ? el("span", { class: "review", title: "Detection is unsure about this frame" }, "check frame") : null,
        el("button", { class: "rotate", title: "Turn the exported photo 90° clockwise (R, Shift+R back)",
          onclick: (e) => { e.stopPropagation(); select({ kind: "photo", id: p.id }); rotateSelected(90); } },
          p.rotation ? `↻ ${p.rotation}°` : "↻"),
        el("span", { class: "muted" }, tids.length ? tids.join(", ") : "no text")),
      el("label", {}, "Caption"),
      overrideField({ key: `cap${p.id}`, multiline: true, value: p.caption, derived: p.derived_caption,
        placeholder: "no caption", onSet: (v, now) => setPhotoField(p.id, "caption", v, now) }),
      el("label", {}, "Date (YYYY, YYYY-MM or YYYY-MM-DD)",
        p.date == null && p.date_source === "guess"
          ? el("span", { class: "guessed", title: "No stamp or dated caption: Claude guessed it from the album order" }, " guessed") : null),
      overrideField({ key: `date${p.id}`, value: p.date, derived: p.derived_date, placeholder: "unknown", pattern: DATE_RE,
        onSet: (v, now) => setPhotoField(p.id, "date", v, now) }));
    const stampBtn = el("button", { class: "ask", disabled: S.busy, title: "Ask Claude to read the date the camera printed into this photo",
      onclick: () => askStamp(p.id) }, "Read stamp with Claude");
    const markBtn = el("button", { class: "ask", disabled: S.busy,
      title: "Drag a box around the date stamp, for Claude to read (S, or Alt+drag on the photo)",
      onclick: () => startStampMode(p.id) }, S.stampFor === p.id ? "Drag on the photo…" : p.stamp ? "Re-mark" : "Mark stamp");
    if (p.stamp) {
      const st = p.stamp, cur = p.date ?? p.derived_date;
      const by = st.reader && st.reader.startsWith("llm") ? " (Claude)" : st.manual ? " (marked)" : "";
      const head = st.manual && !st.text ? `Stamp${by}: ` : `Stamp "${st.text || "?"}"${by}: `;
      card.append(el("div", { class: "chips" }, head,
        [st.date, ...(st.date_alts || [])].filter(Boolean).map((d) =>
          el("span", { class: "chip" + (d === cur ? " on" : ""), title: "Use this date",
            onclick: () => setPhotoField(p.id, "date", d, true) }, d)),
        st.date ? null : el("span", { class: "muted" }, "unread "), stampBtn, markBtn,
        el("button", { class: "ask", disabled: S.busy, title: "Not a date stamp: remove the stamp box",
          onclick: () => removeStamp(p.id) }, "×")));
    } else {
      card.append(el("div", { class: "chips" }, el("span", { class: "muted" }, "No date stamp found "), stampBtn, markBtn));
    }
    if (p.date_guess) {
      const g = p.date_guess, cur = p.date ?? p.derived_date;
      card.append(el("div", { class: "chips", title: g.reason || "" }, `Guess from the album order (${g.confidence}): `,
        el("span", { class: "chip" + (g.date === cur ? " on" : ""), title: "Use this date", onclick: () => setPhotoField(p.id, "date", g.date, true) }, g.date),
        g.reason ? el("div", { class: "muted reason" }, g.reason) : null));
    }
    panel.append(card);
  }

  const t = S.sel && S.sel.kind === "text" ? textById(S.sel.id) : null;
  if (t) {
    const pids = linksOfText(t.id);
    panel.append(el("h3", {}, `Text T${t.id}`),
      el("div", { class: "card selected", style: `border-left-color:${pids.length ? colorOf(pids[0]) : GREY}` },
        el("div", { class: "muted" }, pids.length ? `Linked to photo ${pids.join(", ")}` : "Not linked: click a photo to link"),
        el("label", {}, "Transcription"),
        el("textarea", { "data-key": `tr${t.id}`, rows: rowsFor(t.transcription || ""),
          oninput: (e) => setTextField(t.id, "transcription", e.target.value) }, t.transcription || ""),
        el("label", {}, "Date"),
        el("input", { "data-key": `td${t.id}`, type: "text", value: t.date || "", placeholder: "none", pattern: DATE_RE,
          oninput: (e) => setTextField(t.id, "date", e.target.value) })));
  }

  const linked = new Set(pg.links.map((l) => l.text_id));
  const unlinked = pg.texts.filter((x) => !linked.has(x.id));
  if (unlinked.length) {
    panel.append(el("h3", {}, "Unlinked texts"),
      el("div", {}, unlinked.map((x) => el("span", { class: "chip", title: x.transcription || "",
        onclick: () => select({ kind: "text", id: x.id }) }, `T${x.id}`))));
  }

  if (fkey) {
    const n = panel.querySelector(`[data-key="${fkey}"]`);
    if (n) { n.focus(); if (caret && n.setSelectionRange) n.setSelectionRange(...caret); }
  }
}

// Typing: one undo step per field focus, not per keystroke; saves are debounced.
let typingKey = null;
document.addEventListener("focusout", () => { typingKey = null; });

function editField(key, fn, now) {
  if (S.busy) { hint("Wait for Claude's answer… (this edit was not kept)"); renderPanel(); return; }
  if (now || typingKey !== key) { S.undo.push(clone(S.page)); typingKey = now ? null : key; }
  fn();
  changed({ delay: now ? 0 : SAVE_DELAY, panel: now });
}

function setPhotoField(pid, field, value, now = false) {
  editField(`p${pid}${field}`, () => { photoById(pid)[field] = value; }, now);
}

function setTextField(tid, field, value) {
  editField(`t${tid}${field}`, () => { textById(tid)[field] = value.trim() === "" ? null : value; }, false);
}

init().catch((e) => { console.error(e); hint("Failed to load: " + e.message); });
