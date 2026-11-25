"use strict";
// Start page: choose the scan and work folders, run detection, go to review.

const $ = (id) => document.getElementById(id);
let pollTimer = null, exportTimer = null;
let stats = null;  // last project stats, for the re-detect-all warning

async function api(path, opts) {
  const r = await fetch(path, opts);
  const body = await r.json().catch(() => null);
  if (!r.ok) throw new Error((body && body.detail) || `${r.status} ${r.statusText}`);
  return body;
}
const post = (path, data) => api(path, { method: "POST", headers: { "Content-Type": "application/json" },
                                         body: JSON.stringify(data || {}) });

function msg(id, text, error = false) { const el = $(id); el.textContent = text || ""; el.className = "msg" + (error ? " error" : ""); }
const sepOf = (p) => (p.includes("\\") ? "\\" : "/");
const trimSep = (p) => p.replace(/[\\/]+$/, "");

// ---------- recent & project ----------

async function init() {
  const [proj, recent] = await Promise.all([api("/api/project"), api("/api/recent")]);
  renderRecent(recent, proj);
  if (proj.work_dir) { $("work-dir").value = proj.work_dir; $("input-dir").value = proj.input_dir; }
  else if (recent.length) { $("work-dir").value = recent[0].work_dir; $("input-dir").value = recent[0].input_dir; }
  renderProject(proj);
  if (proj.detect && proj.detect.state !== "finished") poll();
  if (proj.export) pollExport();
}

function renderRecent(items, proj) {
  $("recent-box").hidden = !items.length;
  const box = $("recent");
  box.innerHTML = "";
  for (const r of items) {
    const row = document.createElement("div");
    row.className = "recent";
    row.innerHTML = `<div class="paths"><div><b></b></div><div class="muted"></div></div><button>Open</button>`;
    row.querySelector("b").textContent = r.work_dir;
    row.querySelector(".muted").textContent = "scans: " + r.input_dir;
    row.title = new Date(r.opened * 1000).toLocaleString();
    row.addEventListener("click", () => { $("work-dir").value = r.work_dir; $("input-dir").value = r.input_dir; openProject(); });
    box.append(row);
  }
}

async function openProject() {
  msg("open-msg", "Opening…");
  try {
    const proj = await post("/api/project", { work_dir: $("work-dir").value, input_dir: $("input-dir").value });
    $("input-dir").value = proj.input_dir;
    $("work-dir").value = proj.work_dir;
    msg("open-msg", "");
    $("export-dir").value = "";  // the new project's own export folder
    renderProject(proj);
    renderRecent(await api("/api/recent"), proj);
  } catch (e) { msg("open-msg", e.message, true); }
}

function renderProject(proj) {
  $("project").hidden = !proj.work_dir;
  if (!proj.work_dir) return;
  const s = proj.stats;
  stats = s;
  const stat = (n, label, cls = "") => `<div class="stat ${cls}"><b>${n ?? "?"}</b>${label}</div>`;
  $("stats").innerHTML = stat(s.scans, "scans") + stat(s.pages, "pages detected")
    + stat(s.new, "not detected yet", s.new ? "warn" : "") + stat(s.approved, "approved", s.approved ? "ok" : "")
    + stat(s.edited, "edited, not approved") + stat(s.needs_review, "flagged for a check", s.needs_review ? "warn" : "");
  if (s.scan_error) msg("detect-msg", s.scan_error, true);
  renderModels(proj);
  $("opt-guess-dates").checked = !!proj.guess_dates;
  $("llm-note").textContent = proj.api_key ? ""
    : "(needs ANTHROPIC_API_KEY set before starting albumsplit, or an `ant auth login`)";
  $("llm-mode").classList.toggle("off", !$("opt-llm").checked);
  if (proj.pending && proj.pending.batches) { $("batches-box").hidden = false; schedulePollBatches(1000); }
  updateDetectButton(s);
  $("review").disabled = !s.pages;
  const left = s.pages - s.approved;
  msg("review-msg", s.pages ? `${left} of ${s.pages} pages left to approve` : "Detect the pages first.");
  if (!$("export-dir").value) $("export-dir").value = proj.export_dir || "";
  const ex = proj.exiftool || {};
  $("exif-note").textContent = ex.found ? `(exiftool ${ex.version})`
    : "(exiftool not found: install it or set ALBUMSPLIT_EXIFTOOL, then restart albumsplit)";
  if (!ex.found) $("opt-meta").checked = false;
  $("opt-meta").disabled = !ex.found;
  updateExportButton(s);
}

function renderModels(proj) {
  const sel = $("llm-model");
  if (sel.options.length !== proj.llm_models.length) {
    sel.innerHTML = "";
    for (const m of proj.llm_models) sel.add(new Option(m.label, m.id));
  }
  sel.value = proj.llm_model;
  const cur = proj.llm_models.find((m) => m.id === proj.llm_model);
  $("model-note").textContent = cur && cur.note ? cur.note + " per million tokens in / out" : "";
  const eff = $("llm-effort");
  if (eff.options.length !== proj.llm_efforts.length + 1) {
    eff.innerHTML = "";
    eff.add(new Option("Model default", ""));
    for (const e of proj.llm_efforts) eff.add(new Option(e[0].toUpperCase() + e.slice(1), e));
  }
  eff.value = proj.llm_effort || "";
}

async function setEffort() {
  try {
    renderProject(await post("/api/settings", { llm_effort: $("llm-effort").value || null }));
  } catch (e) { msg("detect-msg", e.message, true); }
}

async function setModel() {
  try {
    renderProject(await post("/api/settings", { llm_model: $("llm-model").value }));
  } catch (e) { msg("detect-msg", e.message, true); }
}

async function setGuessDates() {
  try {
    renderProject(await post("/api/settings", { guess_dates: $("opt-guess-dates").checked }));
  } catch (e) { msg("detect-msg", e.message, true); }
}

async function guessDatesNow() {
  msg("detect-msg", "");
  try {
    renderProject(await post("/api/dates"));
    poll();
  } catch (e) { msg("detect-msg", e.message, true); }
}

const llmMode = () => document.querySelector('input[name="llm-mode"]:checked').value;

function updateExportButton(s) {
  const n = $("opt-all").checked ? s.pages : s.approved;
  $("export").textContent = n ? `Export ${n} page${n === 1 ? "" : "s"}` : "Export";
  if (!(exportRunning)) $("export").disabled = !n;
  if (!n && !exportRunning) msg("export-msg", s.pages ? "Approve pages in review first, or tick the box above." : "");
  else if (!exportRunning) msg("export-msg", "");
}

const claudeOnly = () => $("opt-llm").checked && $("opt-claude-only").checked;
let detectRunning = false;

function updateDetectButton(s) {
  syncOptions();
  if (claudeOnly()) {
    const n = s.pages - s.approved;
    $("detect").textContent = n ? `Ask Claude about ${n} page${n === 1 ? "" : "s"}` : "Ask Claude";
    $("detect").dataset.n = n;
    return;
  }
  const redo = $("opt-redo").checked;
  const n = redo ? (s.new || 0) + (s.pages - s.approved - s.edited) : s.new;
  $("detect").textContent = n ? `Detect ${n} page${n === 1 ? "" : "s"}` : "Detect";
  $("detect").dataset.n = n || 0;
}

// Options that do not apply grey out: Batch/Now without Claude; redo and Re-detect all when
// only Claude is asked again.
function syncOptions() {
  $("llm-mode").classList.toggle("off", !$("opt-llm").checked);
  $("opt-redo").disabled = claudeOnly();
  $("opt-redo").parentElement.classList.toggle("muted", claudeOnly());
  $("redetect-all").disabled = detectRunning || claudeOnly();
  $("guess-dates").disabled = detectRunning;
}

// ---------- detection ----------

async function startDetect(force = false) {
  msg("detect-msg", "");
  try {
    const only = claudeOnly() && !force;
    const proj = await post("/api/detect", { llm: $("opt-llm").checked, batch: llmMode() === "batch",
      redo: $("opt-redo").checked && !only, force, claude_only: only });
    renderProject(proj);
    poll();
  } catch (e) { msg("detect-msg", e.message, true); }
}

function redetectAll() {
  const s = stats || {};
  const reviewed = (s.approved || 0) + (s.edited || 0);
  const lines = [`Detect all ${s.scans ?? ""} scans again from scratch?`];
  if (reviewed) lines.push("", `${reviewed} page${reviewed === 1 ? "" : "s"} you approved or edited `
    + "will lose your changes (frames, links, captions, dates) and go back to unreviewed.");
  if ($("opt-llm").checked) lines.push("", `Claude (${$("llm-model").selectedOptions[0].text}) is asked about every page`
    + (llmMode() === "batch" ? " as a batch, at half price." : ", at full price.")
    + " A page that detects the same as last time, with the same model and effort, is answered from the cache"
    + " for free, so this costs little unless detection or the model has changed.");
  if (confirm(lines.join("\n"))) startDetect(true);
}

async function poll() {
  clearTimeout(pollTimer);
  let st;
  try { st = await api("/api/detect"); } catch (e) { msg("detect-msg", e.message, true); return; }
  if (!st) return;
  const running = st.state === "starting" || st.state === "running";
  $("progress").hidden = false;
  $("detect").disabled = running;
  detectRunning = running;
  syncOptions();
  $("cancel").hidden = !running;
  $("bar-fill").style.width = st.total ? `${(100 * st.done) / st.total}%` : "0";
  const eta = st.done && running && st.phase !== "dates" ? Math.round((st.elapsed / st.done) * (st.total - st.done)) : 0;
  if (st.phase === "dates") {
    $("progress-text").textContent = `Guessing dates: ${st.done} / ${st.total} parts of the albums`
      + (running ? "" : ` · ${st.state}`) + (st.error ? ` · ${st.error}` : "");
  } else $("progress-text").textContent = `${st.done} / ${st.total}: ${st.detected} `
    + `${st.claude_only ? "read by Claude" : "detected"}, ${st.skipped} skipped`
    + (st.failed ? `, ${st.failed} failed` : "") + (st.queued ? `, ${st.queued} queued for Claude` : "")
    + (st.current && running ? ` · ${st.current}` : "")
    + (eta ? ` · about ${fmtTime(eta)} left` : "") + (running ? "" : ` · ${st.state}`)
    + (st.error ? ` · ${st.error}` : "");
  const log = $("log");
  const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
  log.textContent = st.log.join("\n");
  if (atEnd) log.scrollTop = log.scrollHeight;
  if (running) pollTimer = setTimeout(poll, 1000);
  else renderProject(await api("/api/project"));
}

// ---------- Claude batches ----------

let batchTimer = null, collecting = false;
const BATCH_POLL_MS = 60000;

function schedulePollBatches(ms) {
  clearTimeout(batchTimer);
  batchTimer = setTimeout(pollBatches, ms);
}

async function pollBatches() {
  clearTimeout(batchTimer);
  let list;
  try { list = await api("/api/batches"); }
  catch (e) { msg("batch-msg", e.message, true); schedulePollBatches(BATCH_POLL_MS); return; }
  renderBatches(list);
  if (!list.length) return;
  if (list.some((b) => b.status === "ended")) await collectBatches();
  else schedulePollBatches(BATCH_POLL_MS);
}

function renderBatches(list) {
  $("batches-box").hidden = !list.length && !$("batch-msg").textContent;
  const box = $("batches");
  box.innerHTML = "";
  for (const b of list) {
    const div = document.createElement("div");
    div.className = "batch";
    const done = (b.succeeded || 0) + (b.errored || 0) + (b.canceled || 0) + (b.expired || 0);
    const age = b.submitted ? ` · sent ${fmtTime(Math.round(Date.now() / 1000 - b.submitted))} ago` : "";
    const state = b.status === "ended" ? "done, applying the answers…"
      : b.status === "in_progress" ? `${done} of ${b.pages} read` : b.status;
    const opt = [...$("llm-model").options].find((o) => o.value === b.model);
    div.textContent = `${b.pages} page${b.pages === 1 ? "" : "s"} · ${opt ? opt.text : b.model} · ${state}${age}`
      + (b.errored ? ` · ${b.errored} failed` : "") + (b.error ? ` · ${b.error}` : "");
    const bar = document.createElement("div");
    bar.className = "bar";
    bar.innerHTML = `<div style="width:${b.pages ? (100 * done) / b.pages : 0}%"></div>`;
    div.append(bar);
    box.append(div);
  }
}

async function collectBatches() {
  if (collecting) return;
  collecting = true;
  try {
    const r = await post("/api/batches/collect");
    const parts = [`Claude's answers applied to ${r.updated} page${r.updated === 1 ? "" : "s"}`];
    if (r.not_updated) parts.push(`${r.not_updated} left as they were (see the log)`);
    msg("batch-msg", parts.join("; ") + ".");
    if (r.log.length) {
      const log = $("log");
      $("progress").hidden = false;
      log.textContent += (log.textContent ? "\n" : "") + r.log.join("\n");
      log.scrollTop = log.scrollHeight;
    }
    renderProject(r.project);
    if (r.dates) poll();  // the date guesses started now that every answer is in
    if (r.pending.batches) schedulePollBatches(BATCH_POLL_MS);
    else renderBatches([]);
  } catch (e) {
    msg("batch-msg", e.message, true);
    schedulePollBatches(BATCH_POLL_MS);
  } finally { collecting = false; }
}

// ---------- export ----------

let exportRunning = false;

async function startExport() {
  msg("export-msg", "");
  try {
    const proj = await post("/api/export", { out_dir: $("export-dir").value, all: $("opt-all").checked,
      cards: $("opt-cards").checked, metadata: $("opt-meta").checked });
    $("export-dir").value = proj.export_dir;
    pollExport();
  } catch (e) { msg("export-msg", e.message, true); }
}

async function pollExport() {
  clearTimeout(exportTimer);
  let st;
  try { st = await api("/api/export"); } catch (e) { msg("export-msg", e.message, true); return; }
  if (!st) return;
  exportRunning = st.state === "starting" || st.state === "running";
  $("export-progress").hidden = false;
  $("export").disabled = exportRunning;
  $("export-cancel").hidden = !exportRunning;
  $("export-fill").style.width = st.total ? `${(100 * st.done) / st.total}%` : "0";
  const eta = st.done && exportRunning ? Math.round((st.elapsed / st.done) * (st.total - st.done)) : 0;
  $("export-text").textContent = `${st.done} / ${st.total} pages` + (st.photos ? `, ${st.photos} photos` : "")
    + (st.failed ? `, ${st.failed} failed` : "") + (st.current && exportRunning ? ` · ${st.current}` : "")
    + (eta ? ` · about ${fmtTime(eta)} left` : "") + (exportRunning ? "" : ` · ${st.state}`)
    + (st.error ? ` · ${st.error}` : "");
  const log = $("export-log");
  const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
  log.textContent = st.log.join("\n");
  if (atEnd) log.scrollTop = log.scrollHeight;
  if (exportRunning) exportTimer = setTimeout(pollExport, 1000);
  else {
    if (st.state === "finished") msg("export-msg", `Saved in ${st.out_dir}`);
    const proj = await api("/api/project");
    if (proj.stats) updateExportButton(proj.stats);
  }
}

function fmtTime(s) { return s < 90 ? `${s} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`; }

// ---------- folder browser ----------

let browseTarget = null, browsing = null;

async function browse(target) {
  browseTarget = target;
  $("browser-title").textContent = { "input-dir": "Choose the scan folder", "work-dir": "Choose the work folder",
    "export-dir": "Choose the export folder" }[target];
  let start = $(target).value.trim();
  if (!start && target === "work-dir" && $("input-dir").value) start = $("input-dir").value;
  $("browser").showModal();
  await show(start).catch(() => show(""));
}

async function show(path) {
  const r = await api("/api/fs?path=" + encodeURIComponent(path));
  browsing = r;
  $("browser-path").value = r.path;
  $("up").disabled = r.parent === null;
  const ul = $("dirs");
  ul.innerHTML = "";
  if (!r.dirs.length) ul.innerHTML = '<li class="empty">No sub-folders</li>';
  for (const d of r.dirs) {
    const li = document.createElement("li");
    li.textContent = d.name;
    li.addEventListener("click", () => show(d.path).catch((e) => alert(e.message)));
    ul.append(li);
  }
  const info = [];
  if (r.n_images) info.push(`${r.n_images} scan${r.n_images === 1 ? "" : "s"} here`);
  if (r.is_work_dir) info.push("albumsplit work folder");
  $("browser-info").textContent = info.join(" · ");
  $("choose").disabled = !r.path;
}

function choose(path, info) {
  $(browseTarget).value = path;
  if (browseTarget === "work-dir" && info && info.input_dir && !$("input-dir").value) $("input-dir").value = info.input_dir;
  if (browseTarget === "input-dir" && !$("work-dir").value) $("work-dir").value = trimSep(path) + "_albumsplit";
  $("browser").close();
}

$("up").addEventListener("click", () => browsing && browsing.parent !== null && show(browsing.parent));
$("go").addEventListener("click", () => show($("browser-path").value).catch((e) => alert(e.message)));
$("browser-path").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); $("go").click(); } });
$("choose").addEventListener("click", () => choose(browsing.path, browsing));
$("new-folder").addEventListener("click", () => {
  if (!browsing || !browsing.path) return;
  const name = prompt("Name of the new folder (it is created when you press Open):", "albumsplit_work");
  if (name) choose(trimSep(browsing.path) + sepOf(browsing.path) + name.trim(), null);
});
document.querySelectorAll("[data-browse]").forEach((b) => b.addEventListener("click", () => browse(b.dataset.browse)));

// ---------- wiring ----------

$("open").addEventListener("click", openProject);
for (const id of ["input-dir", "work-dir"]) $(id).addEventListener("keydown", (e) => { if (e.key === "Enter") openProject(); });
$("detect").addEventListener("click", () => startDetect(false));
$("redetect-all").addEventListener("click", redetectAll);
$("cancel").addEventListener("click", () => post("/api/detect/cancel").then(poll));
$("opt-redo").addEventListener("change", () => api("/api/project").then((p) => p.stats && updateDetectButton(p.stats)));
for (const id of ["opt-llm", "opt-claude-only"]) {
  $(id).addEventListener("change", () => { syncOptions(); if (stats) updateDetectButton(stats); });
}
$("llm-model").addEventListener("change", setModel);
$("llm-effort").addEventListener("change", setEffort);
$("opt-guess-dates").addEventListener("change", setGuessDates);
$("guess-dates").addEventListener("click", (e) => { e.preventDefault(); guessDatesNow(); });
$("batch-check").addEventListener("click", () => { msg("batch-msg", "Checking…"); pollBatches().then(() => {
  if ($("batch-msg").textContent === "Checking…") msg("batch-msg", ""); }); });
$("review").addEventListener("click", () => { location.href = "/review"; });
$("export").addEventListener("click", startExport);
$("export-cancel").addEventListener("click", () => post("/api/export/cancel").then(pollExport));
$("opt-all").addEventListener("change", () => api("/api/project").then((p) => p.stats && updateExportButton(p.stats)));

init().catch((e) => msg("open-msg", "Failed to load: " + e.message, true));

// Build shown in the corner, so it is clear which version runs (e.g. on the NAS).
fetch("/api/version").then((r) => r.json()).then((v) => {
  const el = document.getElementById("version");
  el.textContent = "albumsplit " + v.text;
  el.title = v.commit ? `commit ${v.commit}${v.dirty ? " with uncommitted changes" : ""}` : "commit unknown";
}).catch(() => {});
