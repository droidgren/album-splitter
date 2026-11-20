"use strict";
// Start page: choose the scan and work folders, run detection, go to review.

const $ = (id) => document.getElementById(id);
let pollTimer = null;
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
  $("llm-note").textContent = proj.api_key ? `(${proj.llm_model}, costs a few cents per page)`
    : "(needs ANTHROPIC_API_KEY set before starting albumsplit, or an `ant auth login`)";
  updateDetectButton(s);
  $("review").disabled = !s.pages;
  const left = s.pages - s.approved;
  msg("review-msg", s.pages ? `${left} of ${s.pages} pages left to approve` : "Detect the pages first.");
}

function updateDetectButton(s) {
  const redo = $("opt-redo").checked;
  const n = redo ? (s.new || 0) + (s.pages - s.approved - s.edited) : s.new;
  $("detect").textContent = n ? `Detect ${n} page${n === 1 ? "" : "s"}` : "Detect";
  $("detect").dataset.n = n || 0;
}

// ---------- detection ----------

async function startDetect(force = false) {
  msg("detect-msg", "");
  try {
    const proj = await post("/api/detect", { llm: $("opt-llm").checked, redo: $("opt-redo").checked, force });
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
  if ($("opt-llm").checked) lines.push("", "Claude reads every page again: this costs a few cents per page.");
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
  $("redetect-all").disabled = running;
  $("cancel").hidden = !running;
  $("bar-fill").style.width = st.total ? `${(100 * st.done) / st.total}%` : "0";
  const eta = st.done && running ? Math.round((st.elapsed / st.done) * (st.total - st.done)) : 0;
  $("progress-text").textContent = `${st.done} / ${st.total}: ${st.detected} detected, ${st.skipped} skipped`
    + (st.failed ? `, ${st.failed} failed` : "") + (st.current && running ? ` · ${st.current}` : "")
    + (eta ? ` · about ${fmtTime(eta)} left` : "") + (running ? "" : ` · ${st.state}`)
    + (st.error ? ` · ${st.error}` : "");
  const log = $("log");
  const atEnd = log.scrollTop + log.clientHeight >= log.scrollHeight - 4;
  log.textContent = st.log.join("\n");
  if (atEnd) log.scrollTop = log.scrollHeight;
  if (running) pollTimer = setTimeout(poll, 1000);
  else renderProject(await api("/api/project"));
}

function fmtTime(s) { return s < 90 ? `${s} s` : s < 5400 ? `${Math.round(s / 60)} min` : `${(s / 3600).toFixed(1)} h`; }

// ---------- folder browser ----------

let browseTarget = null, browsing = null;

async function browse(target) {
  browseTarget = target;
  $("browser-title").textContent = target === "input-dir" ? "Choose the scan folder" : "Choose the work folder";
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
$("review").addEventListener("click", () => { location.href = "/review"; });

init().catch((e) => msg("open-msg", "Failed to load: " + e.message, true));
