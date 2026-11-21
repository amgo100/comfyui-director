// ComfyUI Director - timeline panel for long Qwen-Image + Wan videos.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const CSS_URL = new URL("./director.css", import.meta.url).href;

// ------------------------------------------------------------------ helpers
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k.startsWith("on")) el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === "value") el.value = v;
    else if (k === "checked") el.checked = !!v;
    else if (k === "html") el.innerHTML = v;
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const k of kids.flat(Infinity)) {
    if (k === null || k === undefined || k === false) continue;
    el.append(k instanceof Node ? k : document.createTextNode(String(k)));
  }
  return el;
}
const fmtSec = (frames, fps) => (frames / fps).toFixed(1) + "s";

async function J(path, opts = {}) {
  const r = await api.fetchApi(path, {
    method: opts.body !== undefined || opts.form ? "POST" : "GET",
    headers: opts.form ? undefined : { "Content-Type": "application/json" },
    body: opts.form ? opts.form : opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
  });
  const d = await r.json().catch(() => ({}));
  if (!r.ok || d.error) throw new Error(d.error || r.statusText);
  return d;
}

// ------------------------------------------------------------------ state
const S = {
  open: false, root: null, pid: localStorage.getItem("director.pid") || null,
  project: null, status: null, config: null, projects: [],
  sel: null,             // {type:'keyframe'|'segment', id}
  viewTake: {},          // entity id -> take id shown in viewer
  pending: new Map(),    // prompt_id -> {target, label}
  progress: null, zoom: 3, autofill: false, lastRender: null,
  finish: null,          // finishing status: {settings, items, count}
  loaded: "",            // diffusion weights of the last finished job (model key), to keep auto-fill on them
  tried: new Set(),      // ids queued by the current auto-fill run (each is tried once per run)
  autoFailed: 0,         // jobs of the current auto-fill run that failed
  showFinished: true,    // viewer shows a take's finished (upscaled) version when it has one
};
const applyData = (d) => { S.project = d.project; S.status = d.status; if (d.finish) S.finish = d.finish; };

const fileURL = (rel) => api.apiURL(`/director/file/${S.pid}/${rel}`);
const kfById = (id) => S.project?.keyframes?.[id];
const segById = (id) => S.project?.timeline?.find((s) => s.id === id);
const takeOf = (ent, id) => ent?.takes?.find((t) => t.id === (id || ent.selected));
const kfThumb = (kf) => { const t = takeOf(kf); return t ? fileURL(t.file) : null; };
const pendingFor = (target) => [...S.pending.values()].filter((p) => p.target === target).length;

function toast(msg, err = false) {
  const t = h("div", { class: "dr-toast" + (err ? " err" : "") }, msg);
  document.body.append(t);
  setTimeout(() => t.remove(), err ? 6000 : 2500);
}

// ------------------------------------------------------------------ server calls
async function loadProject(pid) {
  if (!pid) { S.project = null; S.status = null; return; }
  try {
    const d = await J(`/director/project/${pid}`);
    S.pid = pid; applyData(d);
    localStorage.setItem("director.pid", pid);
  } catch (e) { S.pid = null; S.project = null; localStorage.removeItem("director.pid"); }
}
async function op(o) {
  const d = await J(`/director/project/${S.pid}/op`, { body: o });
  applyData(d);
  render();
  return d.result || {};
}
async function queueAction(req, label) {
  const d = await J(`/director/project/${S.pid}/build`, { body: req });
  for (const job of d.jobs) {
    const r = await api.fetchApi("/prompt", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt: job.prompt, client_id: api.clientId, extra_data: { director: { target: job.target } } }) });
    const res = await r.json();
    if (!r.ok || res.error) {
      const ne = res.node_errors ? Object.values(res.node_errors).map((n) => n.errors?.map((e) => e.details || e.message).join("; ")).join(" | ") : "";
      throw new Error((res.error?.message || res.error || "queue failed") + (ne ? `: ${ne}` : ""));
    }
    S.pending.set(res.prompt_id, { target: job.target, label, ...planProgress(job.prompt) });
  }
  render();
  return d.jobs.length;
}
async function generate(req, label) {
  try { const n = await queueAction(req, label); toast(`Queued ${n} × ${label}`); }
  catch (e) { toast(e.message, true); }
}

// same rule as graphs.model_key(): the diffusion weights a graph loads
const MODEL_LOADERS = { UNETLoader: "unet_name", UnetLoaderGGUF: "unet_name", CheckpointLoaderSimple: "ckpt_name" };
const modelKey = (graph) => [...new Set(Object.values(graph).filter((n) => MODEL_LOADERS[n.class_type])
  .map((n) => n.inputs[MODEL_LOADERS[n.class_type]]))].sort().join("|");

function onJobDone(pid, ok, msg) {
  if (!S.pending.has(pid)) return;
  const p = S.pending.get(pid);
  S.pending.delete(pid);
  const key = ok && p.graph ? modelKey(p.graph) : "";
  if (key) S.loaded = key;
  if (!ok && p.auto) S.autoFailed++;
  if (!ok) toast(`${p.label} failed: ${msg || "error"}`, true);
  if (S.running === pid) S.running = null;
  if (S.open) loadProject(S.pid).then(() => { render(); if (S.autofill && ![...S.pending.values()].some((x) => x.auto)) autoFillStep(); });
}
api.addEventListener("execution_success", (e) => onJobDone(e.detail?.prompt_id, true));
api.addEventListener("execution_error", (e) => onJobDone(e.detail?.prompt_id, false, e.detail?.exception_message));
api.addEventListener("execution_interrupted", (e) => onJobDone(e.detail?.prompt_id, false, "interrupted"));
// ---- job progress: every node of the queued graph gets a weight (rough share of run time);
// ComfyUI's per-node events tell us which nodes are done, running (with step progress) or cached.
const LOADERS = new Set(["UNETLoader", "CheckpointLoaderSimple", "UnetLoaderGGUF"]);
const ENCODERS = new Set(["VAEEncode", "WanImageToVideo", "WanFirstLastFrameToVideo", "WanVaceToVideo", "WanFunInpaintToVideo",
  "WanCameraImageToVideo", "Wan22FunControlToVideo"]);
function nodeWeight(n) {
  const c = n.class_type, i = n.inputs || {};
  if (c === "KSampler") return 3 + 3 * (+i.steps || 20);
  if (c === "KSamplerAdvanced") return 3 + 3 * Math.max(1, Math.min(+i.end_at_step || 0, +i.steps || 20) - (+i.start_at_step || 0));
  if (LOADERS.has(c)) return 6;
  if (c === "CLIPLoader" || c === "CLIPLoaderGGUF") return 4;
  if (ENCODERS.has(c)) return 3;
  if (c === "CLIPTextEncode" || c.startsWith("TextEncodeQwen")) return 2;
  if (c === "VAEDecode") return 5;
  if (c === "VAEDecodeTiled") return 8;
  if (c === "VAEEncodeTiled") return 4;
  if (c === "DirectorUpscaleFrames") return i.upscale_model ? 15 : 2;
  if (c === "UpscaleModelLoader" || c === "FrameInterpolationModelLoader") return 1;
  if (c === "DirectorSaveVideo") return 3;
  if (c === "FrameInterpolate") return 25;
  if (c === "VAELoader" || c === "LoraLoaderModelOnly" || c === "DirectorLoadVideoFrames" || c === "DirectorSaveImage") return 1;
  return 0.3;
}
function stageName(n) {
  if (!n) return "";
  const c = n.class_type, t = n._meta?.title && n._meta.title !== c ? n._meta.title : "";
  if (c.startsWith("KSampler")) return `Sampling${t ? " – " + t.toLowerCase() : ""}`;
  if (LOADERS.has(c)) return `Loading ${t || "model"}`;
  if (c.startsWith("CLIPLoader")) return "Loading text encoder";
  if (c === "VAELoader") return "Loading VAE";
  if (c === "LoraLoaderModelOnly") return "Applying LoRA";
  if (c === "CLIPTextEncode" || c.startsWith("TextEncodeQwen")) return "Encoding prompt";
  if (ENCODERS.has(c) || c === "VAEEncodeTiled") return "Encoding frames";
  if (c === "VAEDecode" || c === "VAEDecodeTiled") return "Decoding";
  if (c === "DirectorUpscaleFrames") return t ? t : "Upscaling";
  if (c === "UpscaleModelLoader") return "Loading upscale model";
  if (c === "FrameInterpolationModelLoader") return "Loading interpolation model";
  if (c === "DirectorJoinFrames") return "Joining refine windows";
  if (c === "DirectorPadFrames" || c === "DirectorFrameRange") return "Preparing frames";
  if (c === "DirectorColorMatch") return "Colour matching";
  if (c === "DirectorVaceBridge" || c === "DirectorVaceInpaint" || c === "DirectorMaskCrop") return "Preparing frames";
  if (c === "DirectorMaskStitch" || c === "DirectorCompositeFrames") return "Blending result";
  if (c.startsWith("DirectorSave")) return n.inputs?.target?.startsWith("finish:") ? "Saving finished version" : "Saving take";
  if (c === "FrameInterpolate") return "Interpolating";
  if (c.startsWith("DirectorLoad")) return "Loading inputs";
  return t || c;
}
function planProgress(graph) {
  const weights = {};
  let total = 0;
  for (const [id, n] of Object.entries(graph)) { weights[id] = nodeWeight(n); total += weights[id]; }
  return { graph, weights, total, nodes: {}, current: null, queuedAt: Date.now() };
}
function jobFraction(p) {
  let done = 0;
  for (const [id, w] of Object.entries(p.weights)) {
    const st = p.nodes[id];
    if (!st) continue;
    if (st.state === "finished" || st.state === "cached") done += w;
    else if (st.state === "running") done += w * (st.max > 0 ? Math.min(1, st.value / st.max) : 0.15);
  }
  return p.total ? Math.min(0.99, done / p.total) : 0;
}
function jobStage(p) {
  const id = p.current;
  if (!id) return "starting";
  const st = p.nodes[id];
  const base = stageName(p.graph[id]);
  return st && st.max > 1 && p.graph[id]?.class_type?.startsWith("KSampler") ? `${base} ${st.value}/${st.max}` : base;
}
const fmtTime = (ms) => { const s = Math.round(ms / 1000); return s < 60 ? `${s}s` : `${Math.floor(s / 60)}m ${String(s % 60).padStart(2, "0")}s`; };
function updateProgressUI() {
  if (!S.open || !S.root) return;
  const p = S.running && S.pending.get(S.running);
  const queued = S.pending.size - (p ? 1 : 0);
  const pct = p ? Math.round(100 * jobFraction(p)) : 0;
  const lb = S.root.querySelector(".dr-q .lb");
  if (lb) {
    const el = p?.startedAt ? fmtTime(Date.now() - p.startedAt) : "";
    const eta = p?.startedAt && pct >= 8 ? ` · ~${fmtTime(((Date.now() - p.startedAt) * (100 - pct)) / pct)} left` : "";
    lb.textContent = p ? `${p.label} · ${jobStage(p)} · ${pct}%${el ? " · " + el : ""}${eta}${queued ? ` · +${queued} queued` : ""}` : `${S.pending.size} job${S.pending.size > 1 ? "s" : ""} queued`;
  }
  const bar = S.root.querySelector(".dr-q .bar i");
  if (bar) bar.style.width = `${pct}%`;
  for (const el of S.root.querySelectorAll(".dr-take.pending[data-pid]")) {
    const q = S.pending.get(el.dataset.pid);
    const t = el.querySelector(".t");
    if (!q || !t) continue;
    t.textContent = el.dataset.pid === S.running ? `${Math.round(100 * jobFraction(q))}% · ${jobStage(q)}` : "queued";
    const b = el.querySelector(".pbar i");
    if (b) b.style.width = el.dataset.pid === S.running ? `${Math.round(100 * jobFraction(q))}%` : "0";
  }
}
const ours = (e) => S.pending.get(e.detail?.prompt_id);
api.addEventListener("execution_start", (e) => {
  const p = ours(e); if (!p) return;
  S.running = e.detail.prompt_id; p.startedAt = Date.now(); render();
});
api.addEventListener("execution_cached", (e) => {
  const p = ours(e); if (!p) return;
  for (const id of e.detail.nodes || []) p.nodes[id] = { state: "cached", value: 1, max: 1 };
  updateProgressUI();
});
api.addEventListener("executing", (e) => {
  const p = ours(e); if (!p) return;
  const id = e.detail.node;
  if (p.current && p.current !== id && p.nodes[p.current]?.state === "running") p.nodes[p.current].state = "finished";
  if (id) { p.current = String(id); p.nodes[p.current] = { ...(p.nodes[p.current] || {}), state: "running", value: 0, max: 0 }; }
  if (S.running !== e.detail.prompt_id) { S.running = e.detail.prompt_id; p.startedAt = p.startedAt || Date.now(); render(); }
  updateProgressUI();
});
api.addEventListener("progress", (e) => {
  const p = ours(e); if (!p) return;
  const id = String(e.detail.node ?? p.current);
  p.nodes[id] = { ...(p.nodes[id] || {}), state: "running", value: e.detail.value, max: e.detail.max };
  if (id) p.current = id;
  updateProgressUI();
});
api.addEventListener("progress_state", (e) => {   // newer ComfyUI: authoritative per-node states
  const p = ours(e); if (!p) return;
  for (const [id, st] of Object.entries(e.detail.nodes || {})) {
    const key = String(st.real_node_id || st.display_node_id || id).split(":").pop();
    p.nodes[key] = { state: st.state, value: st.value, max: st.max };
    if (st.state === "running") p.current = key;
  }
  updateProgressUI();
});
setInterval(() => { if (S.running) updateProgressUI(); }, 1000);  // keeps elapsed time / ETA ticking

// Auto-fill: generate everything that is ready, wait, repeat (follows dependencies down the timeline).
// Each step queues one model group (the loaded weights first), so models are not swapped back and forth.
// Mode "gaps" fills items without takes; mode "stale" also regenerates stale takes (new take replaces
// the selection, which makes whatever was built from it stale in turn: a cascade down the timeline).
function startAutofill(mode) {
  S.autofill = mode; S.tried = new Set(); S.autoFailed = 0;
  render(); autoFillStep();
}
function stopAutofill(msg) { S.autofill = false; if (msg) toast(msg); render(); }
async function autoFillStep() {
  if (!S.autofill) return;
  const q = new URLSearchParams({ stale: S.autofill === "stale" ? "1" : "0", loaded: S.loaded, skip: [...S.tried].join(",") });
  let step;
  try { step = await J(`/director/project/${S.pid}/plan?${q}`); }
  catch (e) { stopAutofill(); toast(e.message, true); return; }
  const todo = step.jobs.filter((j) => !pendingFor((j.action.startsWith("keyframe") ? "keyframe:" : "segment:") + j.id));
  if (!todo.length) {
    if (!S.pending.size) {
      const what = S.autofill === "stale" ? "Update" : "Auto-fill";
      stopAutofill(S.autoFailed ? `${what} finished; ${S.autoFailed} job(s) failed and were not retried`
        : S.autofill === "stale" ? "Everything is up to date" : "Auto-fill: nothing left that can be generated");
    }
    return;
  }
  const others = step.groups.filter((g) => g.key !== todo[0].models).reduce((a, g) => a + g.count, 0);
  toast(`${S.autofill === "stale" ? "Updating" : "Auto-fill"}: ${todo.length} × ${step.group}` + (others ? ` · ${others} more wait for other models` : ""));
  let queued = 0;
  for (const j of todo) {
    const ent = j.action.startsWith("keyframe") ? kfById(j.id) : segById(j.id);
    S.tried.add(j.id);
    try { await queueAction({ ...j, count: 1 }, `${j.stale ? "update" : "auto"}: ${ent?.name || j.id}`); queued++; }
    catch (e) { S.autoFailed++; toast(e.message, true); }
  }
  for (const v of S.pending.values()) v.auto = true;
  if (!queued && !S.pending.size) autoFillStep();  // nothing of this group could be queued: try the next group
}

// ------------------------------------------------------------------ panel shell
function ensureRoot() {
  if (S.root) return;
  if (!document.querySelector(`link[href="${CSS_URL}"]`)) document.head.append(h("link", { rel: "stylesheet", href: CSS_URL }));
  S.root = h("div", { class: "dr-root" });
  document.body.append(S.root);
  // keep ComfyUI's canvas shortcuts away from the panel
  S.root.addEventListener("keydown", (e) => e.stopPropagation());
  window.addEventListener("keydown", (e) => {
    if (S.open && e.key === "Escape" && !document.querySelector(".dr-modal") && !e.target.matches?.("input,textarea,select")) closePanel();
  });
}
async function openPanel() {
  ensureRoot();
  S.open = true; S.root.style.display = "";
  S.config = S.config || (await J("/director/config").catch(() => ({ _loras: [] })));
  S.projects = (await J("/director/projects")).projects;
  if (!S.pid && S.projects.length) S.pid = S.projects[0].id;
  await loadProject(S.pid);
  render();
}
function closePanel() { S.open = false; if (S.root) S.root.style.display = "none"; }

function render() {
  if (!S.open || !S.root) return;
  const insp = S.root.querySelector(".dr-insp");
  const keepInsp = insp && insp.contains(document.activeElement) && document.activeElement.matches("input,textarea,select");
  const scroll = S.root.querySelector(".dr-track")?.scrollLeft || 0;
  const newKids = [renderHeader(), renderLibrary(), renderViewer(), keepInsp ? insp : renderInspector(), renderTimeline()];
  S.root.replaceChildren(...newKids);
  const tr = S.root.querySelector(".dr-track"); if (tr) tr.scrollLeft = scroll;
  updateProgressUI();
  S.finishRedraw?.();
  S.staleRedraw?.();
}

// ------------------------------------------------------------------ header
function renderHeader() {
  const sel = h("select", { onchange: async (e) => { await loadProject(e.target.value); S.sel = null; render(); } },
    S.projects.map((p) => h("option", { value: p.id, selected: p.id === S.pid }, p.name)));
  const n = S.pending.size;
  return h("div", { class: "dr-hd" },
    h("span", { class: "title" }, "🎬 Director"),
    S.projects.length ? sel : null,
    h("button", { onclick: newProject }, "New project"),
    S.project && h("button", { onclick: () => settingsDialog() }, "Settings"),
    h("span", { class: "sp" }),
    n ? h("span", { class: "dr-q" }, h("span", { class: "dr-spin" }), h("span", { class: "lb" }, ""), h("span", { class: "bar" }, h("i"))) : null,
    S.project && staleCount().total && S.autofill !== "stale" ? h("button", { class: "stale", title: "Takes made from keyframes or clips whose selected take has since changed", onclick: staleDialog },
      `↻ ${staleCount().direct} stale` + (staleCount().downstream ? ` +${staleCount().downstream}` : "")) : null,
    S.project && h("button", { class: S.autofill ? "warn" : "", title: "Generate every keyframe and clip whose inputs are ready, then continue down the timeline as results arrive. Works through one model at a time to avoid reloading weights.",
      onclick: () => (S.autofill ? stopAutofill() : startAutofill("gaps")) }, S.autofill === "stale" ? "■ Stop updating" : S.autofill ? "■ Stop auto-fill" : "⚡ Auto-fill gaps"),
    S.project && h("button", { title: "Render the selected takes at the project size, as generated", onclick: () => doRender(false) }, "▶ Render draft"),
    S.project && h("button", { class: "pri", title: "Upscale the selected takes to the delivery size and render the final video", onclick: () => finishDialog() },
      `✨ Final ${finishLabel()}…`),
    h("button", { onclick: closePanel, title: "Close (Esc)" }, "✕"));
}
async function newProject() {
  const name = prompt("Project name", "My film");
  if (!name) return;
  const d = await J("/director/projects", { body: { name } });
  S.projects = (await J("/director/projects")).projects;
  S.pid = d.project.id; applyData(d); S.sel = null;
  localStorage.setItem("director.pid", S.pid);
  render();
}
async function doRender(final = false) {
  toast(final ? "Rendering final video…" : "Rendering…");
  try {
    const r = await J(`/director/project/${S.pid}/render`, { body: { final } });
    S.lastRender = r; S.sel = { type: "render" };
    render();
    const notes = [];
    if (r.skipped?.length) notes.push(`${r.skipped.length} clip(s) skipped (no take yet)`);
    if (r.fallbacks?.length) notes.push(`${r.fallbacks.length} clip(s) not finished – resized instead`);
    if (notes.length) toast(`Rendered ${r.seconds.toFixed(1)}s; ${notes.join("; ")}`);
  } catch (e) { toast(e.message, true); }
}

// ------------------------------------------------------------------ keyframe library
function renderLibrary() {
  const box = h("div", { class: "dr-lib" });
  if (!S.project) return box;
  const kfs = Object.values(S.project.keyframes).sort((a, b) => a.created - b.created);
  const fileIn = h("input", { type: "file", accept: "image/*", multiple: true, style: { display: "none" },
    onchange: async (e) => { for (const f of e.target.files) await upload(f, "keyframe"); } });
  box.append(
    h("h3", {}, `Keyframes (${kfs.length})`),
    h("div", { class: "btns" },
      h("button", { onclick: () => newKeyframe("t2i") }, "+ Text→Image"),
      h("button", { onclick: () => fileIn.click() }, "Import"), fileIn),
    h("div", { class: "dr-kfs" }, kfs.map((kf) => {
      const th = kfThumb(kf);
      const pend = pendingFor(`keyframe:${kf.id}`);
      return h("div", {
        class: "dr-kf" + (S.sel?.type === "keyframe" && S.sel.id === kf.id ? " sel" : ""), draggable: "true",
        title: kf.recipe?.prompt || kf.name,
        ondragstart: (e) => { e.dataTransfer.setData("text/director-kf", kf.id); e.dataTransfer.effectAllowed = "copy"; },
        onclick: () => { S.sel = { type: "keyframe", id: kf.id }; render(); },
      },
        th ? h("img", { src: th, loading: "lazy" }) : h("div", { class: "ph" }, pend ? h("span", { class: "dr-spin" }) : "not generated"),
        kf.takes?.length > 1 ? h("span", { class: "cnt" }, `${kf.takes.length} takes`) : null,
        S.status.keyframes[kf.id]?.stale?.length ? h("span", { class: "dr-stalemark", title: "Out of date: " + S.status.keyframes[kf.id].stale.join("; ") }, "↻") : null,
        h("div", { class: "nm" }, h("span", { class: "dr-badge " + kf.recipe.op }, kf.recipe.op), pend ? h("span", { class: "dr-spin" }) : null, " ", kf.name || kf.id));
    })),
    h("div", { class: "dr-help", style: { marginTop: "12px" } }, "Drag a keyframe onto the left or right half of a clip in the timeline to make it that clip's start or end frame."));
  return box;
}
async function upload(file, purpose, extra = {}) {
  const fd = new FormData();
  fd.append("file", file); fd.append("purpose", purpose);
  for (const [k, v] of Object.entries(extra)) fd.append(k, v);
  try {
    toast(`Importing ${file.name}…`);
    const d = await J(`/director/project/${S.pid}/upload`, { form: fd });
    applyData(d);
    if (d.result.keyframe) S.sel = { type: "keyframe", id: d.result.keyframe };
    if (d.result.segment) S.sel = { type: "segment", id: d.result.segment };
    render();
    return d.result;
  } catch (e) { toast(e.message, true); }
}
async function newKeyframe(opName, parents = [], prompt = "") {
  const r = await op({ op: "keyframe.create", fields: { name: opName === "t2i" ? "New keyframe" : "Derived keyframe", recipe: { op: opName, prompt, parents } } });
  S.sel = { type: "keyframe", id: r.id };
  render();
  setTimeout(() => S.root.querySelector(".dr-insp textarea")?.focus(), 50);
}

// ------------------------------------------------------------------ viewer (center)
function renderViewer() {
  const box = h("div", { class: "dr-view" });
  if (!S.project) {
    box.append(h("div", { class: "dr-stage" }, h("div", { class: "empty" },
      h("h2", {}, "No project yet"), "Create a project to start building a timeline.", h("br"), h("br"),
      h("button", { class: "pri", onclick: newProject }, "New project"))));
    return box;
  }
  const sel = S.sel;
  if (sel?.type === "render" && S.lastRender) {
    const r = S.lastRender;
    box.append(h("div", { class: "dr-stage" }, h("video", { src: fileURL(r.file), controls: true, autoplay: true, loop: true }),
      h("div", { class: "cap" }, `${r.final ? "Final" : "Draft"} render · ${r.width}×${r.height} @ ${r.fps} fps · ${r.seconds.toFixed(1)}s · ${r.frames} frames`)),
      h("div", { class: "dr-takebar" }, h("span", { class: "info" }, [
          r.skipped?.length ? `${r.skipped.length} clip(s) without a take were skipped` : "All clips included",
          r.fallbacks?.length ? ` · ${r.fallbacks.length} clip(s) were not finished and were resized instead` : ""].join("")),
        h("a", { href: fileURL(r.file), download: "" }, h("button", {}, "Download mp4")),
        r.final ? h("button", { onclick: () => finishDialog() }, "Finishing…") :
        h("button", { title: "FILM frame interpolation ×2 of this draft render. For the final video, set interpolation in ✨ Final instead (it runs per clip).",
          onclick: () => generate({ action: "render.interpolate", file: r.file, multiplier: 2 }, "interpolate ×2") }, "Interpolate ×2")));
    return box;
  }
  const ent = sel?.type === "keyframe" ? kfById(sel.id) : sel?.type === "segment" ? segById(sel.id) : null;
  if (!ent) {
    box.append(h("div", { class: "dr-stage" }, h("div", { class: "empty" },
      h("h3", {}, S.project.name),
      "1. Make keyframes on the left (text→image, import, or derive from another keyframe).", h("br"),
      "2. Add clips to the timeline and drop keyframes on their start / end.", h("br"),
      "3. Describe what happens in each clip and generate takes – or press ⚡ Auto-fill.", h("br"),
      "4. Pick the best take of everything, then ▶ Render.")));
    return box;
  }
  const isSeg = sel.type === "segment";
  const still = isSeg && ent.kind === "still";
  const shownId = S.viewTake[ent.id] || ent.selected;
  const take = still ? null : takeOf(ent, shownId) || takeOf(ent);
  const stage = h("div", { class: "dr-stage" });
  if (still) {
    const kf = kfById(ent.keyframe);
    stage.append(kf && kfThumb(kf) ? h("img", { src: kfThumb(kf) }) : h("div", { class: "empty" }, "Choose a keyframe for this still"));
  } else if (take) {
    const fin = S.showFinished && take.finished ? take.finished : null;
    const fstate = finishState(sel.type, take);
    if (isSeg) {
      const v = h("video", { src: fileURL(fin ? fin.file : take.file), controls: true, loop: true, autoplay: true, muted: true, id: "dr-video" });
      stage.append(v);
    } else stage.append(h("img", { src: fileURL(fin ? fin.file : take.file), id: "dr-image" }));
    stage.append(h("div", { class: "cap" }, `${ent.name || ent.id} · take ${ent.takes.indexOf(take) + 1}/${ent.takes.length}` + (take.id === ent.selected ? " ★" : "") + (take.mode ? ` · ${take.mode}` : "") + (take.op ? ` · ${take.op}` : "")
      + (fin ? ` · finished ${fin.width}×${fin.height}${fstate === "outdated" ? " (outdated)" : ""}` : take.finished ? " · draft" : "")));
  } else {
    const st = isSeg ? S.status.segments[ent.id] : S.status.keyframes[ent.id];
    stage.append(h("div", { class: "empty" }, pendingFor(`${sel.type}:${ent.id}`) ? [h("span", { class: "dr-spin" }), " generating…"] :
      ["No takes yet.", h("br"), st?.ready ? "Ready – press Generate in the inspector." : `Not ready: ${st?.why || ""}`]));
  }
  box.append(stage);
  if (!still) {
    const pend = pendingFor(`${sel.type}:${ent.id}`);
    box.append(h("div", { class: "dr-takes" },
      ent.takes.map((t, i) => h("div", { class: "dr-take" + (t.id === (take?.id) ? " view" : "") + (t.id === ent.selected ? " chosen" : ""),
        onclick: () => { S.viewTake[ent.id] = t.id; render(); }, ondblclick: () => op({ op: `${sel.type}.select`, id: ent.id, take: t.id }) },
        h("img", { src: fileURL(isSeg ? t.first : t.file), loading: "lazy" }),
        h("div", { class: "meta" }, h("span", {}, `#${i + 1} ${t.op || ""}`), h("span", {}, isSeg ? fmtSec(t.frames || 0, S.project.settings.fps) : (t.seed ? `seed ${String(t.seed).slice(-5)}` : ""))))),
      [...S.pending.entries()].filter(([, q]) => q.target === `${sel.type}:${ent.id}`).map(([pid]) =>
        h("div", { class: "dr-take pending", "data-pid": pid },
          h("div", { class: "row" }, h("span", { class: "dr-spin" }), h("span", { class: "t" }, pid === S.running ? "starting" : "queued")),
          h("div", { class: "pbar" }, h("i"))))));
    if (take) {
      box.append(h("div", { class: "dr-takebar" },
        h("span", { class: "info" }, take.prompt ? `“${take.prompt.slice(0, 110)}${take.prompt.length > 110 ? "…" : ""}”` : ""),
        take.id !== ent.selected ? h("button", { class: "pri", onclick: () => op({ op: `${sel.type}.select`, id: ent.id, take: take.id }) }, "★ Use this take") : null,
        isSeg ? h("button", { title: "Save the current video frame as a new keyframe", onclick: () => grabFrame(ent, take) }, "📷 Frame → keyframe") : null,
        take.finished ? h("button", { title: "Switch between the generated take and its finished (upscaled) version",
          onclick: () => { S.showFinished = !S.showFinished; render(); } }, S.showFinished ? "Show draft" : "Show finished") : null,
        isSeg || stillKeyframes().has(ent.id) ? finishButton(sel.type, ent, take) : null,
        h("button", { onclick: () => (isSeg ? videoInpaintDialog(ent, take) : imageInpaintDialog(ent, take)) }, "🖌 Inpaint region…"),
        !isSeg ? h("button", { onclick: () => newKeyframe("edit", [ent.id]) }, "↳ Derive new keyframe") : null,
        h("button", { class: "warn", onclick: () => confirm("Delete this take?") && op({ op: `${sel.type}.delete_take`, id: ent.id, take: take.id }) }, "Delete take")));
    }
  }
  return box;
}
async function grabFrame(seg, take) {
  const v = S.root.querySelector("#dr-video");
  const frame = v ? Math.round(v.currentTime * S.project.settings.fps) : 0;
  try {
    const d = await J(`/director/project/${S.pid}/grab`, { body: { segment: seg.id, take: take.id, frame } });
    applyData(d);
    S.sel = { type: "keyframe", id: d.result.keyframe };
    render(); toast(`Frame ${frame} saved as keyframe`);
  } catch (e) { toast(e.message, true); }
}

// ------------------------------------------------------------------ inspector (right)
function loraEditor(list, onChange, withStage) {
  const loras = S.config?._loras || [];
  const wrap = h("div", {});
  const draw = () => wrap.replaceChildren(...list.map((l, i) => h("div", { class: "dr-lora" },
    h("select", { onchange: (e) => { list[i].name = e.target.value; onChange(list); } },
      h("option", { value: "" }, "— choose LoRA —"), (loras.includes(l.name) || !l.name ? loras : [l.name, ...loras]).map((n) => h("option", { value: n, selected: n === l.name }, n))),
    h("input", { type: "number", step: "0.05", value: l.strength ?? 1, title: "strength", onchange: (e) => { list[i].strength = parseFloat(e.target.value); onChange(list); } }),
    withStage ? h("select", { title: "Wan 2.2 stage: high-noise, low-noise or both", onchange: (e) => { list[i].stage = e.target.value; onChange(list); } },
      ["both", "high", "low"].map((s) => h("option", { value: s, selected: (l.stage || "both") === s }, s))) : h("span"),
    h("button", { title: "remove", onclick: () => { list.splice(i, 1); draw(); onChange(list); } }, "✕"))),
    h("button", { style: { fontSize: "12px" }, onclick: () => { list.push({ name: "", strength: 1, stage: "both" }); draw(); } }, "+ LoRA"));
  draw();
  return wrap;
}
function countPicker() {
  return h("input", { type: "number", min: 1, max: 8, value: S.project.settings.takes_per_run || 2, class: "fit", style: { width: "54px" }, id: "dr-count", title: "takes per run" });
}
const getCount = () => parseInt(S.root.querySelector("#dr-count")?.value || "1", 10);

function renderInspector() {
  const box = h("div", { class: "dr-insp" });
  if (!S.project) return box;
  const sel = S.sel;
  if (sel?.type === "keyframe" && kfById(sel.id)) box.append(...inspectKeyframe(kfById(sel.id)).filter(Boolean));
  else if (sel?.type === "segment" && segById(sel.id)) box.append(...inspectSegment(segById(sel.id)).filter(Boolean));
  else box.append(h("h3", {}, "Inspector"), h("div", { class: "dr-help" }, "Select a keyframe or a clip to edit it."),
    h("label", {}, "Project"), h("div", {}, `${S.project.timeline.length} clips · ${Object.keys(S.project.keyframes).length} keyframes`),
    h("div", { class: "dr-help" }, `${S.project.settings.width}×${S.project.settings.height} @ ${S.project.settings.fps} fps · max ${S.project.settings.max_segment_frames} frames per generated clip`));
  return box;
}

const qi21 = () => (S.project?.settings?.image_model || S.config?.image_model || "qwen21") === "qwen21";
const maxRefs = () => (qi21() ? 16 : 3);

// Source/reference keyframes. Qwen Image 2.1 sees them as <image1>, <image2>, ... in the prompt.
function referencePicker(kf, rec, others, upd) {
  const edit = rec.op === "edit";
  if (!edit && !qi21()) return [];
  const parents = [...(rec.parents || [])];
  const nProj = (S.project.settings.image_reference_keyframes || []).filter((k) => k !== kf.id && !parents.includes(k)).length;
  const limit = maxRefs() - nProj;
  const rows = [];
  const n = Math.min(limit, parents.length + 1);
  for (let i = 0; i < Math.max(edit ? 1 : 0, n); i++) {
    const tag = qi21() ? `image${i + 1}` : String(i + 1);
    const th = parents[i] && kfThumb(kfById(parents[i]));
    rows.push(h("div", { class: "dr-ref" },
      h("span", { class: "tag" }, tag),
      th ? h("img", { src: th }) : h("span", { class: "noimg" }),
      h("select", { onchange: (e) => { parents[i] = e.target.value; upd({ recipe: { parents: parents.filter(Boolean) } }); } },
        h("option", { value: "" }, edit && i === 0 ? "— image to edit —" : "— add reference —"),
        others.map((o) => h("option", { value: o.id, selected: parents[i] === o.id }, o.name || o.id)))));
  }
  return [
    h("label", {}, edit ? "Image to edit and references" : "References (optional)"),
    h("div", { class: "dr-parents" }, rows),
    h("div", { class: "dr-help" }, qi21()
      ? (edit ? "image1 is edited; the others are references. Refer to them in the prompt, e.g. “put the coat from <image2> on the man in <image1>”."
              : "Optional: keyframes the new image should borrow from (character, location, style). Refer to them as <image1>, <image2>…")
      : "1 = base image, 2–3 = references"),
  ];
}
function projectRefsNote(kf) {
  if (!qi21()) return null;
  const parents = (kf.recipe.parents || []).filter(Boolean);
  const ids = (S.project.settings.image_reference_keyframes || []).filter((k) => k !== kf.id && !parents.includes(k) && kfById(k)?.selected);
  if (!ids.length) return null;
  const start = parents.length + 1;
  return h("div", { class: "dr-help" }, "+ project references: ",
    ids.map((k, i) => `${kfById(k).name || k} as <image${start + i}>`).join(", "));
}
function inspectKeyframe(kf) {
  const rec = kf.recipe;
  const upd = (fields) => op({ op: "keyframe.update", id: kf.id, fields });
  const st = S.status.keyframes[kf.id] || {};
  const others = Object.values(S.project.keyframes).filter((k) => k.id !== kf.id);
  const generatable = rec.op === "t2i" || rec.op === "edit";
  const els = [
    h("h3", {}, "Keyframe"),
    h("label", {}, "Name"), h("input", { value: kf.name, onchange: (e) => upd({ name: e.target.value }) }),
    h("label", {}, "Made by"),
    h("select", { onchange: (e) => upd({ recipe: { op: e.target.value } }) },
      [["t2i", qi21() ? "Text → image (+ optional references)" : "Text → image (Qwen-Image)"], ["edit", qi21() ? "Edit a keyframe (+ references)" : "Edit other keyframes (Qwen-Image-Edit)"], ["import", "Imported image"], ["grab", "Frame from a clip"]]
        .map(([v, t]) => h("option", { value: v, selected: rec.op === v, disabled: (v === "import" || v === "grab") && rec.op !== v }, t))),
  ];
  if (generatable) els.push(...referencePicker(kf, rec, others, upd));
  if (rec.op === "edit") {
    const base = kfById(rec.parents?.[0]);
    els.push(h("label", {}, "What to change"),
      rec.mask
        ? h("div", { class: "dr-region" },
            h("div", { class: "thumb" }, base && kfThumb(base) ? h("img", { src: kfThumb(base) }) : null, h("img", { class: "mask", src: fileURL(rec.mask) })),
            h("div", {}, h("div", {}, "Only the painted region of image1"),
              h("div", { class: "dr-row", style: { marginTop: "6px" } },
                h("button", { onclick: () => recipeRegionDialog(kf) }, "Edit region…"),
                h("button", { onclick: () => upd({ recipe: { mask: null } }) }, "Whole image"))))
        : h("div", { class: "dr-row" }, h("span", { class: "dr-help", style: { margin: 0 } }, "The whole image"),
            h("button", { class: "fit", disabled: !base || !kfThumb(base), title: base ? "" : "Choose image1 first", onclick: () => recipeRegionDialog(kf) }, "🖌 Only a painted region…")));
  }
  if (generatable) {
    els.push(h("label", {}, rec.op === "edit" ? "Edit instruction" : "Prompt"),
      h("textarea", { rows: 5, value: rec.prompt || "", placeholder: rec.op === "edit" ? "e.g. Same scene at night, the lamp is lit" : "Describe the frame", onchange: (e) => upd({ recipe: { prompt: e.target.value } }) }),
      S.project.settings.style_prompt ? h("div", { class: "dr-help" }, `+ project style: ${S.project.settings.style_prompt}`) : null,
      projectRefsNote(kf),
      h("label", {}, qi21() ? "LoRAs (Qwen Image 2.1)" : "LoRAs (Qwen)"), loraEditor(JSON.parse(JSON.stringify(rec.loras || [])), (l) => upd({ recipe: { loras: l } }), false),
      h("div", { class: "dr-status " + (st.ready ? "ok" : "no") }, st.ready ? "Ready to generate" : `Not ready: ${st.why}`),
      staleBox("keyframe", kf, st),
      h("div", { class: "dr-actions" }, h("div", { class: "dr-row", style: { flex: "1 1 100%" } }, countPicker(),
        h("button", { class: "pri", disabled: !st.ready, onclick: () => generate({ action: "keyframe.generate", id: kf.id, count: getCount() }, kf.name || "keyframe") }, "Generate takes"))));
  } else if (rec.op === "grab" && rec.source) {
    els.push(h("div", { class: "dr-help" }, `Frame ${rec.source.frame} of clip ${segById(rec.source.segment)?.name || rec.source.segment}`));
  }
  const selTake = takeOf(kf);
  if (selTake) els.push(h("label", {}, "Improve"), h("div", { class: "dr-actions", style: { marginTop: 0 } },
    h("button", { title: "Repaint part of the selected take; results become new takes of this keyframe", onclick: () => imageInpaintDialog(kf, selTake) }, "🖌 Inpaint selected take…"),
    h("button", { title: "New keyframe in which only a painted region of this one changes", onclick: () => deriveRegionDialog(kf, selTake) }, "↳ Derive by painting a region…")));
  els.push(h("label", {}, "Use on timeline"), h("div", { class: "dr-actions", style: { marginTop: 0 } },
    h("button", { onclick: () => addSegment({ kind: "generate", name: `From ${kf.name}`, start: { type: "keyframe", keyframe: kf.id } }) }, "+ Clip starting here"),
    h("button", { onclick: () => addSegment({ kind: "still", name: `${kf.name} (hold)`, keyframe: kf.id, frames: 32 }) }, "+ Still"),
    h("button", { onclick: () => newKeyframe("edit", [kf.id]) }, "↳ Derive"),
    h("button", { class: "warn", onclick: () => confirm("Delete keyframe and all its takes?") && op({ op: "keyframe.delete", id: kf.id }).then(() => { S.sel = null; render(); }) }, "Delete")));
  return els;
}

async function addSegment(fields, index) {
  const s = S.project.settings;
  const r = await op({ op: "segment.insert", index, fields: { frames: s.max_segment_frames, ...fields } });
  S.sel = { type: "segment", id: r.id };
  render();
}

const MODE_NAMES = { i2v: "i2v", flf2v: "first/last", inp: "Fun InP", t2v: "t2v", vace: "VACE", camera: "Fun Camera", control: "Fun Control" };
const MODE_DESC = { i2v: "image→video from start frame", flf2v: "first/last-frame interpolation", inp: "first/last frame with Wan 2.2 Fun InP",
  t2v: "text→video", vace: "VACE with surrounding frames as context", camera: "camera move with Wan 2.2 Fun Camera", control: "motion from control video (Wan 2.2 Fun Control)" };
const CAMERA_MOVES = ["Static", "Pan Up", "Pan Down", "Pan Left", "Pan Right", "Zoom In", "Zoom Out", "Anti Clockwise (ACW)", "ClockWise (CW)"];
const motionType = (seg) => seg.motion?.type || "none";

// Wan 2.2 Fun Camera / Fun Control settings of a clip
function motionSection(seg, idx, upd) {
  const mo = { type: "none", camera: "Zoom In", speed: 1, preprocess: "none", offset: 0, canny_low: 0.4, canny_high: 0.8, ...(seg.motion || {}) };
  const set = (patch) => upd({ motion: { ...mo, ...patch } });
  const els = [h("label", {}, "Camera & motion control"),
    h("select", { onchange: (e) => set({ type: e.target.value }) },
      [["none", "None – method above"], ["camera", "Camera move (Wan 2.2 Fun Camera)"], ["control", "Follow a control video (Wan 2.2 Fun Control)"]]
        .map(([v, t]) => h("option", { value: v, selected: mo.type === v }, t)))];
  if (mo.type === "camera") {
    els.push(h("div", { class: "dr-grid2" },
      h("div", {}, h("label", {}, "Move"), h("select", { onchange: (e) => set({ camera: e.target.value }) },
        CAMERA_MOVES.map((c) => h("option", { value: c, selected: mo.camera === c }, c)))),
      h("div", {}, h("label", {}, "Speed"), h("input", { type: "number", step: 0.1, min: 0, max: 10, value: mo.speed, onchange: (e) => set({ speed: parseFloat(e.target.value) }) }))),
      h("div", { class: "dr-help" }, "Starts from the start frame and moves the camera; an end frame is ignored."));
  }
  if (mo.type === "control") {
    const src = mo.source || {};
    const clips = S.project.timeline.filter((x, j) => j !== idx && x.kind !== "still" && x.takes?.length);
    const fin = h("input", { type: "file", accept: "video/*", style: { display: "none" }, onchange: async (e) => {
      const f = e.target.files[0]; if (!f) return;
      const r = await upload(f, "control"); if (r?.path) set({ source: { file: r.path, name: f.name } }); } });
    const cur = src.segment ? `seg:${src.segment}` : src.file ? "file" : "";
    els.push(h("label", {}, "Control video"),
      h("div", { class: "dr-row" },
        h("select", { onchange: (e) => { const v = e.target.value; if (v.startsWith("seg:")) set({ source: { segment: v.slice(4) } }); } },
          h("option", { value: "" }, "— choose —"),
          src.file ? h("option", { value: "file", selected: cur === "file" }, `Uploaded: ${src.name || src.file}`) : null,
          clips.map((x) => h("option", { value: `seg:${x.id}`, selected: cur === `seg:${x.id}` }, `Clip: ${x.name || x.id}`))),
        h("button", { class: "fit", onclick: () => fin.click() }, "Upload…"), fin),
      h("div", { class: "dr-grid2" },
        h("div", {}, h("label", {}, "Preprocess"), h("select", { onchange: (e) => set({ preprocess: e.target.value }) },
          [["none", "None (already depth / pose / edges)"], ["canny", "Canny edges"]].map(([v, t]) => h("option", { value: v, selected: mo.preprocess === v }, t)))),
        h("div", {}, h("label", {}, "Start at frame"), h("input", { type: "number", min: 0, value: mo.offset, onchange: (e) => set({ offset: parseInt(e.target.value, 10) }) }))),
      mo.preprocess === "canny" ? h("div", { class: "dr-grid2" },
        h("div", {}, h("label", {}, "Canny low"), h("input", { type: "number", step: 0.05, min: 0.01, max: 0.99, value: mo.canny_low, onchange: (e) => set({ canny_low: parseFloat(e.target.value) }) })),
        h("div", {}, h("label", {}, "Canny high"), h("input", { type: "number", step: 0.05, min: 0.01, max: 0.99, value: mo.canny_high, onchange: (e) => set({ canny_high: parseFloat(e.target.value) }) }))) : null,
      h("div", { class: "dr-help" }, "Motion and layout follow the control video. The start frame, if set, is used as the appearance reference (it is not pinned as frame 1). Using another clip with Canny edges re-renders it in a new look; for depth or pose, upload a preprocessed video."));
  }
  return els;
}

function inspectSegment(seg) {
  const s = S.project.settings;
  const idx = S.project.timeline.indexOf(seg);
  const upd = (fields) => op({ op: "segment.update", id: seg.id, fields });
  const st = S.status.segments[seg.id] || {};
  const kfs = Object.values(S.project.keyframes);
  const anchorSel = (which) => {
    const a = seg[which];
    const cur = a.type === "keyframe" ? `kf:${a.keyframe}` : a.type;
    const opts = [["none", "— none —"], which === "start" ? ["prev", "↪ Continue from previous clip"] : ["next", "↩ Lead into next clip"],
      ...kfs.map((k) => [`kf:${k.id}`, `Keyframe: ${k.name || k.id}`])];
    return h("select", { onchange: (e) => { const v = e.target.value; upd({ [which]: v.startsWith("kf:") ? { type: "keyframe", keyframe: v.slice(3) } : { type: v, keyframe: null } }); } },
      opts.map(([v, t]) => h("option", { value: v, selected: v === cur, disabled: (v === "prev" && idx === 0) || (v === "next" && idx === S.project.timeline.length - 1) }, t)));
  };
  const kindLabel = { generate: "Generated clip", video: "Imported video", still: "Still (held keyframe)" }[seg.kind];
  const els = [h("h3", {}, kindLabel), h("label", {}, "Name"), h("input", { value: seg.name, onchange: (e) => upd({ name: e.target.value }) })];

  if (seg.kind === "still") {
    els.push(h("label", {}, "Keyframe"), h("select", { onchange: (e) => upd({ keyframe: e.target.value }) },
      h("option", { value: "" }, "—"), kfs.map((k) => h("option", { value: k.id, selected: seg.keyframe === k.id }, k.name || k.id))),
      h("label", {}, `Duration (${fmtSec(seg.frames, s.fps)})`), h("input", { type: "number", min: 1, value: seg.frames, onchange: (e) => upd({ frames: parseInt(e.target.value, 10) }) }));
  }
  if (seg.kind === "generate") {
    els.push(
      h("div", { class: "dr-grid2" },
        h("div", {}, h("label", {}, "Start frame"), anchorSel("start")),
        h("div", {}, h("label", {}, "End frame"), anchorSel("end"))),
      h("label", {}, `Length: ${seg.frames} frames (${fmtSec(seg.frames, s.fps)})`),
      h("input", { type: "range", min: 17, max: s.max_segment_frames, step: 4, value: Math.min(seg.frames, s.max_segment_frames), onchange: (e) => upd({ frames: parseInt(e.target.value, 10) }) }),
      seg.frames > s.max_segment_frames ? h("div", { class: "dr-help" }, "Longer than one generation – use Split.") : null,
      h("label", {}, "What happens"),
      h("textarea", { rows: 5, value: seg.prompt, placeholder: "Describe motion, action and camera, e.g. slow dolly-in as the keeper climbs the path", onchange: (e) => upd({ prompt: e.target.value }) }),
      s.video_style_prompt ? h("div", { class: "dr-help" }, `+ project style: ${s.video_style_prompt}`) : null,
      h("div", { class: "dr-grid2" },
        h("div", {}, h("label", {}, "Method"), motionType(seg) !== "none"
          ? h("select", { disabled: true }, h("option", {}, motionType(seg) === "camera" ? "Fun Camera ↓" : "Fun Control ↓"))
          : h("select", { onchange: (e) => upd({ mode: e.target.value }) },
          [["auto", `Auto${st.mode ? ` (${MODE_NAMES[st.mode] || st.mode})` : ""}`], ["i2v", "Image → video"], ["flf2v", s.flf_engine === "fun_inp" ? "First/last frame (Fun InP)" : "First/last frame"], ["t2v", "Text → video"], ["vace", "VACE (context)"]]
            .map(([v, t]) => h("option", { value: v, selected: seg.mode === v }, t)))),
        motionType(seg) === "none" ? h("div", {}, h("label", {}, "Context frames"), h("input", { type: "number", min: 1, max: 33, value: seg.context_frames || s.context_frames, onchange: (e) => upd({ context_frames: parseInt(e.target.value, 10) }) })) : h("div")),
      motionType(seg) === "none" ? h("label", { class: "dr-chk" }, h("input", { type: "checkbox", checked: seg.use_context !== false, onchange: (e) => upd({ use_context: e.target.checked }) }),
        "Use motion context from neighbouring clips (VACE) instead of a single frame") : null,
      ...motionSection(seg, idx, upd),
      h("label", {}, "LoRAs (Wan)"), loraEditor(JSON.parse(JSON.stringify(seg.loras || [])), (l) => upd({ loras: l }), true),
      h("div", { class: "dr-status " + (st.ready ? "ok" : "no") }, st.ready ? `Ready · ${MODE_DESC[st.mode] || st.mode}${st.why ? ` (${st.why})` : ""}` : `Not ready: ${st.why}`),
      staleBox("segment", seg, st),
      h("div", { class: "dr-actions" }, h("div", { class: "dr-row", style: { flex: "1 1 100%" } }, countPicker(),
        h("button", { class: "pri", disabled: !st.ready, onclick: () => generate({ action: "segment.generate", id: seg.id, count: getCount() }, seg.name || "clip") }, "Generate takes"))));
  }
  if (seg.kind !== "still") {
    els.push(h("div", { class: "dr-grid2" },
      h("div", {}, h("label", {}, "Trim in (frames)"), h("input", { type: "number", min: 0, value: seg.trim_in || 0, onchange: (e) => upd({ trim_in: parseInt(e.target.value, 10) }) })),
      h("div", {}, h("label", {}, "Trim out (frames)"), h("input", { type: "number", min: 0, value: seg.trim_out || 0, onchange: (e) => upd({ trim_out: parseInt(e.target.value, 10) }) }))));
  }
  els.push(h("label", {}, "Timeline"), h("div", { class: "dr-actions", style: { marginTop: 0 } },
    h("button", { title: "New clip after this one that continues from its last frame", onclick: () => addSegment({ kind: "generate", name: `${seg.name || "clip"} (cont.)`, prompt: seg.prompt, start: { type: "prev" }, loras: seg.loras }, idx + 1) }, "+ Extend"),
    seg.kind === "generate" ? h("button", { onclick: () => op({ op: "segment.split", id: seg.id }) }, "Split") : null,
    h("button", { disabled: idx === 0, onclick: () => op({ op: "segment.move", id: seg.id, index: idx - 1 }) }, "◀"),
    h("button", { disabled: idx === S.project.timeline.length - 1, onclick: () => op({ op: "segment.move", id: seg.id, index: idx + 1 }) }, "▶"),
    h("button", { class: "warn", onclick: () => confirm("Remove this clip from the timeline?") && op({ op: "segment.delete", id: seg.id }).then(() => { S.sel = null; render(); }) }, "Delete")));
  return els;
}

// ------------------------------------------------------------------ timeline (bottom)
function segStateDot(seg) {
  if (seg.kind === "still") return "static";
  const st = S.status.segments[seg.id];
  if (st?.stale?.length) return "done stale";
  if (seg.takes?.length) return st?.downstream ? "done downstream" : "done";
  return st?.ready ? "ready" : seg.kind === "generate" ? "blocked" : "static";
}
function renderTimeline() {
  const box = h("div", { class: "dr-tl" });
  if (!S.project) return box;
  const s = S.project.settings;
  const ppf = S.zoom; // pixels per frame
  const total = S.project.timeline.reduce((a, x) => a + (x.kind === "video" ? (takeOf(x)?.frames || x.frames) : x.frames), 0);
  const vin = h("input", { type: "file", accept: "video/*", style: { display: "none" }, onchange: (e) => e.target.files[0] && upload(e.target.files[0], "video") });
  box.append(h("div", { class: "dr-tlbar" },
    h("b", { style: { color: "#e6e6ea" } }, "Timeline"), `${S.project.timeline.length} clips · ${fmtSec(total, s.fps)}`,
    h("span", { style: { flex: 1 } }),
    h("span", {}, h("span", { class: "dr-dot done" }), " has take  ", h("span", { class: "dr-dot done stale" }), " out of date  ", h("span", { class: "dr-dot ready" }), " ready  ", h("span", { class: "dr-dot blocked" }), " waiting  "),
    "zoom", h("input", { type: "range", min: 0.5, max: 4, step: 0.1, value: S.zoom, oninput: (e) => { S.zoom = parseFloat(e.target.value); render(); } })));
  const segs = h("div", { class: "dr-segs" });
  S.project.timeline.forEach((seg, i) => {
    const take = takeOf(seg);
    const frames = seg.kind === "video" ? take?.frames || seg.frames : seg.frames;
    const w = Math.max(90, frames * ppf);
    const film = h("div", { class: "film" });
    if (seg.kind === "still") {
      const kf = kfById(seg.keyframe);
      film.append(kf && kfThumb(kf) ? h("img", { src: kfThumb(kf) }) : h("div", { class: "gap" }, "no keyframe"));
    } else if (take) {
      film.append(h("img", { src: fileURL(take.first) }));
      if (w > 200) film.append(h("img", { src: fileURL(take.last) }));
    } else {
      const pend = pendingFor(`segment:${seg.id}`);
      film.append(h("div", { class: "gap" }, pend ? [h("span", { class: "dr-spin" }), " generating"] : (S.status.segments[seg.id]?.ready ? "gap · ready" : "gap")));
    }
    const anchors = [];
    if (seg.kind === "generate") {
      for (const [which, cls] of [["start", "s"], ["end", "e"]]) {
        const a = seg[which];
        if (a.type === "keyframe" && kfById(a.keyframe)) {
          const th = kfThumb(kfById(a.keyframe));
          anchors.push(h("div", { class: `dr-anchor ${cls}`, title: `${which}: ${kfById(a.keyframe).name}` }, th ? h("img", { src: th }) : h("span", { style: { color: "#e0b33d", fontSize: "18px", display: "flex", alignItems: "center", justifyContent: "center", height: "100%" } }, "?")));
        } else if (a.type === "prev" || a.type === "next") {
          anchors.push(h("div", { class: `dr-anchor ${cls} cont`, title: a.type === "prev" ? "continues previous clip" : "leads into next clip" }, a.type === "prev" ? "↪" : "↩"));
        }
      }
    }
    const el = h("div", {
      class: `dr-seg ${seg.kind}` + (S.sel?.type === "segment" && S.sel.id === seg.id ? " sel" : ""), style: { width: `${w}px` },
      onclick: () => { S.sel = { type: "segment", id: seg.id }; render(); },
      ondragover: (e) => { if (!e.dataTransfer.types.includes("text/director-kf")) return; e.preventDefault();
        const r = el.getBoundingClientRect(); const left = e.clientX - r.left < r.width / 2;
        el.classList.toggle("drop-s", left); el.classList.toggle("drop-e", !left); },
      ondragleave: () => el.classList.remove("drop-s", "drop-e"),
      ondrop: (e) => { e.preventDefault(); el.classList.remove("drop-s", "drop-e");
        const kid = e.dataTransfer.getData("text/director-kf"); if (!kid) return;
        if (seg.kind === "still") return op({ op: "segment.update", id: seg.id, fields: { keyframe: kid } });
        if (seg.kind !== "generate") return toast("Keyframes anchor generated clips", true);
        const r = el.getBoundingClientRect(); const which = e.clientX - r.left < r.width / 2 ? "start" : "end";
        op({ op: "segment.update", id: seg.id, fields: { [which]: { type: "keyframe", keyframe: kid } } }); },
    }, film, ...anchors, finishBadge(seg),
      h("div", { class: "lbl" }, h("span", { class: "dr-dot " + segStateDot(seg) }),
        seg.kind === "generate" && S.status.segments[seg.id]?.mode ? h("span", { class: "dr-badge " + (take?.mode || S.status.segments[seg.id].mode) }, take?.mode || S.status.segments[seg.id].mode) : h("span", { class: "dr-badge" }, seg.kind),
        h("b", {}, seg.name || `#${i + 1}`), h("span", { class: "dur" }, fmtSec(frames, s.fps)),
        seg.takes?.length > 1 ? h("span", { class: "dur" }, `×${seg.takes.length}`) : null));
    segs.append(el);
  });
  const last = S.project.timeline[S.project.timeline.length - 1];
  segs.append(h("div", { class: "dr-addseg" },
    h("button", { onclick: () => addSegment({ kind: "generate", name: "New clip" }) }, "+ Gap"),
    last ? h("button", { onclick: () => addSegment({ kind: "generate", name: `${last.name || "clip"} (cont.)`, start: { type: "prev" }, prompt: last.prompt || "" }) }, "+ Extend last") : null,
    h("button", { onclick: () => addSegment({ kind: "still", name: "Still", frames: 32 }) }, "+ Still"),
    h("button", { onclick: () => vin.click() }, "Import video"), vin));
  // ruler: map seconds to pixels through the actual clip layout (clips have a minimum width)
  const ruler = h("div", { class: "dr-ruler" });
  const layout = []; let px = 0, fr = 0;
  for (const seg of S.project.timeline) {
    const f = seg.kind === "video" ? takeOf(seg)?.frames || seg.frames : seg.frames;
    const w = Math.max(90, f * ppf);
    layout.push({ px, fr, f, w }); px += w + 3; fr += f;
  }
  const step = S.zoom < 1 ? 10 : S.zoom < 2 ? 5 : S.zoom < 3.5 ? 2 : 1;
  for (let t = 0; t * s.fps <= fr; t += step) {
    const F = t * s.fps;
    const L = layout.find((l) => F >= l.fr && F < l.fr + l.f) || layout[layout.length - 1];
    if (!L) break;
    ruler.append(h("span", { style: { left: `${L.px + Math.min(1, (F - L.fr) / L.f) * L.w}px` } }, `${t}s`));
  }
  box.append(h("div", { class: "dr-track" }, ruler, segs));
  return box;
}

// ------------------------------------------------------------------ dialogs
function modal(content, onClose) {
  const m = h("div", { class: "dr-modal", onmousedown: (e) => { if (e.target === m) { m.remove(); onClose?.(); } } }, h("div", { class: "dr-dlg" }, content));
  m.addEventListener("keydown", (e) => { e.stopPropagation(); if (e.key === "Escape") m.remove(); });
  document.body.append(m);
  return m;
}

async function settingsDialog(tab = "general") {
  const md = await J("/director/models").catch((e) => { toast(e.message, true); return null; });
  const s = JSON.parse(JSON.stringify(S.project.settings));
  const f = (k, label, type = "text", attrs = {}) => [h("label", {}, label), h(type === "textarea" ? "textarea" : "input", { type, value: s[k], ...attrs,
    onchange: (e) => { s[k] = type === "number" ? parseFloat(e.target.value) : e.target.value; } })];
  const chk = (k, label) => h("label", { class: "dr-chk" }, h("input", { type: "checkbox", checked: s[k], onchange: (e) => { s[k] = e.target.checked; } }), label);
  const presets = [[832, 480], [480, 832], [1280, 720], [720, 1280], [640, 640], [960, 544]];
  let m;
  s.models = { ...(s.models || {}) };
  const general = h("div", {},
    h("label", {}, "Video resolution"),
    h("select", { onchange: (e) => { const [w, hh] = e.target.value.split("x").map(Number); s.width = w; s.height = hh; } },
      presets.map(([w, hh]) => h("option", { value: `${w}x${hh}`, selected: s.width === w && s.height === hh }, `${w} × ${hh}`))),
    h("div", { class: "dr-grid2" },
      h("div", {}, ...f("fps", "Frames per second", "number")),
      h("div", {}, ...f("max_segment_frames", "Max frames per generated clip", "number", { step: 4 })),
      h("div", {}, ...f("keyframe_megapixels", "Keyframe size (megapixels)", "number", { step: 0.1 })),
      h("div", {}, ...f("takes_per_run", "Default takes per run", "number")),
      h("div", {}, ...f("context_frames", "VACE context frames", "number"))),
    ...f("style_prompt", "Image style (appended to every keyframe prompt)", "textarea"),
    ...f("video_style_prompt", "Video style (appended to every clip prompt)", "textarea"),
    chk("fast_image", qi21() ? "Fast keyframes (Qwen Image 2.1 Turbo LoRA, 8 steps)" : "Fast keyframes (Lightning 4-step LoRA)"),
    chk("fast_video", "Fast video (lightx2v 4-step LoRAs)"),
    h("label", {}, "Project-wide image LoRAs (character / style)"), loraEditor(s.image_loras, (l) => { s.image_loras = l; }, false),
    qi21() ? [h("label", {}, "Reference keyframes for every keyframe (character sheet, location, style)"),
      h("div", { class: "dr-refgrid" }, Object.values(S.project.keyframes).filter((k) => k.selected).map((k) => {
        const on = (s.image_reference_keyframes || []).includes(k.id);
        const el = h("div", { class: "dr-refpick" + (on ? " on" : ""), title: k.name,
          onclick: () => { const L = new Set(s.image_reference_keyframes || []); L.has(k.id) ? L.delete(k.id) : L.add(k.id); s.image_reference_keyframes = [...L]; el.classList.toggle("on"); } },
          h("img", { src: kfThumb(k) }), h("span", {}, k.name || k.id));
        return el;
      })),
      h("div", { class: "dr-help" }, "Qwen Image 2.1 receives these as extra reference images in every text→image, edit and inpaint run, which keeps characters and places consistent without training a LoRA.")] : null,
    h("label", {}, "Project-wide video LoRAs"), loraEditor(s.video_loras, (l) => { s.video_loras = l; }, true));
  const models = md ? modelsPanel(s, md) : h("div", { class: "dr-help" }, "Could not load the model list.");
  s.finish = { ...(s.finish || {}) };
  const finishing = h("div", {});
  const drawFinishing = () => finishing.replaceChildren(
    h("div", { class: "dr-help", style: { marginTop: "10px" } }, "How the selected takes are upscaled for the final video (✨ Final in the header). Defaults come from config.json."),
    finishSettingsForm(s.finish, md, drawFinishing));
  drawFinishing();
  const panes = { general, models, finishing };
  const tabs = h("div", { class: "dr-tabs" }, [["general", "General"], ["models", "Models"], ["finishing", "Finishing"]].map(([k, t]) =>
    h("button", { class: k === tab ? "on" : "", onclick: (e) => { for (const b of tabs.children) b.classList.remove("on"); e.target.classList.add("on");
      for (const [pk, pv] of Object.entries(panes)) pv.style.display = pk === k ? "" : "none"; } }, t)));
  for (const [pk, pv] of Object.entries(panes)) pv.style.display = pk === tab ? "" : "none";
  const body = h("div", { style: { width: "680px" } },
    h("h2", {}, "Project settings"), tabs, general, models, finishing,
    h("div", { class: "dr-actions" }, h("span", { style: { flex: 1 } }),
      h("button", { onclick: () => m.remove() }, "Cancel"),
      h("button", { class: "pri", onclick: async () => { m.remove(); await op({ op: "settings.update", fields: s }); } }, "Save")));
  m = modal(body);
}


// Models tab: per-project choice of checkpoints, text encoders, VAEs, fast LoRAs and sampling.
function modelsPanel(s, md) {
  const files = md.files;
  const has = (kind, v) => (files[kind] || []).some((f) => f.value === v);
  const special = { "@checkpoint": "From checkpoint (all-in-one)", "@none": "None" };
  const srcLabel = { diffusion_models: "diffusion_models", checkpoints: "checkpoints (all-in-one)", gguf: "GGUF (ComfyUI-GGUF)", text_encoders: "text_encoders", vae: "vae", loras: "loras" };
  const wrap = h("div", {});
  const family = h("select", { onchange: (e) => { s.image_model = e.target.value; draw(); } },
    h("option", { value: "" }, `Default (${md.image_model === "qwen" ? "Qwen-Image 2512 / Edit 2511" : "Qwen Image 2.1"})`),
    h("option", { value: "qwen21", selected: s.image_model === "qwen21" }, "Qwen Image 2.1"),
    h("option", { value: "qwen", selected: s.image_model === "qwen" }, "Qwen-Image 2512 + Edit 2511"));
  const groupsBox = h("div", {});
  function control(slot) {
    const cur = s.models[slot.id] ?? "";
    const set = (v) => { if (v === "" || v === undefined || v === null || Number.isNaN(v)) delete s.models[slot.id]; else s.models[slot.id] = v; draw(); };
    const eff = cur === "" ? slot.default : cur === "@none" ? null : cur;
    let el, warn = null;
    if (slot.kind === "int" || slot.kind === "float") {
      el = h("input", { type: "number", step: slot.kind === "int" ? 1 : 0.1, value: cur, placeholder: `default ${slot.default}`,
        onchange: (e) => set(e.target.value === "" ? "" : slot.kind === "int" ? parseInt(e.target.value, 10) : parseFloat(e.target.value)) });
    } else if (slot.kind === "dtype") {
      el = h("select", { onchange: (e) => set(e.target.value) }, h("option", { value: "" }, `Default (${slot.default || "default"})`),
        md.dtypes.map((d) => h("option", { value: d, selected: cur === d }, d)));
    } else {
      const kind = slot.kind;
      const extra = [];
      if (kind === "text_encoder" || kind === "vae") extra.push("@checkpoint");
      if (kind === "lora" || slot.id.endsWith(".low")) extra.push("@none");
      const groups = {};
      for (const f of files[kind] || []) (groups[f.src] = groups[f.src] || []).push(f.value);
      const dflt = slot.default ? `Default (${slot.default}${has(kind, slot.default) ? "" : " – not found"})` : "Default (none)";
      el = h("select", { onchange: (e) => set(e.target.value) },
        h("option", { value: "" }, dflt),
        extra.map((x) => h("option", { value: x, selected: cur === x }, special[x])),
        Object.entries(groups).map(([src, list]) => h("optgroup", { label: srcLabel[src] || src },
          list.map((v) => h("option", { value: v, selected: cur === v }, v)))),
        cur && !extra.includes(cur) && !has(kind, cur) ? h("option", { value: cur, selected: true }, `${cur} – not found`) : null);
      if (eff && !String(eff).startsWith("@") && !has(kind, eff)) warn = h("span", { class: "dr-warn", title: "File not found in ComfyUI's model folders" }, "not installed");
    }
    return h("div", { class: "dr-mrow" + (cur !== "" ? " changed" : "") }, h("span", { class: "lbl" }, slot.label, warn), el);
  }
  function presetBar(group) {
    const ps = Object.entries(md.presets || {}).filter(([, p]) => p.group === group);
    if (!ps.length) return null;
    return h("div", { class: "dr-presets" }, h("span", {}, "Presets:"), ps.map(([k, p]) => h("button", {
      onclick: () => { for (const [sid, v] of Object.entries(p.values)) { if (v === "") delete s.models[sid]; else s.models[sid] = v; } draw(); } }, p.label)));
  }
  function draw() {
    const fam = s.image_model || md.image_model || "qwen21";
    const shown = [fam, "wan", "wan_fun"];
    groupsBox.replaceChildren(...shown.map((g) => {
      const slots = md.slots.filter((x) => x.group === g);
      const changed = slots.filter((x) => (s.models[x.id] ?? "") !== "").length;
      return h("details", { class: "dr-mgroup", open: true },
        h("summary", {}, md.groups[g], changed ? h("span", { class: "dr-badge", style: { marginLeft: "8px" } }, `${changed} changed`) : null,
          changed ? h("button", { style: { marginLeft: "auto", fontSize: "11px", padding: "2px 8px" },
            onclick: (e) => { e.preventDefault(); for (const x of slots) delete s.models[x.id]; draw(); } }, "Reset") : null),
        g === "wan" ? presetBar("wan") : null,
        g === "wan_fun" ? h("div", { class: "dr-mrow" + ((s.flf_engine || "i2v") !== "i2v" ? " changed" : "") },
          h("span", { class: "lbl" }, "First/last-frame clips use"),
          h("select", { onchange: (e) => { s.flf_engine = e.target.value; draw(); } },
            h("option", { value: "i2v", selected: (s.flf_engine || "i2v") === "i2v" }, "Wan 2.2 I2V (first/last frame)"),
            h("option", { value: "fun_inp", selected: s.flf_engine === "fun_inp" }, "Wan 2.2 Fun InP"))) : null,
        slots.map(control));
    }));
  }
  draw();
  wrap.append(
    h("div", { class: "dr-help", style: { marginTop: "10px" } }, "Defaults come from config.json. Pick another file here to use it for this project only: a different precision or quantisation, a fine-tune, an all-in-one checkpoint (its text encoder and VAE can then be set to “From checkpoint”), or a GGUF file" + (files.gguf_installed ? "." : " (needs the ComfyUI-GGUF custom nodes, not detected).")),
    h("label", {}, "Keyframe model family"), family, groupsBox);
  return wrap;
}

// ------------------------------------------------------------------ stale takes
// A take is stale when a keyframe or clip it was made from now has a different selected take.
function staleCount() {
  let direct = 0, downstream = 0;
  for (const kind of ["keyframes", "segments"]) for (const v of Object.values(S.status?.[kind] || {})) {
    if (v.stale?.length) direct++; else if (v.downstream) downstream++;
  }
  return { direct, downstream, total: direct + downstream };
}
const regenerate = (kind, ent) => generate({ action: `${kind}.generate`, id: ent.id, count: 1, select: true }, `update ${ent.name || ent.id}`);
const keepTake = (kind, ent) => op({ op: `${kind}.keep`, id: ent.id });
function staleBox(kind, ent, st) {
  if (st?.stale?.length) {
    return h("div", { class: "dr-status stale" },
      h("div", {}, h("b", {}, "Out of date: "), st.stale.join("; "), "."),
      h("div", { class: "dr-row", style: { marginTop: "6px" } },
        h("button", { class: "pri", disabled: !st.ready, title: st.ready ? "Generate one new take from the current inputs; it replaces the selected take" : st.why, onclick: () => regenerate(kind, ent) }, "↻ Regenerate"),
        h("button", { title: "Keep the selected take and treat it as up to date", onclick: () => keepTake(kind, ent) }, "Keep as is")));
  }
  if (st?.downstream) return h("div", { class: "dr-status downstream" }, "Built from something that is out of date; it will be too once that is regenerated.");
  return null;
}
async function staleDialog() {
  let m;
  const body = h("div", { style: { width: "640px" } });
  const draw = () => {
    if (!document.body.contains(m)) { S.staleRedraw = null; return; }
    const items = [], down = [];
    for (const kf of Object.values(S.project.keyframes)) {
      const st = S.status.keyframes[kf.id];
      if (st?.stale?.length) items.push(["keyframe", kf, st]); else if (st?.downstream) down.push(kf.name || kf.id);
    }
    for (const seg of S.project.timeline) {
      const st = S.status.segments[seg.id];
      if (st?.stale?.length) items.push(["segment", seg, st]); else if (st?.downstream) down.push(seg.name || seg.id);
    }
    if (!items.length && !down.length) { m.remove(); S.staleRedraw = null; return; }
    body.replaceChildren(
      h("h2", {}, "Out-of-date takes"),
      h("div", { class: "dr-help" }, "These were generated from a keyframe or clip that has since got a different selected take (regenerated, inpainted, re-selected or swapped). Regenerate them, or keep them if the change doesn’t matter."),
      h("div", { class: "dr-stalelist" }, items.map(([kind, ent, st]) => {
        const pend = pendingFor(`${kind}:${ent.id}`);
        return h("div", { class: "row" },
          h("div", {}, h("b", {}, (kind === "keyframe" ? "▣ " : "") + (ent.name || ent.id)), h("div", { class: "dr-help", style: { margin: 0 } }, st.stale.join("; "))),
          pend ? h("span", { class: "dr-chip run" }, h("span", { class: "dr-spin" }), " generating") :
            h("button", { disabled: !st.ready, title: st.ready ? "" : st.why, onclick: () => { regenerate(kind, ent); } }, "↻ Regenerate"),
          h("button", { disabled: !!pend, onclick: () => keepTake(kind, ent) }, "Keep"));
      })),
      down.length ? h("div", { class: "dr-help" }, `Further down the chain (stale once the above are regenerated): ${down.join(", ")}.`) : null,
      h("div", { class: "dr-actions" }, h("span", { style: { flex: 1 } }),
        h("button", { onclick: () => { m.remove(); S.staleRedraw = null; } }, "Close"),
        items.length ? h("button", { onclick: async () => {
          const d = await J(`/director/project/${S.pid}/op`, { body: items.map(([kind, ent]) => ({ op: `${kind}.keep`, id: ent.id })) });
          applyData(d); render(); } }, "Keep all") : null,
        h("button", { class: "pri", title: "Regenerate every stale take, then whatever becomes stale because of that, down the timeline (one new take each, which replaces the selection)",
          onclick: () => { m.remove(); S.staleRedraw = null; startAutofill("stale"); } }, "↻ Update all")));
  };
  m = modal(body, () => { S.staleRedraw = null; });
  S.staleRedraw = draw;
  draw();
}

// ------------------------------------------------------------------ finishing (upscale / refine / interpolate)
const stillKeyframes = () => new Set((S.project?.timeline || []).filter((x) => x.kind === "still" && x.keyframe).map((x) => x.keyframe));
function finishState(kind, take) {
  if (!take?.finished) return "missing";
  const key = kind === "segment" ? S.finish?.video_key : S.finish?.still_key;
  return take.finished.key === key ? "ok" : "outdated";
}
const finishLabel = () => { const f = S.finish?.settings; return f ? `${Math.min(f.width, f.height)}p` : "1080p"; };
const finishTarget = (kind, id, take) => `finish:${kind}:${id}:${take}`;
function finishButton(kind, ent, take) {
  const pend = pendingFor(finishTarget(kind, ent.id, take.id));
  const st = finishState(kind, take);
  return h("button", { disabled: !!pend, title: "Upscale this take to the delivery size with the finishing settings (✨ Final)",
    onclick: () => generate({ action: `finish.${kind}`, id: ent.id, take: take.id }, `finish ${ent.name || ent.id}`) },
    pend ? "Finishing…" : st === "ok" ? "✨ Finish again" : st === "outdated" ? "✨ Update finish" : `✨ Finish ${finishLabel()}`);
}
function finishBadge(seg) {
  const kind = seg.kind === "still" ? "keyframe" : "segment";
  const ent = kind === "keyframe" ? kfById(seg.keyframe) : seg;
  const t = ent && takeOf(ent);
  if (!t) return null;
  if (pendingFor(finishTarget(kind, ent.id, t.id))) return h("span", { class: "dr-badge fin", title: "finishing" }, h("span", { class: "dr-spin" }));
  const st = finishState(kind, t);
  if (st === "missing") return null;
  return h("span", { class: "dr-badge fin" + (st === "ok" ? "" : " old"), title: st === "ok" ? `finished ${t.finished.width}×${t.finished.height}` : "finished with other settings – update it in ✨ Final" }, "HD");
}

// output size with the project's aspect ratio (mirrors project.size_for_short_side)
function sizeForShort(short) {
  const s = S.project.settings, ar = s.width / s.height, even = (x) => Math.max(2, Math.round(x / 2) * 2);
  if (Math.abs(ar - 16 / 9) / (16 / 9) < 0.03) return [even((short * 16) / 9), short];
  if (Math.abs(ar - 9 / 16) / (9 / 16) < 0.03) return [short, even((short * 16) / 9)];
  return ar >= 1 ? [even(short * ar), short] : [short, even(short / ar)];
}
// effective finishing settings for an override object (defaults from config via /director/models)
function effFinish(o, md) {
  const D = md?.finish_defaults || S.finish?.settings || {};
  const pick = (v, d) => (v === undefined || v === null || v === "" ? d : v);
  const r = { ...(D.refine || {}) };
  for (const [k, v] of Object.entries(o.refine || {})) if (v !== "" && v !== null && v !== undefined) r[k] = v;
  const e = { method: pick(o.method, D.method), upscale_model: pick(o.upscale_model, D.upscale_model), interpolate: +pick(o.interpolate, D.interpolate || 1),
    interpolation_model: pick(o.interpolation_model, D.interpolation_model), short_side: +pick(o.short_side, D.short_side || 1080), refine: r };
  [e.width, e.height] = o.width && o.height ? [o.width, o.height] : sizeForShort(e.short_side);
  return e;
}
const METHOD_HELP = {
  resize: "Resizes the frames only. Fast, but adds no detail.",
  model: "Runs an upscale model (ESRGAN family, e.g. 4x-UltraSharp) on every frame, then resizes to the output size. Sharp and quick (seconds per clip); fine texture can shimmer a little because each frame is upscaled on its own.",
  refine: "Upscales to the refine size, then runs a short low-denoise pass of the Wan low-noise model with the clip’s prompt and LoRAs. This adds real detail that stays consistent over time, and the colour is matched back to the draft. The upscale model then takes it to the output size. Costs about one fast generation per clip at the refine size.",
};
function finishSettingsForm(o, md, onChange) {
  const e = effFinish(o, md);
  const files = md?.files || {};
  const set = (patch) => { Object.assign(o, patch); for (const k of Object.keys(o)) if (o[k] === "" || o[k] === undefined) delete o[k]; onChange(); };
  const setR = (patch) => { o.refine = { ...(o.refine || {}), ...patch }; for (const k of Object.keys(o.refine)) if (o.refine[k] === "" || o.refine[k] === undefined) delete o.refine[k]; if (!Object.keys(o.refine).length) delete o.refine; onChange(); };
  const D = md?.finish_defaults || {};
  const fileSel = (list, cur, dflt, onSel, extra = []) => {
    const has = (v) => list.some((f) => f.value === v);
    return h("select", { onchange: (ev) => onSel(ev.target.value) },
      h("option", { value: "" }, dflt ? `Default (${dflt}${has(dflt) || !/\.\w+$/.test(dflt) ? "" : " – not found"})` : "Default"),
      extra.map(([v, t]) => h("option", { value: v, selected: cur === v }, t)),
      list.map((f) => h("option", { value: f.value, selected: cur === f.value }, f.value)),
      cur && !extra.some(([v]) => v === cur) && !has(cur) ? h("option", { value: cur, selected: true }, `${cur} – not found`) : null);
  };
  const custom = !!(o.width && o.height);
  const sizeSel = h("select", { onchange: (ev) => { const v = ev.target.value;
      if (v === "custom") { const [w, hh] = sizeForShort(e.short_side); set({ width: w, height: hh, short_side: undefined }); }
      else set({ short_side: +v, width: undefined, height: undefined }); } },
    [720, 1080, 1440, 2160].map((v) => { const [w, hh] = sizeForShort(v); return h("option", { value: v, selected: !custom && e.short_side === v }, `${v}p – ${w} × ${hh}`); }),
    h("option", { value: "custom", selected: custom }, "Custom size…"));
  const fps = S.project.settings.fps;
  const upList = files.upscale || [];
  const els = [
    h("div", { class: "dr-grid2" },
      h("div", {}, h("label", {}, "Output size"), sizeSel),
      h("div", {}, h("label", {}, "Frame rate"), h("select", { onchange: (ev) => set({ interpolate: +ev.target.value }) },
        [1, 2, 3, 4].map((k) => h("option", { value: k, selected: e.interpolate === k }, k === 1 ? `${fps} fps (no interpolation)` : `${fps * k} fps (interpolate ×${k})`))))),
    custom ? h("div", { class: "dr-grid2" },
      h("div", {}, h("label", {}, "Width"), h("input", { type: "number", step: 2, value: o.width, onchange: (ev) => set({ width: parseInt(ev.target.value, 10) }) })),
      h("div", {}, h("label", {}, "Height"), h("input", { type: "number", step: 2, value: o.height, onchange: (ev) => set({ height: parseInt(ev.target.value, 10) }) }))) : null,
    h("div", { class: "dr-help" }, `Clips are generated at ${S.project.settings.width} × ${S.project.settings.height}` +
      (Math.abs(e.width / e.height - S.project.settings.width / S.project.settings.height) > 0.005 ? "; the output aspect ratio differs slightly, so a thin strip is cropped from the edges." : ".")),
    h("label", {}, "Method"),
    h("select", { onchange: (ev) => set({ method: ev.target.value }) },
      [["resize", "Resize only"], ["model", "Upscale model"], ["refine", "Wan refine + upscale model (best)"]].map(([v, t]) => h("option", { value: v, selected: e.method === v }, t))),
    h("div", { class: "dr-help" }, METHOD_HELP[e.method]),
  ];
  if (e.method !== "resize") {
    els.push(h("label", {}, "Upscale model (models/upscale_models)"),
      fileSel(upList, o.upscale_model || "", D.upscale_model, (v) => set({ upscale_model: v })),
      !upList.length ? h("div", { class: "dr-help", style: { color: "#ff9a8a" } }, "No upscale models found. Put one (e.g. 4x-UltraSharp.pth, RealESRGAN_x4plus.pth or 4x_NMKD-Siax_200k.pth) in ComfyUI/models/upscale_models and restart, or the frames are only resized.") : null);
  }
  if (e.method === "refine") {
    const r = e.refine;
    els.push(h("div", { class: "dr-grid2" },
      h("div", {}, h("label", {}, "Refine at"), h("select", { onchange: (ev) => setR({ short_side: +ev.target.value }) },
        [480, 720, 1080].filter((v) => v <= Math.min(e.width, e.height)).map((v) => { const [w, hh] = sizeForShort(v); return h("option", { value: v, selected: +r.short_side === v }, `${v}p – ${w} × ${hh}`); }))),
      h("div", {}, h("label", {}, "Denoise"), h("input", { type: "number", step: 0.05, min: 0.05, max: 0.7, value: r.denoise, onchange: (ev) => setR({ denoise: parseFloat(ev.target.value) }) }))),
      h("div", { class: "dr-help" }, `Denoise 0.2–0.35 adds detail without changing the shot; higher starts repainting it. ${S.project.settings.fast_video ? `Fast video is on: ${r.fast?.steps ?? 4} steps with the lightx2v LoRA.` : `Fast video is off: ${r.quality?.steps ?? 10} steps at CFG ${r.quality?.cfg ?? 3.5}.`} Refining at 1080p takes roughly 2–3× as long as at 720p.`),
      h("div", { class: "dr-grid2" },
        h("div", {}, h("label", {}, "Refine model"), fileSel(files.diffusion || [], o.refine?.model || "", "Wan t2v low noise", (v) => setR({ model: v }))),
        h("div", {}, h("label", {}, "Fast LoRA"), fileSel(files.lora || [], o.refine?.fast_lora || "", "its fast LoRA", (v) => setR({ fast_lora: v }), [["@none", "None"]]))),
      h("div", { class: "dr-grid2" },
        h("div", {}, h("label", {}, "Colour match to draft (0–1)"), h("input", { type: "number", step: 0.1, min: 0, max: 1, value: r.color_match, onchange: (ev) => setR({ color_match: parseFloat(ev.target.value) }) })),
        h("div", {})));
  }
  if (e.interpolate > 1) {
    els.push(h("label", {}, "Interpolation model (models/frame_interpolation)"),
      fileSel(files.interpolation || [], o.interpolation_model || "", D.interpolation_model, (v) => set({ interpolation_model: v })));
  }
  return h("div", {}, els);
}

async function finishDialog() {
  const md = await J("/director/models").catch((e) => { toast(e.message, true); return null; });
  const o = JSON.parse(JSON.stringify(S.project.settings.finish || {}));
  const body = h("div", { style: { width: "760px" } });
  let m;
  const STATE = { ok: ["up to date", "ok"], outdated: ["outdated", "old"], missing: ["not finished", "todo"], no_take: ["no take yet", "none"] };
  const save = async () => { await op({ op: "settings.update", fields: { finish: o } }); };
  const draw = () => {
    if (!document.body.contains(m)) { S.finishRedraw = null; return; }
    const F = S.finish, f = F.settings, c = F.count;
    const todo = c.missing + c.outdated;
    const items = F.items;
    const pending = items.filter((it) => it.take && pendingFor(finishTarget(it.kind, it.id, it.take))).length;
    body.replaceChildren(
      h("h2", {}, `Final video · ${f.width} × ${f.height} @ ${f.fps} fps`),
      h("div", { class: "dr-help" }, "Each clip’s selected take is finished on its own and the result is kept with the take, so re-rendering is instant and changing one clip only re-finishes that clip. Changing these settings marks existing finished versions as outdated."),
      h("div", { class: "dr-fin" },
        h("div", {}, finishSettingsForm(o, md, save)),
        h("div", {},
          h("label", {}, `Timeline (${c.ok}/${items.length - c.no_take} finished)`),
          items.length ? h("div", { class: "dr-finlist" }, items.map((it) => {
            const pend = it.take && pendingFor(finishTarget(it.kind, it.id, it.take));
            const [lbl, cls] = STATE[it.state];
            return h("div", { class: "row" },
              h("span", { class: "nm", title: it.name }, it.kind === "keyframe" ? "▣ " : "", it.name),
              pend ? h("span", { class: "dr-chip run" }, h("span", { class: "dr-spin" }), " finishing") : h("span", { class: `dr-chip ${cls}` }, lbl),
              it.take ? h("button", { disabled: !!pend, title: it.state === "ok" ? "Finish again" : "Finish this one",
                onclick: () => generate({ action: `finish.${it.kind}`, id: it.id, take: it.take }, `finish ${it.name}`).then(draw) }, it.state === "ok" ? "↻" : "✨") : h("span"));
          })) : h("div", { class: "dr-help" }, f.method === "resize" && f.interpolate === 1 ? "Resize only: the final render resizes on the fly, so there is nothing to finish." : "The timeline is empty."),
          c.no_take ? h("div", { class: "dr-help" }, `${c.no_take} clip(s) have no selected take and will be skipped.`) : null)),
      h("div", { class: "dr-actions" },
        h("span", { class: "dr-help", style: { flex: 1, margin: 0 } }, pending ? `${pending} finishing job(s) running…` : todo ? `${todo} to finish. Clips that aren’t finished are resized in the render.` : "Everything is finished."),
        h("button", { onclick: () => { m.remove(); S.finishRedraw = null; } }, "Close"),
        h("button", { disabled: !todo || pending >= todo, onclick: () => generate({ action: "finish.all" }, "finish").then(draw) }, `✨ Finish ${todo || ""}`.trim()),
        h("button", { class: "pri", onclick: () => { m.remove(); S.finishRedraw = null; doRender(true); } }, "▶ Render final")));
  };
  m = modal(body, () => { S.finishRedraw = null; });
  S.finishRedraw = draw;
  draw();
}

// Mask painter shared by image and video inpainting
function maskPainter(imgSrc, title, extraFields, onSubmit, opts = {}) {
  const img = new Image();
  img.crossOrigin = "anonymous";
  img.src = imgSrc;
  const base = h("canvas"), ov = h("canvas", { class: "ov" });
  let brush = 48, erase = false, down = false;
  const ctx = () => ov.getContext("2d");
  img.onload = () => {
    for (const c of [base, ov]) { c.width = img.naturalWidth; c.height = img.naturalHeight; }
    base.getContext("2d").drawImage(img, 0, 0);
    if (opts.initialMask) {  // show an existing region so it can be edited
      const mi = new Image();
      mi.onload = () => {
        const t = h("canvas"); t.width = ov.width; t.height = ov.height;
        const tc = t.getContext("2d"); tc.drawImage(mi, 0, 0, t.width, t.height);
        const d = tc.getImageData(0, 0, t.width, t.height);
        const o = ctx().createImageData(t.width, t.height);
        for (let i = 0; i < d.data.length; i += 4) if (d.data[i] > 127) { o.data[i] = 255; o.data[i + 1] = 51; o.data[i + 2] = 102; o.data[i + 3] = 255; }
        ctx().putImageData(o, 0, 0);
      };
      mi.src = opts.initialMask;
    }
    const scale = Math.min(1, (window.innerWidth * 0.68) / img.naturalWidth, (window.innerHeight * 0.7) / img.naturalHeight);
    for (const c of [base, ov]) { c.style.width = `${img.naturalWidth * scale}px`; c.style.height = `${img.naturalHeight * scale}px`; }
  };
  const paint = (e) => {
    const r = ov.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * ov.width, y = ((e.clientY - r.top) / r.height) * ov.height;
    const c = ctx();
    c.globalCompositeOperation = erase ? "destination-out" : "source-over";
    c.fillStyle = "#ff3366";
    c.beginPath(); c.arc(x, y, brush * (ov.width / r.width) / 2, 0, Math.PI * 2); c.fill();
  };
  ov.addEventListener("pointerdown", (e) => { down = true; ov.setPointerCapture(e.pointerId); paint(e); });
  ov.addEventListener("pointermove", (e) => down && paint(e));
  ov.addEventListener("pointerup", () => { down = false; });
  const toMaskBlob = () => new Promise((res) => {
    const out = h("canvas"); out.width = ov.width; out.height = ov.height;
    const oc = out.getContext("2d"); oc.fillStyle = "#000"; oc.fillRect(0, 0, out.width, out.height);
    const d = ctx().getImageData(0, 0, ov.width, ov.height);
    const od = oc.getImageData(0, 0, out.width, out.height);
    for (let i = 3; i < d.data.length; i += 4) if (d.data[i] > 10) { od.data[i - 3] = od.data[i - 2] = od.data[i - 1] = 255; }
    oc.putImageData(od, 0, 0);
    out.toBlob(res, "image/png");
  });
  let m;
  const promptEl = h("textarea", { rows: 4, placeholder: "What should appear in the painted region?" });
  const grow = h("input", { type: "number", value: opts.grow ?? 16, min: 0, max: 128 });
  const side = h("div", {},
    h("label", {}, "Brush"), h("input", { type: "range", min: 6, max: 200, value: brush, oninput: (e) => { brush = +e.target.value; } }),
    h("div", { class: "dr-row", style: { marginTop: "6px" } },
      h("button", { onclick: (e) => { erase = false; e.target.classList.add("pri"); e.target.nextSibling.classList.remove("pri"); }, class: "pri" }, "Paint"),
      h("button", { onclick: (e) => { erase = true; e.target.classList.add("pri"); e.target.previousSibling.classList.remove("pri"); } }, "Erase"),
      h("button", { onclick: () => ctx().clearRect(0, 0, ov.width, ov.height) }, "Clear")),
    opts.regionOnly ? null : [h("label", {}, "Prompt"), promptEl],
    h("label", {}, "Grow mask (px)"), grow,
    ...(extraFields || []),
    opts.regionOnly ? null : [h("label", {}, "Takes"), h("input", { type: "number", min: 1, max: 8, value: S.project.settings.takes_per_run || 2, id: "dr-mask-count" })],
    h("div", { class: "dr-actions" },
      h("button", { onclick: () => m.remove() }, "Cancel"),
      h("button", { class: "pri", onclick: async () => {
        const blob = await toMaskBlob();
        const up = await upload(new File([blob], "mask.png", { type: "image/png" }), "mask");
        if (!up) return;
        m.remove();
        onSubmit({ mask: up.path, prompt: promptEl.value, grow: +grow.value, count: +document.querySelector("#dr-mask-count")?.value || 1 });
      } }, opts.submitLabel || "Generate")));
  m = modal(h("div", {}, h("h2", {}, title), h("div", { class: "dr-mask" }, h("div", { class: "cv" }, base, ov), side)));
  return { promptEl };
}

// New keyframe that repaints only a region of this one (recipe: edit + mask, image1 = this keyframe)
function deriveRegionDialog(kf, take) {
  maskPainter(fileURL(take.file), `Derive from ${kf.name} – paint the region to change`, [h("div", { class: "dr-help" },
    "Creates a new keyframe from this one in which only the painted region is regenerated. The new keyframe keeps the region, so you can regenerate or edit it later.")],
    async (r) => {
      const res = await op({ op: "keyframe.create", fields: { name: `${kf.name} (edit)`, recipe: { op: "edit", prompt: r.prompt, parents: [kf.id], mask: r.mask, grow: r.grow } } });
      S.sel = { type: "keyframe", id: res.id }; render();
      if (r.prompt.trim()) generate({ action: "keyframe.generate", id: res.id, count: r.count }, `${kf.name} (edit)`);
      else toast("Write what should change, then press Generate takes");
    });
}

// Region editor for a derived keyframe's recipe (no generation)
function recipeRegionDialog(kf) {
  const base = kfById(kf.recipe.parents?.[0]);
  const bt = takeOf(base);
  if (!bt) return toast("Choose image1 (with a selected take) first", true);
  maskPainter(fileURL(bt.file), `${kf.name} – region of image1 to change`, [h("div", { class: "dr-help" }, "Everything outside the painted region stays exactly as in image1.")],
    (r) => op({ op: "keyframe.update", id: kf.id, fields: { recipe: { mask: r.mask, grow: r.grow } } }),
    { regionOnly: true, submitLabel: "Use this region", grow: kf.recipe.grow, initialMask: kf.recipe.mask ? fileURL(kf.recipe.mask) : null });
}

function imageInpaintDialog(kf, take) {
  const p = maskPainter(fileURL(take.file), `Inpaint – ${kf.name}`, [h("div", { class: "dr-help" }, qi21() ? "Only the painted area is regenerated: Qwen Image 2.1 edits a crop around it (the crop is image1, so the prompt can say what to change there), then it is blended back. Results are added as new takes of this keyframe." : "Only the painted area is regenerated (Qwen-Image-Edit on a crop around it), then blended back. Results are added as new takes of this keyframe.")],
    (r) => generate({ action: "keyframe.inpaint", id: kf.id, take: take.id, mask: r.mask, prompt: r.prompt, grow: r.grow, count: r.count }, `inpaint ${kf.name}`));
  p.promptEl.value = "";
}

function videoInpaintDialog(seg, take) {
  const v = S.root.querySelector("#dr-video");
  const fps = S.project.settings.fps;
  const cur = v ? Math.round(v.currentTime * fps) : 0;
  // capture current frame from the video element (same origin)
  const c = h("canvas");
  let src = fileURL(take.first);
  if (v && v.videoWidth) {
    // paint on the draft's pixel grid; a finished version on screen is a centre crop of it, so map it back
    const dw = take.width || S.project.settings.width, dh = take.height || S.project.settings.height;
    const sc = Math.max(v.videoWidth / dw, v.videoHeight / dh), rw = v.videoWidth / sc, rh = v.videoHeight / sc;
    c.width = dw; c.height = dh;
    c.getContext("2d").drawImage(v, (dw - rw) / 2, (dh - rh) / 2, rw, rh);
    src = c.toDataURL("image/png");
  }
  const n = take.frames || 81;
  const a = h("input", { type: "number", min: 0, max: n - 1, value: 0 });
  const b = h("input", { type: "number", min: 0, max: n - 1, value: n - 1 });
  const p = maskPainter(src, `Inpaint clip – ${seg.name} (frame ${cur})`, [
    h("div", { class: "dr-grid2" }, h("div", {}, h("label", {}, "From frame"), a), h("div", {}, h("label", {}, "To frame"), b)),
    h("div", { class: "dr-help" }, "The painted region is regenerated with Wan VACE over the frame range (a static mask, so paint generously around moving objects). Everything outside the mask is kept pixel-exact; the result becomes a new take.")],
    (r) => generate({ action: "segment.inpaint", id: seg.id, take: take.id, mask: r.mask, prompt: r.prompt || seg.prompt, grow: r.grow, count: r.count, frame_start: +a.value, frame_end: +b.value }, `inpaint ${seg.name}`));
  p.promptEl.value = seg.prompt || "";
}

// ------------------------------------------------------------------ registration
app.registerExtension({
  name: "comfyui.director",
  commands: [{ id: "Director.Open", label: "Open Director", icon: "pi pi-video", function: () => openPanel() }],
  keybindings: [{ combo: { key: "d", alt: true }, commandId: "Director.Open" }],
  menuCommands: [{ path: ["Director"], commands: ["Director.Open"] }],
  async setup() {
    const em = app.extensionManager;
    if (em?.registerSidebarTab) {
      em.registerSidebarTab({
        id: "director", icon: "pi pi-video", title: "Director", tooltip: "Director – long video timeline (Alt+D)", type: "custom",
        render: (el) => {
          el.replaceChildren(h("div", { style: { padding: "12px", display: "flex", flexDirection: "column", gap: "10px" } },
            h("b", {}, "Director"), h("div", { style: { opacity: 0.75, fontSize: "12px" } }, "Build long videos from Qwen-Image keyframes and Wan clips on a timeline."),
            h("button", { class: "p-button p-component", onclick: openPanel, style: { padding: "8px" } }, "Open Director")));
        },
      });
    }
    document.body.append(h("button", { class: "dr-fab", onclick: openPanel, title: "Open Director (Alt+D)" }, "🎬 Director"));
    if (!document.querySelector(`link[href="${CSS_URL}"]`)) document.head.append(h("link", { rel: "stylesheet", href: CSS_URL }));
    if (new URLSearchParams(location.search).get("director") === "1") setTimeout(openPanel, 500);
  },
});
window.ComfyDirector = { open: openPanel, state: S, render, planProgress, jobFraction, jobStage };
