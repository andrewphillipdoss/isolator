// Iso layer mixer. Every stem plays from one AudioContext clock, so layers
// stay sample-locked; separate <audio> elements can drift by milliseconds,
// and layered drums then phase.
"use strict";

const $ = (id) => document.getElementById(id);
const GROUPS = [
  ["pieces", "Pieces", "--pieces"],
  ["triggers", "Triggers", "--triggers"],
  ["kit", "Full kit", "--kit"],
  ["room", "Room", "--room"],
  ["song", "Song", "--song"],
];
const dbToGain = (db) => (db <= -60 ? 0 : Math.pow(10, db / 20));
const fmt = (s) => `${Math.floor(s / 60)}:${(s % 60).toFixed(1).padStart(4, "0")}`;

let ctx = null;
let master = null;
let kit = null; // {name, tracks, report}
const buffers = new Map(); // track name -> AudioBuffer
const loading = new Map(); // track name -> Promise
const gains = new Map(); // track name -> GainNode
let sources = [];
let playing = false;
let startAt = 0; // ctx time when playback (re)started
let startOffset = 0; // song position at startAt
let duration = 0;
let saveTimer = null;

function audio() {
  if (!ctx) {
    // 44.1 kHz keeps a dozen decoded 4-minute stems around 1 GB instead of more at 48k.
    ctx = new AudioContext({ latencyHint: "playback", sampleRate: 44100 });
    master = ctx.createGain();
    master.gain.value = dbToGain(parseFloat($("master").value));
    master.connect(ctx.destination);
  }
  return ctx;
}

const position = () => (playing ? startOffset + (ctx.currentTime - startAt) : startOffset);
const anySolo = () => kit.tracks.some((t) => t.solo);
const audible = (t) => !t.muted && (!anySolo() || t.solo);

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${r.status} ${await r.text()}`);
  return r.headers.get("content-type")?.includes("json") ? r.json() : r;
}

// ---------- library & jobs ----------
async function refreshLibrary(active) {
  const items = await api("/api/kits");
  const ul = $("library");
  ul.innerHTML = items.length ? "" : '<li class="muted">Nothing yet</li>';
  for (const k of items) {
    const li = document.createElement("li");
    li.innerHTML = `<span></span><span>${fmt(k.seconds || 0)} · ${k.bpm ?? "?"} BPM</span>`;
    li.firstChild.textContent = k.name;
    if (k.name === active) li.classList.add("active");
    li.onclick = () => openKit(k.name);
    ul.appendChild(li);
  }
}

async function refreshTriggerKits() {
  const r = await api("/api/trigger-kits");
  const sel = $("kit");
  for (const k of r.kits) {
    const o = document.createElement("option");
    o.value = o.textContent = k;
    sel.appendChild(o);
  }
  sel.title = `Trigger kits live in ${r.root}/<kit>/<piece>/*.wav`;
}

function wireUpload() {
  const drop = $("drop");
  const input = $("file");
  const pick = (f) => {
    if (!f) return;
    input._file = f;
    $("dropLabel").textContent = f.name;
    $("start").disabled = false;
  };
  input.onchange = () => pick(input.files[0]);
  drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
  drop.ondragleave = () => drop.classList.remove("over");
  drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); pick(e.dataTransfer.files[0]); };

  $("start").onclick = async () => {
    const fd = new FormData();
    fd.append("file", input._file);
    fd.append("preset", $("preset").value);
    fd.append("kit", $("kit").value);
    fd.append("gate", $("gate").checked);
    $("start").disabled = true;
    $("job").hidden = false;
    const job = await api("/api/jobs", { method: "POST", body: fd });
    pollJob(job.id);
  };
}

async function pollJob(id) {
  const j = await api(`/api/jobs/${id}`);
  $("jobBar").style.width = `${Math.round((j.progress || 0) * 100)}%`;
  $("jobMsg").textContent = j.state === "error" ? `Failed: ${j.message}` : j.message;
  if (j.state === "done") {
    $("start").disabled = false;
    await refreshLibrary(j.name);
    openKit(j.name);
  } else if (j.state === "error") {
    $("start").disabled = false;
  } else {
    setTimeout(() => pollJob(id), 1000);
  }
}

// ---------- mixer ----------
async function openKit(name) {
  stop();
  buffers.clear();
  loading.clear();
  gains.clear();
  kit = await api(`/api/kits/${encodeURIComponent(name)}`);
  for (const t of kit.tracks) t.solo = false;
  duration = kit.report.seconds;
  startOffset = 0;
  $("empty").hidden = true;
  $("mixer").hidden = false;
  $("kitName").textContent = name;
  const hits = Object.entries(kit.report.hits || {}).map(([k, v]) => `${k} ${v}`).join(" · ");
  $("kitInfo").textContent = `${fmt(duration)} · ${kit.report.bpm} BPM · ${hits}`;
  const base = `/files/${encodeURIComponent(name)}/`;
  $("dlMidi").href = base + encodeURIComponent(kit.report.midi);
  $("dlMidi").download = kit.report.midi;
  $("dlRpp").href = base + "layer_kit.rpp";
  $("dlRpp").download = `${name}.rpp`;
  $("dlZip").href = `/api/kits/${encodeURIComponent(name)}/zip`;
  renderStrips();
  refreshLibrary(name);
  for (const t of kit.tracks) if (audible(t)) ensureBuffer(t);
}

function renderStrips() {
  const root = $("groups");
  root.innerHTML = "";
  for (const [g, label, color] of GROUPS) {
    const tracks = kit.tracks.filter((t) => t.group === g);
    if (!tracks.length) continue;
    const box = document.createElement("div");
    box.className = "group";
    box.style.setProperty("--c", `var(${color})`);
    box.innerHTML = `<h3>${label}</h3>`;
    for (const t of tracks) box.appendChild(strip(t));
    root.appendChild(box);
  }
}

function strip(t) {
  const el = document.createElement("div");
  el.className = "strip";
  el.innerHTML = `
    <div class="name"></div>
    <canvas width="600" height="60"></canvas>
    <input type="range" min="-48" max="12" step="0.5">
    <div class="db"></div>
    <div class="ms"><button class="m" title="Mute">M</button><button class="s" title="Solo">S</button></div>`;
  el.querySelector(".name").textContent = t.name;
  const fader = el.querySelector("input");
  const db = el.querySelector(".db");
  const [mBtn, sBtn] = el.querySelectorAll(".ms button");
  fader.value = t.gain_db;
  const paint = () => {
    db.textContent = `${t.gain_db > 0 ? "+" : ""}${t.gain_db.toFixed(1)} dB`;
    mBtn.classList.toggle("on", t.muted);
    sBtn.classList.toggle("on", t.solo);
    el.classList.toggle("off", !audible(t));
  };
  fader.oninput = () => { t.gain_db = parseFloat(fader.value); paint(); applyGains(); scheduleSave(); };
  fader.ondblclick = () => { fader.value = 0; fader.oninput(); };
  mBtn.onclick = () => { t.muted = !t.muted; onAudibilityChange(); scheduleSave(); };
  sBtn.onclick = () => { t.solo = !t.solo; onAudibilityChange(); };
  t._paint = paint;
  t._canvas = el.querySelector("canvas");
  paint();
  if (buffers.has(t.name)) drawWave(t);
  return el;
}

function onAudibilityChange() {
  for (const t of kit.tracks) t._paint();
  applyGains();
  for (const t of kit.tracks) if (audible(t) && !buffers.has(t.name)) ensureBuffer(t).then(() => playing && joinLate(t));
}

function applyGains() {
  if (!ctx) return;
  for (const t of kit.tracks) {
    const g = gains.get(t.name);
    if (g) g.gain.setTargetAtTime(audible(t) ? dbToGain(t.gain_db) : 0, ctx.currentTime, 0.01);
  }
}

function ensureBuffer(t) {
  if (buffers.has(t.name)) return Promise.resolve(buffers.get(t.name));
  if (!loading.has(t.name)) {
    const url = `/files/${encodeURIComponent(kit.name)}/${encodeURIComponent(t.file)}`;
    const p = fetch(url)
      .then((r) => r.arrayBuffer())
      .then((ab) => audio().decodeAudioData(ab))
      .then((buf) => { buffers.set(t.name, buf); drawWave(t); return buf; });
    loading.set(t.name, p);
  }
  return loading.get(t.name);
}

function drawWave(t) {
  const c = t._canvas;
  const buf = buffers.get(t.name);
  if (!c || !buf) return;
  const g = c.getContext("2d");
  const w = c.width, h = c.height;
  const data = buf.getChannelData(0);
  const step = Math.max(1, Math.floor(data.length / w));
  g.clearRect(0, 0, w, h);
  g.fillStyle = getComputedStyle(c).getPropertyValue("--c") || "#888";
  for (let x = 0; x < w; x++) {
    let peak = 0;
    for (let i = x * step, end = Math.min(data.length, i + step); i < end; i += 4) peak = Math.max(peak, Math.abs(data[i]));
    const y = Math.min(1, peak) * (h / 2);
    g.fillRect(x, h / 2 - y, 1, Math.max(1, 2 * y));
  }
}

function gainFor(t) {
  if (!gains.has(t.name)) {
    const g = ctx.createGain();
    g.gain.value = audible(t) ? dbToGain(t.gain_db) : 0;
    g.connect(master);
    gains.set(t.name, g);
  }
  return gains.get(t.name);
}

function startSource(t, when, offset) {
  const buf = buffers.get(t.name);
  if (!buf || offset >= buf.duration) return;
  const src = ctx.createBufferSource();
  src.buffer = buf;
  src.connect(gainFor(t));
  src.start(when, offset);
  sources.push(src);
}

// A track that finishes loading mid-playback joins on the shared clock, so it lands sample-locked.
function joinLate(t) {
  const when = ctx.currentTime + 0.05;
  startSource(t, when, startOffset + (when - startAt));
}

async function play() {
  audio();
  await ctx.resume();
  const needed = kit.tracks.filter(audible);
  $("play").disabled = true;
  await Promise.all(needed.map(ensureBuffer));
  $("play").disabled = false;
  const when = ctx.currentTime + 0.1;
  for (const t of kit.tracks) if (buffers.has(t.name)) startSource(t, when, startOffset);
  startAt = when;
  playing = true;
  $("play").innerHTML = "&#10074;&#10074;";
  tick();
}

function pause() {
  if (!playing) return;
  startOffset = position();
  for (const s of sources) try { s.stop(); } catch (_) {}
  sources = [];
  playing = false;
  $("play").innerHTML = "&#9654;";
}

function stop() {
  pause();
  startOffset = 0;
  paintTime();
}

function paintTime() {
  const p = Math.min(position(), duration);
  $("time").textContent = fmt(p);
  $("seekFill").style.width = `${(100 * p) / (duration || 1)}%`;
}

function tick() {
  if (!playing) return;
  if (position() >= duration) return stop();
  paintTime();
  requestAnimationFrame(tick);
}

function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    api(`/api/kits/${encodeURIComponent(kit.name)}/session`, {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ tracks: kit.tracks.map(({ name, gain_db, muted }) => ({ name, gain_db, muted })) }),
    });
  }, 400);
}

async function bounce(withSong) {
  clearTimeout(saveTimer);
  await api(`/api/kits/${encodeURIComponent(kit.name)}/session`, {
    method: "PUT",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ tracks: kit.tracks.map(({ name, gain_db, muted }) => ({ name, gain_db, muted })) }),
  });
  const r = await api(`/api/kits/${encodeURIComponent(kit.name)}/bounce?with_song=${withSong}`, { method: "POST" });
  const a = document.createElement("a");
  a.href = `/files/${encodeURIComponent(kit.name)}/${encodeURIComponent(r.file)}`;
  a.download = `${kit.name}_${r.file}`;
  a.click();
}

function wireTransport() {
  $("play").onclick = () => (playing ? pause() : play());
  $("stop").onclick = stop;
  $("seek").onclick = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const was = playing;
    pause();
    startOffset = Math.max(0, Math.min(duration, ((e.clientX - r.left) / r.width) * duration));
    paintTime();
    if (was) play();
  };
  $("master").oninput = () => {
    const v = parseFloat($("master").value);
    $("masterDb").textContent = `${v.toFixed(1)} dB`;
    if (master) master.gain.setTargetAtTime(dbToGain(v), ctx.currentTime, 0.01);
  };
  $("bounce").onclick = () => bounce(false);
  $("bounceSong").onclick = () => bounce(true);
  document.addEventListener("keydown", (e) => {
    if (e.code === "Space" && kit && e.target.tagName !== "INPUT") { e.preventDefault(); $("play").click(); }
  });
}

wireUpload();
wireTransport();
refreshLibrary();
refreshTriggerKits();
