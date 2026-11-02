"use strict";
// admute studio: recordings, a timeline next to the video, labels, tuning, live status and jobs.

const $ = (s, el = document) => el.querySelector(s);
const main = $("#main");
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

function hms(t, frac) {
  if (t == null || !isFinite(t)) return "–";
  const neg = t < 0; t = Math.abs(t);
  const s = frac ? t : Math.round(t);
  const h = Math.floor(s / 3600), m = Math.floor(s / 60) % 60, sec = s % 60;
  const ss = frac ? sec.toFixed(1).padStart(4, "0") : String(Math.floor(sec)).padStart(2, "0");
  return (neg ? "-" : "") + `${h}:${String(m).padStart(2, "0")}:${ss}`;
}
function parseHms(s) {
  const parts = String(s).trim().split(":").map(Number);
  if (parts.some(isNaN)) return null;
  return parts.reduce((a, p) => a * 60 + p, 0);
}
const dur = s => s >= 60 ? `${Math.floor(s / 60)}m ${String(Math.round(s % 60)).padStart(2, "0")}s` : `${Math.round(s)}s`;
const gb = b => (b / 1e9).toFixed(1) + " GB";
const ago = t => { const s = Date.now() / 1000 - t; return s < 90 ? "just now" : s < 5400 ? `${Math.round(s / 60)} min ago` : s < 129600 ? `${Math.round(s / 3600)} h ago` : `${Math.round(s / 86400)} days ago`; };

async function api(path, opts = {}) {
  if (opts.json !== undefined) {
    opts = { ...opts, method: opts.method || "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(opts.json) };
  }
  const r = await fetch(path, opts);
  if (r.status === 401) { location.href = "login"; throw new Error("signed out"); }
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || r.statusText);
  return body;
}
let toastTimer;
function toast(msg) {
  const t = $("#toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.hidden = true, 3500);
}
const debounce = (fn, ms) => { let h; return (...a) => { clearTimeout(h); h = setTimeout(() => fn(...a), ms); }; };

// ---------- routing ----------
let cleanup = null;
let timers = [];
function every(ms, fn) { timers.push(setInterval(fn, ms)); }
function route() {
  if (cleanup) { cleanup(); cleanup = null; }
  timers.forEach(clearInterval); timers = [];
  const h = location.hash.slice(1) || "/";
  const [, view, arg] = h.split("/");
  document.querySelectorAll("nav a").forEach(a => a.classList.toggle("on", a.dataset.nav === (view === "rec" ? "recordings" : view || "recordings")));
  if (view === "rec" && arg) return recView(decodeURIComponent(arg));
  if (view === "live") return liveView();
  if (view === "jobs") return jobsView();
  return listView();
}
window.addEventListener("hashchange", route);

async function jobBadge() {
  try {
    const jobs = await api("api/jobs");
    const n = jobs.filter(j => j.state === "running").length;
    const b = $("#jobcount"); b.hidden = !n; b.textContent = n;
  } catch (e) { /* ignore */ }
}
setInterval(jobBadge, 5000); jobBadge();

// ---------- recordings ----------
let META = null; // defaults + tunable ranges
async function listView() {
  main.innerHTML = `<h1>Recordings</h1><div class="sub">Survey a recording to see what would be muted, then open it to check, label and tune.</div><div class="card" id="list"><div class="empty">Loading…</div></div>`;
  const draw = async () => {
    const d = await api("api/recordings"); META = d;
    const rows = d.recordings.map(r => {
      let status;
      if (r.job) status = `<span class="tag warn">${esc(r.job.what)} ${esc(r.job.progress || "starting")}</span>`;
      else if (r.recording) status = `<span class="tag warn">recording</span>`;
      else if (r.surveyed) status = `<span class="tag">surveyed ${ago(r.surveyed_at)}</span>`;
      else status = `<span class="dim">not surveyed</span>`;
      const sc = r.score;
      const score = sc ? `<span title="seconds of ads heard">${sc.ad_heard}s</span> / <span title="seconds of game muted">${sc.show_muted}s</span>` : "";
      const act = r.job || r.recording ? "" : `<button data-survey="${esc(r.name)}">${r.surveyed ? "Survey again" : "Survey"}</button>`;
      return `<tr class="click" data-open="${esc(r.name)}">
        <td><b>${esc(r.title)}</b><div class="dim hide-sm" style="font-size:12px">${esc(r.path)} · ${gb(r.size)}</div></td>
        <td class="num">${r.length ? hms(r.length) : ""}</td>
        <td>${status}</td>
        <td class="num">${r.surveyed ? r.breaks : ""}</td>
        <td>${r.labelled ? `<span class="tag good">${r.label_count} ads, reviewed</span>` : r.label_count ? `<span class="tag">${r.label_count} ads, draft</span>` : ""}</td>
        <td class="num hide-sm">${score}</td>
        <td style="text-align:right">${act}</td></tr>`;
    }).join("");
    $("#list").innerHTML = rows ? `<table><thead><tr><th>Recording</th><th class="num">Length</th><th>Status</th><th class="num">Mutes</th><th>Labels</th><th class="num hide-sm" title="ads heard / game muted, against the labels">Heard / muted</th><th></th></tr></thead><tbody>${rows}</tbody></table>`
      : `<div class="empty">No .ts recordings found.</div>`;
  };
  $("#list").addEventListener("click", async e => {
    const s = e.target.closest("[data-survey]");
    if (s) {
      e.stopPropagation();
      try { await api("api/jobs", { json: { kind: "survey", name: s.dataset.survey } }); toast("Survey started"); draw(); } catch (err) { toast(err.message); }
      return;
    }
    const tr = e.target.closest("[data-open]");
    if (tr) location.hash = "#/rec/" + encodeURIComponent(tr.dataset.open);
  });
  await draw();
  every(5000, draw);
}

// ---------- one recording ----------
async function recView(name) {
  if (!META) META = await api("api/recordings");
  main.innerHTML = `<div class="empty">Loading…</div>`;
  let d = await api("api/rec/" + encodeURIComponent(name));
  const R = {
    name, d, tl: d.timeline, labels: d.labels, survey: d.result ? d.result.breaks : [],
    settings: { ...META.defaults }, sim: null, simBase: null, hear: "survey", base: 0, h: 720,
    sel: -1, len: d.timeline?.length || d.result?.length || 3600, tab: "breaks", drag: null, hover: null,
  };
  R.view = [0, R.len];
  try { R.tab = localStorage.getItem("studio.tab") || "breaks"; } catch (e) { /* no storage */ }

  main.innerHTML = `
    <div class="rec-head">
      <div class="grow"><a href="#/" class="dim" style="text-decoration:none">← Recordings</a><h1>${esc(d.title)}</h1>
        <div class="dim" id="recinfo"></div></div>
    </div>
    <div class="rec-grid">
      <div>
        <div class="player"><video id="vid" playsinline></video><div class="badge" id="badge">sound on</div><div class="clock" id="clock">0:00:00</div><div class="hint" id="hint">Click the timeline or a break to play from there</div></div>
        <div class="controls">
          <button data-skip="-30">−30</button><button data-skip="-10">−10</button><button data-skip="-5">−5</button>
          <button id="play" class="primary" style="min-width:70px">Play</button>
          <button data-skip="5">+5</button><button data-skip="10">+10</button><button data-skip="30">+30</button>
          <span class="sep"></span>
          <label>Mute like <select id="hear"><option value="survey">the survey</option><option value="tuned">tuned settings</option><option value="labels">my labels</option><option value="off">nothing (sound on)</option></select></label>
          <label>Quality <select id="quality"><option>360</option><option>540</option><option selected>720</option><option>1080</option></select></label>
          <span class="sep"></span>
          <label>Go to <input type="text" id="goto" placeholder="1:23:45" style="width:84px"></label>
        </div>
        <div class="legend">
          <span><i style="background:var(--g0)"></i>graphic 1 match</span><span><i style="background:var(--g1)"></i>graphic 2 match</span>
          <span><i style="background:var(--surface)"></i>playing surface</span><span><i style="background:var(--black)"></i>black</span><span><i style="background:var(--silent)"></i>silence</span>
          <span><i style="background:var(--label)"></i>labelled ads</span><span><i style="background:var(--survey)"></i>survey mutes</span><span><i style="background:var(--tuned)"></i>tuned mutes</span>
          <span class="dim">scroll to zoom · drag to pan · shift+drag to label an ad · <span class="kbd">space</span> play · <span class="kbd">←</span><span class="kbd">→</span> 5 s · <span class="kbd">I</span>/<span class="kbd">O</span> set ad start/end</span>
        </div>
        <div class="timeline-wrap"><canvas class="timeline" id="tl"></canvas><div class="tip" id="tip" hidden></div></div>
      </div>
      <div class="card side">
        <div class="tabs"><button data-tab="breaks">Breaks</button><button data-tab="labels">Labels</button><button data-tab="tune">Tune</button><button data-tab="info">Info</button></div>
        <div id="panel"></div>
      </div>
    </div>`;

  const vid = $("#vid"), canvas = $("#tl"), tip = $("#tip");
  const now = () => R.base + (vid.currentTime || 0);
  const hearSet = () => R.hear === "survey" ? R.survey : R.hear === "tuned" ? (R.sim?.breaks || R.survey) : R.hear === "labels" ? R.labels.ads : null;
  const within = (list, t) => !!list && list.some(([s, e]) => t >= s && t < e);

  function info() {
    const r = R.d.result;
    $("#recinfo").innerHTML = r ? `${hms(r.length)} · ${r.breaks.length} mutes in the survey · graphics: ${esc(r.graphics || "none found")}`
      : R.d.job ? `${esc(R.d.job.what)}: ${esc(R.d.job.progress || "starting")}` : "Not surveyed yet: open Info to start a survey.";
  }
  info();

  // --- video ---
  function seek(t, play = true) {
    t = clamp(t, 0, R.len);
    const rel = t - R.base, b = vid.buffered;
    if (vid.currentSrc && rel >= 0 && b.length && rel <= b.end(b.length - 1)) vid.currentTime = rel;
    else { R.base = t; vid.src = `api/rec/${encodeURIComponent(name)}/video?t=${t.toFixed(1)}&h=${R.h}`; }
    $("#hint").hidden = true;
    if (play) vid.play().catch(() => {});
    draw();
  }
  R.seek = seek;
  $("#play").onclick = () => { if (!vid.currentSrc) seek(R.view[0]); else if (vid.paused) vid.play(); else vid.pause(); };
  vid.onplay = () => $("#play").textContent = "Pause";
  vid.onpause = () => $("#play").textContent = "Play";
  document.querySelectorAll("[data-skip]").forEach(b => b.onclick = () => seek(now() + Number(b.dataset.skip), !vid.paused || !vid.currentSrc));
  $("#hear").onchange = e => { R.hear = e.target.value; tick(); };
  $("#quality").onchange = e => { R.h = Number(e.target.value); if (vid.currentSrc) { const t = now(); R.base = -1e9; seek(t, !vid.paused); } };
  $("#goto").onkeydown = e => { if (e.key === "Enter") { const t = parseHms(e.target.value); if (t != null) seek(t); } };

  let lastDraw = 0;
  function tick() {
    const t = now();
    const muted = vid.currentSrc && within(hearSet(), t);
    vid.muted = !!muted;
    const b = $("#badge");
    if (b) { b.textContent = muted ? "MUTED" : "sound on"; b.classList.toggle("muted", !!muted); }
    const c = $("#clock"); if (c) c.textContent = hms(t, true);
    if (!vid.paused && performance.now() - lastDraw > 120) {
      // keep the playhead in view while playing
      const [a, z] = R.view;
      if (t > z || t < a) { const w = z - a; R.view = [clamp(t - w * 0.1, 0, Math.max(0, R.len - w)), 0]; R.view[1] = R.view[0] + w; }
      draw();
    }
  }
  let raf;
  const loop = () => { tick(); raf = requestAnimationFrame(loop); };
  loop();

  // --- timeline ---
  const LANES = { g: [18, 120], surf: [146, 46], ev: [200, 16], lab: [224, 14], sur: [242, 14], tun: [260, 14], axis: [280, 18] };
  const HEIGHT = 300;
  const X = t => (t - R.view[0]) / (R.view[1] - R.view[0]) * canvas.clientWidth;
  const T = x => R.view[0] + x / canvas.clientWidth * (R.view[1] - R.view[0]);

  function surfThreshold(i) {
    const tl = R.tl, s = R.settings;
    if (!s.SURFACE || !tl.surface_learned[i]) return null;
    const show = tl.surface_show[i], ad = tl.surface_ad[i];
    if (show == null || ad == null || show < 0.3 || show - ad < 0.25) return null;
    return ad + (show - ad) * s.SURFACE_AT;
  }

  function draw() {
    lastDraw = performance.now();
    const dpr = window.devicePixelRatio || 1, W = canvas.clientWidth;
    canvas.width = W * dpr; canvas.height = HEIGHT * dpr; canvas.style.height = HEIGHT + "px";
    const g = canvas.getContext("2d");
    g.scale(dpr, dpr);
    g.clearRect(0, 0, W, HEIGHT);
    const [t0, t1] = R.view, spp = (t1 - t0) / W;
    const dim = css("--dim"), line = css("--line");
    g.font = "11px system-ui, sans-serif";

    // lane backgrounds and names
    const lane = (key, label) => { const [y, h] = LANES[key]; g.fillStyle = css("--panel2"); g.fillRect(0, y, W, h); g.fillStyle = dim; if (label) g.fillText(label, 6, y - 4); };
    lane("g", "graphics (shape match)"); lane("surf", "playing surface share"); lane("ev");
    lane("lab"); lane("sur"); lane("tun");
    g.fillStyle = dim;
    [["lab", "labels"], ["sur", "survey"], ["tun", "tuned"], ["ev", "black / silence"]].forEach(([k, n]) => {
      const [y, h] = LANES[k]; g.textAlign = "right"; g.fillText(n, W - 6, y + h - 3); g.textAlign = "left";
    });

    // spans
    const spans = (list, key, color) => {
      if (!list) return;
      const [y, h] = LANES[key]; g.fillStyle = color;
      for (const [s, e] of list) { if (e < t0 || s > t1) continue; const a = X(s), b = X(e); g.fillRect(a, y, Math.max(1.5, b - a), h); }
    };
    // shade mutes behind the graphs, faintly
    const shade = (list, color) => {
      if (!list) return; g.fillStyle = color; g.globalAlpha = 0.09;
      for (const [s, e] of list) { if (e < t0 || s > t1) continue; g.fillRect(X(s), LANES.g[0], Math.max(1, X(e) - X(s)), LANES.ev[0] + LANES.ev[1] - LANES.g[0]); }
      g.globalAlpha = 1;
    };
    shade(R.labels.ads, css("--label"));
    spans(R.labels.ads, "lab", css("--label"));
    spans(R.survey, "sur", css("--survey"));
    spans(R.sim?.breaks, "tun", css("--tuned"));
    if (R.sel >= 0 && R.labels.ads[R.sel]) {
      const [s, e] = R.labels.ads[R.sel], [y, h] = LANES.lab;
      g.strokeStyle = css("--text"); g.lineWidth = 2; g.strokeRect(X(s), y - 1, Math.max(2, X(e) - X(s)), h + 2);
    }

    if (R.tl && R.tl.length) {
      const tl = R.tl;
      // a series drawn as a min-max band per pixel when zoomed out, a line when zoomed in
      const series = (arr, color, [y, h], vmax = 1) => {
        g.strokeStyle = color; g.fillStyle = color; g.lineWidth = 1.3;
        const yv = v => y + h - clamp(v / vmax, 0, 1) * h;
        if (spp <= 1) {
          g.beginPath(); let pen = false;
          for (let i = Math.max(0, Math.floor(t0)); i <= Math.min(arr.length - 1, Math.ceil(t1)); i++) {
            const v = arr[i]; if (v == null) { pen = false; continue; }
            const x = X(i + 0.5), yy = yv(v);
            if (pen) g.lineTo(x, yy); else { g.moveTo(x, yy); pen = true; }
          }
          g.stroke();
        } else {
          for (let px = 0; px < W; px++) {
            const a = Math.max(0, Math.floor(T(px))), b = Math.min(arr.length, Math.max(a + 1, Math.floor(T(px + 1))));
            let lo = Infinity, hi = -Infinity;
            for (let i = a; i < b; i++) { const v = arr[i]; if (v == null) continue; if (v < lo) lo = v; if (v > hi) hi = v; }
            if (lo === Infinity) continue;
            g.fillRect(px, yv(hi), 1, Math.max(1, yv(lo) - yv(hi)));
          }
        }
      };
      const gl = LANES.g;
      (tl.graphics || []).forEach((gr, i) => { g.globalAlpha = i ? 0.75 : 0.9; series(gr.score, css(i ? "--g1" : "--g0"), gl); });
      g.globalAlpha = 1;
      // thresholds
      g.setLineDash([4, 4]); g.lineWidth = 1;
      [["LOGO_ON", "on"], ["LOGO_OFF", "off"]].forEach(([k, n]) => {
        const v = R.settings[k], y = gl[0] + gl[1] - v * gl[1];
        g.strokeStyle = dim; g.beginPath(); g.moveTo(0, y); g.lineTo(W, y); g.stroke();
        g.fillStyle = dim; g.fillText(`${n} ${v.toFixed(2)}`, W - 52, n === "on" ? y - 3 : y + 11);
      });
      g.setLineDash([]);
      const sl = LANES.surf;
      g.globalAlpha = 0.85; series(tl.surface, css("--surface"), sl); g.globalAlpha = 1;
      // surface threshold
      g.strokeStyle = css("--text"); g.lineWidth = 1; g.setLineDash([3, 3]); g.beginPath();
      let pen = false;
      const step = Math.max(1, Math.floor(spp));
      for (let i = Math.max(0, Math.floor(t0)); i <= Math.min(tl.length - 1, Math.ceil(t1)); i += step) {
        const v = surfThreshold(i);
        if (v == null) { pen = false; continue; }
        const x = X(i), y = sl[0] + sl[1] - v * sl[1];
        if (pen) g.lineTo(x, y); else { g.moveTo(x, y); pen = true; }
      }
      g.stroke(); g.setLineDash([]);
      // black frames / silence
      const ev = LANES.ev;
      const ticks = (arr, color, y, h) => {
        g.fillStyle = color;
        for (let px = 0; px < W; px++) {
          const a = Math.max(0, Math.floor(T(px))), b = Math.min(arr.length, Math.max(a + 1, Math.ceil(T(px + 1))));
          for (let i = a; i < b; i++) if (arr[i]) { g.fillRect(px, y, Math.max(1, 1 / spp), h); break; }
        }
      };
      ticks(tl.black, css("--black"), ev[0], ev[1] / 2);
      ticks(tl.silent, css("--silent"), ev[0] + ev[1] / 2, ev[1] / 2);
    } else {
      g.fillStyle = dim; g.textAlign = "center"; g.fillText("Survey this recording to see its timeline", W / 2, 80); g.textAlign = "left";
    }

    // axis
    const [ay] = LANES.axis;
    const steps = [1, 2, 5, 10, 15, 30, 60, 120, 300, 600, 900, 1800, 3600];
    const stepT = steps.find(s => s / spp >= 70) || 3600;
    g.fillStyle = dim; g.strokeStyle = line; g.lineWidth = 1;
    for (let t = Math.ceil(t0 / stepT) * stepT; t <= t1; t += stepT) {
      const x = X(t); g.beginPath(); g.moveTo(x, ay); g.lineTo(x, ay + 4); g.stroke();
      g.fillText(hms(t), x + 2, ay + 13);
    }

    // shift-drag selection
    if (R.drag?.mode === "label") {
      const a = X(Math.min(R.drag.t0, R.drag.t1)), b = X(Math.max(R.drag.t0, R.drag.t1));
      g.fillStyle = css("--label"); g.globalAlpha = 0.25; g.fillRect(a, 0, b - a, HEIGHT); g.globalAlpha = 1;
    }
    // hover + playhead
    if (R.hover != null) { g.strokeStyle = dim; g.lineWidth = 1; g.beginPath(); g.moveTo(X(R.hover), 0); g.lineTo(X(R.hover), HEIGHT); g.stroke(); }
    if (vid.currentSrc) {
      const x = X(now()); g.strokeStyle = css("--accent"); g.lineWidth = 2; g.beginPath(); g.moveTo(x, 0); g.lineTo(x, HEIGHT); g.stroke();
    }
  }
  R.draw = draw;

  function zoomTo(a, b) {
    const w = clamp(b - a, 20, R.len);
    a = clamp(a, 0, Math.max(0, R.len - w));
    R.view = [a, a + w]; draw();
  }
  R.zoomTo = zoomTo;
  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    const t = T(e.offsetX), f = Math.exp((e.deltaY || e.deltaX) * 0.0015);
    zoomTo(t - (t - R.view[0]) * f, t + (R.view[1] - t) * f);
  }, { passive: false });
  canvas.addEventListener("pointerdown", e => {
    canvas.setPointerCapture(e.pointerId);
    R.drag = { x: e.offsetX, view: [...R.view], moved: false, mode: e.shiftKey ? "label" : "pan", t0: T(e.offsetX), t1: T(e.offsetX) };
  });
  canvas.addEventListener("pointermove", e => {
    R.hover = T(e.offsetX);
    showTip(e);
    if (R.drag) {
      const dx = e.offsetX - R.drag.x;
      if (Math.abs(dx) > 3) R.drag.moved = true;
      if (R.drag.mode === "pan" && R.drag.moved) {
        const w = R.drag.view[1] - R.drag.view[0], dt = -dx / canvas.clientWidth * w;
        zoomTo(R.drag.view[0] + dt, R.drag.view[1] + dt); return;
      }
      if (R.drag.mode === "label") R.drag.t1 = T(e.offsetX);
    }
    draw();
  });
  canvas.addEventListener("pointerup", e => {
    const dr = R.drag; R.drag = null;
    if (!dr) return;
    if (dr.mode === "label" && dr.moved) {
      const s = Math.min(dr.t0, dr.t1), en = Math.max(dr.t0, dr.t1);
      addLabel([Math.round(s), Math.round(en)]);
    } else if (!dr.moved) {
      // click on a label selects it; elsewhere plays from there
      const t = T(e.offsetX), [ly, lh] = LANES.lab;
      if (e.offsetY >= ly && e.offsetY <= ly + lh) {
        const i = R.labels.ads.findIndex(([s, en]) => t >= s && t <= en);
        if (i >= 0) { R.sel = i; R.tab = "labels"; renderPanel(); draw(); return; }
      }
      seek(t, true);
    }
    draw();
  });
  canvas.addEventListener("pointerleave", () => { R.hover = null; tip.hidden = true; draw(); });

  function showTip(e) {
    const t = T(e.offsetX), i = Math.floor(t), tl = R.tl;
    const sheet = Math.floor(t / 600) + 1, tile = Math.floor((t % 600) / 10);
    let vals = "";
    if (tl && i >= 0 && i < tl.length) {
      const f = v => v == null ? "–" : v.toFixed(2);
      vals = (tl.graphics || []).map((gr, k) => `g${k + 1} ${f(gr.score[i])}`).join("  ") + `  surface ${f(tl.surface[i])}` +
        (surfThreshold(i) != null ? `/${surfThreshold(i).toFixed(2)}` : "");
    }
    const hasSheets = R.d.result;
    tip.innerHTML = (hasSheets ? `<div class="thumb" style="background-image:url('api/rec/${encodeURIComponent(name)}/sheet/${sheet}.jpg');background-position:-${(tile % 6) * 256}px -${Math.floor(tile / 6) * 144}px"></div>` : "") +
      `<div><b>${hms(t)}</b> ${vals}</div>`;
    tip.hidden = false;
    const W = canvas.clientWidth, tw = 270;
    tip.style.left = clamp(e.offsetX + 14, 0, W - tw) + "px";
    tip.style.top = (hasSheets ? -190 : -40) + "px";
  }

  window.addEventListener("resize", draw);

  // --- labels ---
  const saveLabels = debounce(async () => {
    try { R.labels = { ...R.labels, ...(await api(`api/rec/${encodeURIComponent(name)}/labels`, { json: R.labels })) }; } catch (e) { toast("Couldn't save labels: " + e.message); }
  }, 600);
  function labelsChanged() {
    R.labels.ads.sort((a, b) => a[0] - b[0]);
    saveLabels(); refreshScores(); renderPanel(); draw();
  }
  function addLabel(span) {
    R.labels.ads.push(span);
    labelsChanged();
    R.sel = R.labels.ads.findIndex(a => a === span);
    R.tab = "labels"; renderPanel(); draw();
  }
  R.addLabel = addLabel;

  // --- scoring (client side, for row-by-row verdicts) ---
  function coverage(list, s, e) {
    let c = 0;
    for (const [a, b] of list || []) c += Math.max(0, Math.min(b, e) - Math.max(a, s));
    return e > s ? c / (e - s) : 0;
  }
  function verdicts(breaks) {
    const ads = R.labels.ads;
    return breaks.map(([s, e]) => ({ s, e, ok: ads.length ? coverage(ads, s, e) >= 0.5 : null }));
  }
  function adRows(breaks) {
    return R.labels.ads.map(([s, e]) => {
      const cov = coverage(breaks, s, e);
      let heardStart = 0, heardEnd = 0;
      if (cov > 0) {
        const first = breaks.filter(([a, b]) => b > s && a < e).map(([a]) => a)[0];
        heardStart = Math.max(0, first - s);
        const last = breaks.filter(([a, b]) => b > s && a < e).map(([, b]) => b).pop();
        heardEnd = Math.max(0, e - last);
      }
      return { s, e, cov, heardStart, heardEnd };
    });
  }

  // --- tuning ---
  const runSim = debounce(async () => {
    if (!R.d.result) return;
    const q = new URLSearchParams(Object.entries(R.settings).map(([k, v]) => [k, v]));
    try {
      R.sim = await api(`api/rec/${encodeURIComponent(name)}/sim?${q}`);
      if (!R.simBase) R.simBase = await api(`api/rec/${encodeURIComponent(name)}/sim`);
      draw(); if (R.tab === "tune" || R.tab === "breaks") renderPanel();
    } catch (e) { toast(e.message); }
  }, 250);
  function refreshScores() { R.simBase = null; runSim(); }
  if (R.d.result) runSim();

  // --- side panel ---
  document.querySelectorAll(".tabs button").forEach(b => b.onclick = () => {
    R.tab = b.dataset.tab; try { localStorage.setItem("studio.tab", R.tab); } catch (e) { /* no storage */ }
    renderPanel();
  });

  function renderPanel() {
    document.querySelectorAll(".tabs button").forEach(b => b.classList.toggle("on", b.dataset.tab === R.tab));
    const p = $("#panel");
    if (R.tab === "breaks") p.innerHTML = breaksPanel();
    else if (R.tab === "labels") p.innerHTML = labelsPanel();
    else if (R.tab === "tune") p.innerHTML = tunePanel();
    else p.innerHTML = infoPanel();
  }
  R.renderPanel = renderPanel;

  function breaksPanel() {
    if (!R.d.result) return `<div class="empty">Not surveyed yet. ${R.d.job ? esc(R.d.job.what) + " " + esc(R.d.job.progress || "starting") : `<br><br><button class="primary" data-act="survey">Survey now</button>`}</div>`;
    const useTuned = R.hear === "tuned" && R.sim;
    const breaks = useTuned ? R.sim.breaks : R.survey;
    const v = verdicts(breaks);
    const labelled = R.labels.ads.length > 0;
    let html = `<div class="row" style="justify-content:space-between"><b>${breaks.length} mutes ${useTuned ? "with tuned settings" : "in the survey"}</b>
      ${labelled ? "" : `<button data-act="copy-labels">Use these as labels</button>`}</div>
      <table><thead><tr><th>Start</th><th>End</th><th class="num">Length</th><th></th></tr></thead><tbody>`;
    html += v.map(({ s, e, ok }, i) => `<tr class="click" data-seek="${s - 8}" data-zoom="${s},${e}">
      <td class="t">${hms(s)}</td><td class="t">${hms(e)}</td><td class="num">${dur(e - s)}</td>
      <td>${ok === null ? "" : ok ? `<span class="tag good">ad</span>` : `<span class="tag bad">game muted</span>`}</td></tr>`).join("");
    html += `</tbody></table>`;
    if (labelled) {
      const rows = adRows(breaks).filter(r => r.cov < 0.5 || r.heardStart > 1 || r.heardEnd > 1);
      html += `<h2 style="margin-top:16px">Ads not fully muted</h2>`;
      html += rows.length ? `<table><tbody>${rows.map(r => `<tr class="click" data-seek="${r.s - 8}" data-zoom="${r.s},${r.e}">
        <td class="t">${hms(r.s)}</td><td class="num">${dur(r.e - r.s)}</td>
        <td>${r.cov < 0.5 ? `<span class="tag bad">missed</span>` : [r.heardStart > 1 ? `<span class="tag warn">${Math.round(r.heardStart)}s heard at start</span>` : "", r.heardEnd > 1 ? `<span class="tag warn">${Math.round(r.heardEnd)}s heard at end</span>` : ""].join(" ")}</td></tr>`).join("")}</tbody></table>`
        : `<div class="dim">None. Every labelled ad was muted.</div>`;
    } else {
      html += `<p class="dim" style="margin-top:14px">Label where the ads really are (Labels tab) to see which mutes are right.</p>`;
    }
    return html;
  }

  function labelsPanel() {
    const ads = R.labels.ads;
    let html = `<div class="row"><button class="primary" data-act="add-label">Add ad at playhead</button>
      ${!ads.length && R.survey.length ? `<button data-act="copy-labels">Start from survey mutes</button>` : ""}</div>
      <p class="dim" style="margin:4px 0 10px;font-size:12px">Mark where each ad break really starts and ends. Select a row, then press <span class="kbd">I</span> / <span class="kbd">O</span> while watching to set its start / end to the playhead. Shift+drag on the timeline adds one.</p>`;
    html += ads.length ? `<table><tbody>${ads.map(([s, e], i) => `<tr class="label-row ${i === R.sel ? "sel" : ""}" data-label="${i}">
      <td><input type="text" data-edit="${i},0" value="${hms(s)}"></td><td><input type="text" data-edit="${i},1" value="${hms(e)}"></td>
      <td class="num">${dur(e - s)}</td>
      <td style="white-space:nowrap"><button data-play="${i}" title="play from just before">▶</button> <button data-del="${i}" title="delete">✕</button></td></tr>`).join("")}</tbody></table>`
      : `<div class="empty">No labels yet.</div>`;
    html += `<div class="row" style="margin-top:12px"><label><input type="checkbox" data-act="reviewed" ${R.labels.reviewed ? "checked" : ""}> Reviewed: count this recording when scoring settings</label></div>
      <textarea data-act="notes" rows="3" style="width:100%" placeholder="Notes (e.g. 1:43:04 promo over crowd shot)">${esc(R.labels.notes || "")}</textarea>`;
    return html;
  }

  function scoreBox(sc, base) {
    if (!sc) return "";
    const cell = (k, label, fn = x => x) => {
      const v = fn(sc[k]), b = base ? fn(base[k]) : null;
      const delta = b == null || v === b ? "" : `<span class="delta ${v < b ? "better" : "worse"}"> ${v < b ? "▼" : "▲"}${Math.abs(v - b)}</span>`;
      return `<div><b>${v}${delta}</b><span>${label}</span></div>`;
    };
    return `<div class="scorebox">${cell("ad_heard", "s of ads heard")}${cell("show_muted", "s of game muted")}
      ${cell("false_mutes", "wrong mutes", x => x.length)}${cell("missed", "ads missed", x => x.length)}
      ${cell("late", "ads started late", x => x.length)}${cell("early_end", "ads unmuted early", x => x.length)}</div>`;
  }

  function tunePanel() {
    if (!R.d.result) return `<div class="empty">Survey this recording first.</div>`;
    const T = META.tunable, D = META.defaults, s = R.settings;
    const help = {
      LOGO_ON: "Shape match above this: graphic is on screen.", LOGO_OFF: "Below this: it's gone.",
      AD_MIN: "Seconds with every graphic gone before muting.", SHOW_MIN: "Seconds a graphic must be back to unmute.",
      MERGE: "Mutes this close together become one break.", ARM: "Seconds of logo before muting is armed.",
      SNAP: "Look this far back for a black frame or silence.", SURFACE: "1 = playing surface on screen holds off a mute.",
      SURFACE_AT: "Where the surface threshold sits between ads (0) and game (1).", AD_MAX: "Give up on a mute this long.",
      SURFACE_HOLD: "Seconds the surface must stay on screen to count.",
      SPLICE: "Wait for a black frame or silence before muting: -1 learn per channel, 0 never, 1 always.",
      SPLICE_WAIT: "When waiting for one: mute anyway after this many seconds without graphics.",
    };
    let html = Object.keys(T).map(k => {
      const [lo, hi, st] = T[k];
      return `<div class="slider"><span>${k}</span><input type="range" min="${lo}" max="${hi}" step="${st}" value="${s[k]}" data-set="${k}">
        <span class="v ${s[k] !== D[k] ? "changed" : ""}">${+s[k].toFixed(2)}</span></div><div class="help">${help[k] || ""}</div>`;
    }).join("");
    const changed = Object.keys(T).filter(k => s[k] !== D[k]);
    html += `<div class="row"><button data-act="reset">Reset</button><button data-act="score-all">Score all reviewed recordings</button></div>`;
    const sc = R.sim?.score, base = R.simBase?.score;
    if (sc) html += `<h2 style="margin-top:12px">This recording ${changed.length ? "(tuned vs current settings)" : ""}</h2>${scoreBox(sc, changed.length ? base : null)}`;
    else if (R.sim) html += `<p class="dim">Tuned settings give ${R.sim.breaks.length} mutes (survey: ${R.survey.length}). Add labels to score them.</p>`;
    html += `<div id="scoreall"></div>`;
    if (changed.length) {
      html += `<h2 style="margin-top:12px">To use these settings</h2><div class="dim" style="font-size:12px">Add to the admute environment in docker-compose.yaml:</div>
        <pre class="env">${changed.map(k => `      - ${k}=${+s[k].toFixed(3)}`).join("\n")}</pre>
        <button data-act="survey-tuned">Run a full survey with these settings</button>`;
    }
    html += `<p class="dim" style="font-size:12px;margin-top:14px">Tuning replays the survey's saved scores with new settings, without decoding video. It doesn't redo the learning, so a full survey can differ slightly.</p>`;
    return html;
  }

  function infoPanel() {
    const r = R.d.result;
    let html = "";
    if (r) {
      html += `<dl class="kv"><dt>Length</dt><dd>${hms(r.length)}</dd><dt>Surveyed</dt><dd>${new Date(r.finished * 1000).toLocaleString()} (${(r.took / 60).toFixed(1)} min)</dd>
        <dt>Graphics</dt><dd>${esc(r.graphics || "none")}</dd><dt>Mutes</dt><dd>${r.breaks.length}, ${dur(r.breaks.reduce((a, [s, e]) => a + e - s, 0))} total</dd></dl>
        <h2 style="margin-top:14px">What it learned</h2><img class="logo" src="api/rec/${encodeURIComponent(name)}/logo?${r.finished}" alt="learned edges; bright = followed graphics">
        <div class="dim" style="font-size:12px">Edges that stay put; the bright parts are the graphics it follows.</div>`;
    }
    const job = R.d.job;
    html += `<h2 style="margin-top:14px">Jobs</h2>${job ? `<p>${esc(job.what)}: ${esc(job.progress || "starting")} · <a href="#/jobs">log</a></p>` : ""}
      <div class="row"><button data-act="survey" ${job ? "disabled" : ""}>${r ? "Survey again" : "Survey"}</button>
      <button data-act="survey-fresh" ${job ? "disabled" : ""} title="forget the learned graphics and learn them again">Survey, learning from scratch</button></div>
      <div class="row"><button data-act="render" ${r ? "" : "disabled"}>Render 10 min from the playhead</button></div>
      <p class="dim" style="font-size:12px">A rendered clip, with the survey's muting, goes to the review folder.</p>`;
    return html;
  }

  async function startJob(body) {
    try { await api("api/jobs", { json: { name, ...body } }); toast("Started; see Jobs"); R.d = await api("api/rec/" + encodeURIComponent(name)); info(); renderPanel(); }
    catch (e) { toast(e.message); }
  }

  const panel = $("#panel");
  panel.addEventListener("click", async e => {
    const el = e.target.closest("[data-act],[data-seek],[data-del],[data-play],[data-label]");
    if (!el) return;
    if (el.dataset.seek !== undefined) {
      const [s, en] = el.dataset.zoom.split(",").map(Number);
      zoomTo(s - 60, en + 60); seek(Number(el.dataset.seek)); return;
    }
    if (el.dataset.del !== undefined) { R.labels.ads.splice(Number(el.dataset.del), 1); R.sel = -1; labelsChanged(); return; }
    if (el.dataset.play !== undefined) { const [s, en] = R.labels.ads[Number(el.dataset.play)]; R.sel = Number(el.dataset.play); zoomTo(s - 60, en + 60); seek(s - 5); renderPanel(); return; }
    if (el.dataset.label !== undefined && e.target.tagName !== "INPUT") { R.sel = Number(el.dataset.label); renderPanel(); draw(); return; }
    const act = el.dataset.act;
    if (act === "add-label") { const t = Math.round(now()); addLabel([t, t + 30]); }
    else if (act === "copy-labels") { R.labels.ads = R.survey.map(([s, en]) => [Math.round(s), Math.round(en)]); R.tab = "labels"; labelsChanged(); }
    else if (act === "reset") { R.settings = { ...META.defaults }; renderPanel(); runSim(); draw(); }
    else if (act === "score-all") {
      const q = new URLSearchParams(Object.entries(R.settings));
      $("#scoreall").innerHTML = `<p class="dim">Scoring…</p>`;
      const rows = await api("api/score-all?" + q);
      const sum = (k, w) => rows.reduce((a, r) => a + (Array.isArray(r[w][k]) ? r[w][k].length : r[w][k]), 0);
      $("#scoreall").innerHTML = rows.length ? `<h2 style="margin-top:12px">All reviewed recordings</h2><table><thead><tr><th></th><th class="num">ads heard</th><th class="num">game muted</th><th class="num">wrong</th></tr></thead><tbody>
        ${rows.map(r => `<tr><td>${esc(r.title)}</td><td class="num">${r.now.ad_heard} → ${r.tuned.ad_heard}</td><td class="num">${r.now.show_muted} → ${r.tuned.show_muted}</td><td class="num">${r.now.false_mutes.length} → ${r.tuned.false_mutes.length}</td></tr>`).join("")}
        <tr><td><b>Total</b></td><td class="num"><b>${sum("ad_heard", "now")} → ${sum("ad_heard", "tuned")}</b></td><td class="num"><b>${sum("show_muted", "now")} → ${sum("show_muted", "tuned")}</b></td><td class="num"><b>${sum("false_mutes", "now")} → ${sum("false_mutes", "tuned")}</b></td></tr></tbody></table>
        <div class="dim" style="font-size:12px">survey result → tuned replay, in seconds</div>` : `<p class="dim">No reviewed recordings yet. Tick "Reviewed" on a recording's Labels tab.</p>`;
    }
    else if (act === "survey") startJob({ kind: "survey" });
    else if (act === "survey-fresh") startJob({ kind: "survey", fresh: true });
    else if (act === "survey-tuned") startJob({ kind: "survey", settings: R.settings });
    else if (act === "render") startJob({ kind: "render", start: Math.max(0, Math.floor(now())), length: 600 });
  });
  panel.addEventListener("input", e => {
    const k = e.target.dataset.set;
    if (k) {
      R.settings[k] = Number(e.target.value);
      const v = e.target.parentElement.querySelector(".v");
      v.textContent = +R.settings[k].toFixed(2); v.classList.toggle("changed", R.settings[k] !== META.defaults[k]);
      draw(); runSim();
    }
    if (e.target.dataset.act === "notes") { R.labels.notes = e.target.value; saveLabels(); }
  });
  panel.addEventListener("change", e => {
    const ed = e.target.dataset.edit;
    if (ed) {
      const [i, j] = ed.split(",").map(Number), t = parseHms(e.target.value);
      if (t == null) { toast("Use h:mm:ss"); return; }
      R.labels.ads[i][j] = t;
      if (R.labels.ads[i][1] <= R.labels.ads[i][0]) R.labels.ads[i][1] = R.labels.ads[i][0] + 1;
      labelsChanged();
    }
    if (e.target.dataset.act === "reviewed") { R.labels.reviewed = e.target.checked; saveLabels(); }
  });

  const onKey = e => {
    if (e.target.closest("input, textarea, select")) return;
    if (e.key === " ") { e.preventDefault(); $("#play").click(); }
    else if (e.key === "ArrowLeft" || e.key === "ArrowRight") { e.preventDefault(); seek(now() + (e.key === "ArrowLeft" ? -1 : 1) * (e.shiftKey ? 30 : 5), !vid.paused); }
    else if (e.key === "i" || e.key === "o") {
      const t = Math.round(now());
      if (R.sel < 0 || !R.labels.ads[R.sel]) { if (e.key === "i") addLabel([t, t + 30]); return; }
      const a = R.labels.ads[R.sel];
      if (e.key === "i") { a[0] = t; if (a[1] <= t) a[1] = t + 30; } else { a[1] = Math.max(a[0] + 1, t); }
      labelsChanged(); R.sel = R.labels.ads.indexOf(a); renderPanel();
    }
  };
  document.addEventListener("keydown", onKey);

  // follow a running job for this recording, then reload when it ends
  every(4000, async () => {
    if (!R.d.job) return;
    const fresh = await api("api/rec/" + encodeURIComponent(name));
    const finished = !fresh.job;
    R.d = fresh; info();
    if (finished) { R.tl = fresh.timeline; R.survey = fresh.result ? fresh.result.breaks : []; R.len = R.tl?.length || R.len; R.simBase = null; runSim(); toast("Job finished"); }
    renderPanel(); draw();
  });

  cleanup = () => { cancelAnimationFrame(raf); document.removeEventListener("keydown", onKey); window.removeEventListener("resize", draw); vid.pause(); vid.removeAttribute("src"); vid.load(); };
  renderPanel(); draw();
}

// ---------- live ----------
async function liveView() {
  main.innerHTML = `<h1>Live</h1><div class="sub">What the admute proxy is doing right now. Channels open only while someone watches.</div><div id="live"></div>`;
  const draw = async () => {
    const d = await api("api/live");
    let html = "";
    if (!d.configured) html += `<div class="card">LIVE_URL isn't set, so live status isn't available.</div>`;
    else if (d.error) html += `<div class="card">Can't reach the proxy: ${esc(d.error)}</div>`;
    else {
      const ss = d.status.sessions || [];
      html += `<div class="dim" style="margin-bottom:8px">Delay ${d.status.delay_seconds}s · channels ${esc(d.status.channels)}</div>`;
      html += ss.length ? `<div class="cards">${ss.map(s => `<div class="card">
        <div class="row" style="justify-content:space-between;margin:0 0 8px"><b>${esc(s.channel)} ${esc(s.name)}</b>${s.muted_now ? `<span class="tag bad">muted</span>` : `<span class="tag good">sound on</span>`}</div>
        <dl class="kv"><dt>Viewers</dt><dd>${s.viewers}</dd><dt>Open</dt><dd>${dur(s.open_seconds)}</dd><dt>State</dt><dd>${esc(s.state || "")}${s.armed === false ? " (not armed)" : ""}</dd>
        <dt>Mutes</dt><dd>${s.ads_muted ?? "–"}</dd><dt>Surface</dt><dd>${s.surface ? `${s.surface.share} (show ${s.surface.usual_in_show}, ads ${s.surface.usual_in_ads}${s.surface.in_use ? "" : ", not in use"})` : "–"}</dd></dl>
        ${(s.graphics || []).map(g => `<div style="margin-top:8px;font-size:12px">${esc(g.corner)} ${g.picture_learned ? "" : "(learning)"} <span class="mono">${g.score ?? "–"}</span><div class="meter"><i style="width:${(g.score || 0) * 100}%"></i></div></div>`).join("")}
        ${s.learning_minutes_left ? `<div class="dim" style="margin-top:8px">Learning: ${s.learning_minutes_left} min left</div>` : ""}
        </div>`).join("")}</div>` : `<div class="card empty">Nothing is being watched. The proxy is idle.</div>`;
    }
    html += `<h2 style="margin-top:20px">Recent events</h2><div class="card"><pre class="log" style="max-height:520px;margin:0">${esc(d.events.join("\n")) || "None yet."}</pre></div>`;
    $("#live").innerHTML = html;
  };
  await draw();
  every(3000, draw);
}

// ---------- jobs ----------
async function jobsView() {
  main.innerHTML = `<h1>Jobs</h1><div class="sub">Surveys and clip renders started here. Clips go to the review folder.</div><div id="jobs"></div>`;
  const open = new Set();
  const draw = async () => {
    const jobs = await api("api/jobs");
    $("#jobs").innerHTML = jobs.length ? jobs.map(j => `<div class="card" style="margin-bottom:10px">
      <div class="row" style="justify-content:space-between;margin:0"><div><b>${esc(j.what)}</b> · <a href="#/rec/${encodeURIComponent(j.name)}">${esc(j.title)}</a>
        <div class="dim" style="font-size:12px">started ${new Date(j.started * 1000).toLocaleTimeString()}${j.ended ? `, took ${dur(j.ended - j.started)}` : ""}</div></div>
        <div>${j.state === "running" ? `<span class="tag warn">${esc(j.progress || "running")}</span> <button data-stop="${esc(j.id)}">Stop</button>` : `<span class="tag ${j.state === "done" ? "good" : "bad"}">${j.state}</span>`}
        <button data-log="${esc(j.id)}">${open.has(j.id) ? "Hide log" : "Log"}</button></div></div>
      ${open.has(j.id) ? `<pre class="log">${esc(j.log)}</pre>` : ""}</div>`).join("")
      : `<div class="card empty">No jobs since the studio started.</div>`;
  };
  $("#jobs").addEventListener("click", async e => {
    const st = e.target.closest("[data-stop]"), lg = e.target.closest("[data-log]");
    if (st) { await api("api/jobs/stop", { json: { id: st.dataset.stop } }); draw(); }
    if (lg) { open.has(lg.dataset.log) ? open.delete(lg.dataset.log) : open.add(lg.dataset.log); draw(); }
  });
  await draw();
  every(3000, draw);
}

route();
