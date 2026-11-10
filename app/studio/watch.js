"use strict";
// Watch: the admute channels live (muted, DELAY behind), with what's on from the guide, and
// upscaling on the viewer's side: the broadcast's 720p or 1080p is resized to the screen's
// real pixels with a Lanczos-3 filter, then sharpened with AMD's RCAS (from FSR 1), in WebGL.

// ---------- upscaler ----------
const UPSCALE_VS = `#version 300 es
out vec2 uv;
void main() {
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  uv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}`;

// One direction of a Lanczos-3 enlargement: 6 source pixels per output pixel, with weights
// worked out once per size (in `wt`: row 0 holds weights 0-3, row 1 weights 4-5 and the
// first source pixel). The result is pulled towards the two nearest source pixels' range
// (anti-ringing), so hard edges like the score bug don't get halos. The video comes in top
// row first; the vertical pass turns it the right way up for the screen.
const LANCZOS_FS = `#version 300 es
precision highp float;
precision highp int;
uniform sampler2D src;
uniform highp sampler2D wt;
uniform int horiz;
uniform int n;          // source pixels along this direction
uniform int outLen;     // output pixels along this direction
uniform float antiring;
out vec4 o;
void main() {
  ivec2 here = ivec2(gl_FragCoord.xy);
  int a = horiz == 1 ? here.x : outLen - 1 - here.y;
  vec4 w0 = texelFetch(wt, ivec2(a, 0), 0), w1 = texelFetch(wt, ivec2(a, 1), 0);
  int first = int(w1.z);
  float w[6] = float[6](w0.x, w0.y, w0.z, w0.w, w1.x, w1.y);
  vec4 sum = vec4(0.0), lo = vec4(0.0), hi = vec4(0.0);
  for (int i = 0; i < 6; i++) {
    int k = clamp(first + i, 0, n - 1);
    vec4 c = texelFetch(src, horiz == 1 ? ivec2(k, here.y) : ivec2(here.x, k), 0);
    sum += c * w[i];
    if (i == 2) { lo = c; hi = c; }
    if (i == 3) { lo = min(lo, c); hi = max(hi, c); }
  }
  o = vec4(clamp(mix(sum, clamp(sum, lo, hi), antiring).rgb, 0.0, 1.0), 1.0);
}`;

// Lanczos-3 weights for enlarging n pixels to len: for each output pixel, 6 weights and the
// first source pixel, as two rows of RGBA floats.
function lanczosWeights(n, len) {
  const L = x => { x = Math.abs(x); if (x < 1e-6) return 1; if (x >= 3) return 0; const p = Math.PI * x; return 3 * Math.sin(p) * Math.sin(p / 3) / (p * p); };
  const d = new Float32Array(len * 8);
  for (let a = 0; a < len; a++) {
    const pos = (a + 0.5) * n / len - 0.5, base = Math.floor(pos), f = pos - base;
    const w = []; let sum = 0;
    for (let i = -2; i <= 3; i++) { const v = L(i - f); w.push(v); sum += v; }
    for (let i = 0; i < 4; i++) d[a * 4 + i] = w[i] / sum;
    d[len * 4 + a * 4] = w[4] / sum; d[len * 4 + a * 4 + 1] = w[5] / sum; d[len * 4 + a * 4 + 2] = base - 2;
  }
  return d;
}

// RCAS from AMD FidelityFX FSR 1: sharpens by the most each pixel can take without clipping.
const RCAS_FS = `#version 300 es
precision highp float;
uniform sampler2D src;
uniform float sharp;   // 0..1
out vec4 o;
void main() {
  ivec2 p = ivec2(gl_FragCoord.xy), m = textureSize(src, 0) - 1;
  vec3 b = texelFetch(src, clamp(p + ivec2(0, 1), ivec2(0), m), 0).rgb;
  vec3 d = texelFetch(src, clamp(p - ivec2(1, 0), ivec2(0), m), 0).rgb;
  vec3 e = texelFetch(src, p, 0).rgb;
  vec3 f = texelFetch(src, clamp(p + ivec2(1, 0), ivec2(0), m), 0).rgb;
  vec3 h = texelFetch(src, clamp(p - ivec2(0, 1), ivec2(0), m), 0).rgb;
  vec3 mn4 = min(min(b, d), min(f, h));
  vec3 mx4 = max(max(b, d), max(f, h));
  vec3 hitMin = mn4 / (4.0 * mx4 + 1e-5);
  vec3 hitMax = (1.0 - mx4) / (4.0 * mn4 - 4.0 - 1e-5);
  vec3 lobeRGB = max(-hitMin, hitMax);
  float lobe = max(-0.1875, min(max(lobeRGB.r, max(lobeRGB.g, lobeRGB.b)), 0.0)) * sharp;
  o = vec4(clamp((lobe * (b + d + f + h) + e) / (4.0 * lobe + 1.0), 0.0, 1.0), 1.0);
}`;

class Upscaler {
  constructor(canvas) {
    const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, depth: false, preserveDrawingBuffer: false, powerPreference: "high-performance" });
    if (!gl) throw new Error("WebGL 2 isn't available");
    this.gl = gl; this.canvas = canvas;
    this.src = this._tex(); this.mid = this._tex(); this.big = this._tex();
    this.wx = this._tex(); this.wy = this._tex();
    this.fbMid = gl.createFramebuffer(); this.fbBig = gl.createFramebuffer();
    this.lanczos = this._prog(LANCZOS_FS);
    this.rcas = this._prog(RCAS_FS);
    this.vao = gl.createVertexArray();
    this.sizes = "";
  }
  // Worth doing only when the picture shows bigger than the video's own size.
  static wanted(sw, sh, dw, dh) { return sw > 0 && dw > sw * 1.02 && dh > sh * 1.02; }
  _tex() {
    const gl = this.gl, t = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, t);
    for (const k of [gl.TEXTURE_MIN_FILTER, gl.TEXTURE_MAG_FILTER]) gl.texParameteri(gl.TEXTURE_2D, k, gl.NEAREST);
    for (const k of [gl.TEXTURE_WRAP_S, gl.TEXTURE_WRAP_T]) gl.texParameteri(gl.TEXTURE_2D, k, gl.CLAMP_TO_EDGE);
    return t;
  }
  _prog(fs) {
    const gl = this.gl, p = gl.createProgram();
    for (const [type, text] of [[gl.VERTEX_SHADER, UPSCALE_VS], [gl.FRAGMENT_SHADER, fs]]) {
      const s = gl.createShader(type);
      gl.shaderSource(s, text); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      gl.attachShader(p, s);
    }
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const u = {};
    p.u = name => name in u ? u[name] : (u[name] = gl.getUniformLocation(p, name));
    return p;
  }
  _target(tex, fb, w, h) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, w, h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  }
  _resize(sw, sh, dw, dh) {
    const gl = this.gl;
    for (const k of ["src", "mid", "big", "wx", "wy"]) { gl.deleteTexture(this[k]); this[k] = this._tex(); }
    gl.bindTexture(gl.TEXTURE_2D, this.src);
    gl.texStorage2D(gl.TEXTURE_2D, 1, gl.RGBA8, sw, sh);
    this._target(this.mid, this.fbMid, dw, sh);
    this._target(this.big, this.fbBig, dw, dh);
    for (const [t, n, len] of [[this.wx, sw, dw], [this.wy, sh, dh]]) {
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, len, 2, 0, gl.RGBA, gl.FLOAT, lanczosWeights(n, len));
    }
  }
  _pass(prog, input, fb, w, h, set) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fb);
    gl.viewport(0, 0, w, h);
    gl.useProgram(prog);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, input);
    gl.uniform1i(prog.u("src"), 0);
    if (set) set(prog);
    gl.bindVertexArray(this.vao);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }
  // Draws the video's current picture at the canvas's size.
  draw(video, sharp) {
    const gl = this.gl, sw = video.videoWidth, sh = video.videoHeight;
    const dw = this.canvas.width, dh = this.canvas.height;
    if (!sw || !sh || !dw || !dh) return;
    const sizes = `${sw}x${sh}>${dw}x${dh}`;
    if (sizes !== this.sizes) { this._resize(sw, sh, dw, dh); this.sizes = sizes; }
    gl.bindTexture(gl.TEXTURE_2D, this.src);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, sw, sh, gl.RGBA, gl.UNSIGNED_BYTE, video);
    const pass = (wt, horiz, n, outLen) => p => {
      gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, wt); gl.uniform1i(p.u("wt"), 1);
      gl.uniform1i(p.u("horiz"), horiz); gl.uniform1i(p.u("n"), n); gl.uniform1i(p.u("outLen"), outLen);
      gl.uniform1f(p.u("antiring"), 0.8);
    };
    this._pass(this.lanczos, this.src, this.fbMid, dw, sh, pass(this.wx, 1, sw, dw));
    if (sharp > 0) {
      this._pass(this.lanczos, this.mid, this.fbBig, dw, dh, pass(this.wy, 0, sh, dh));
      this._pass(this.rcas, this.big, null, dw, dh, p => gl.uniform1f(p.u("sharp"), sharp));
    } else {
      this._pass(this.lanczos, this.mid, null, dw, dh, pass(this.wy, 0, sh, dh));
    }
  }
}

// ---------- the Watch page ----------
const WATCH_PREFS = (() => { try { return JSON.parse(localStorage.getItem("watch") || "{}"); } catch (e) { return {}; } })();
function saveWatchPrefs() { try { localStorage.setItem("watch", JSON.stringify(WATCH_PREFS)); } catch (e) { /* private mode */ } }
const clock = t => new Date(t * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });

async function watchView(arg) {
  main.innerHTML = `<div class="watch">
    <div class="tv-col">
      <div class="tv" id="tv" tabindex="0">
        <video id="wv" playsinline></video>
        <canvas id="wc"></canvas>
        <div class="tv-msg" id="wmsg">Pick a channel.</div>
        <div class="tv-top" id="wtop"></div>
        <div class="tv-bar">
          <button class="ghost" id="wplay" title="Play / pause (space)">▶</button>
          <button class="ghost" id="wmute" title="Sound (m)">🔊</button>
          <input type="range" id="wvol" min="0" max="1" step="0.05" title="Volume">
          <span class="grow"></span>
          <span class="dim" id="wstat"></span>
          <select id="wup" title="Upscaling">
            <option value="sharp">Upscale + sharpen</option>
            <option value="smooth">Upscale</option>
            <option value="off">Off (browser)</option>
          </select>
          <input type="range" id="wsharp" min="0" max="1" step="0.05" title="Sharpening">
          <button class="ghost" id="wfs" title="Full screen (f)">⛶</button>
        </div>
      </div>
      <div class="tv-info" id="winfo"></div>
    </div>
    <div class="side">
      <div class="guide card" id="wguide"><div class="empty">Loading channels…</div></div>
      <div class="card recaps" id="wrecaps" hidden></div>
    </div>
  </div>`;
  const tv = $("#tv"), vid = $("#wv"), canvas = $("#wc"), msg = $("#wmsg");
  const P = Object.assign({ mode: "sharp", sharp: 0.5, volume: 1, muted: false }, WATCH_PREFS);
  let chans = [], cur = null, up = null, alive = true, frameCb = null, idle, hls = null;
  let recapList = [], rc = null;   // rc: the recap playing ({name, level, clips, i})

  try { up = new Upscaler(canvas); } catch (e) {
    up = null; P.mode = "off";
    $("#wup").disabled = true; $("#wup").title = "Upscaling needs WebGL 2: " + e.message;
  }
  $("#wup").value = P.mode; $("#wsharp").value = P.sharp; $("#wvol").value = P.volume;
  vid.volume = P.volume; vid.muted = P.muted;

  const show = m => { msg.textContent = m || ""; msg.hidden = !m; };
  const upscaling = () => up && P.mode !== "off" && Upscaler.wanted(vid.videoWidth, vid.videoHeight, canvas.width, canvas.height);

  // canvas at the real pixel size of where the picture shows (letterboxed inside the box)
  function layout() {
    const box = tv.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    const ar = vid.videoWidth && vid.videoHeight ? vid.videoWidth / vid.videoHeight : 16 / 9;
    let w = box.width, h = w / ar;
    if (h > box.height) { h = box.height; w = h * ar; }
    Object.assign(canvas.style, { width: w + "px", height: h + "px", left: (box.width - w) / 2 + "px", top: (box.height - h) / 2 + "px" });
    canvas.width = Math.max(1, Math.round(w * dpr)); canvas.height = Math.max(1, Math.round(h * dpr));
    canvas.hidden = !upscaling();   // the video stays under it: hidden videos don't report new frames
    stat();
    paint();
  }
  // What's playing and how well: size, bitrate, buffer, frames the browser dropped, and
  // frames the upscaler missed (it draws once per video frame; a skipped one means too slow).
  const perf = { seen: 0, missed: 0, last: null, since: performance.now(), note: "" };
  function stat() {
    const w = vid.videoWidth, h = vid.videoHeight;
    if (!w) { $("#wstat").textContent = ""; return; }
    const parts = [upscaling() ? `${w}×${h} → ${canvas.width}×${canvas.height}` : `${w}×${h}`];
    const lv = hls && hls.levels[hls.currentLevel];
    if (lv) parts.push(`${(lv.bitrate / 1e6).toFixed(0)} Mb/s`);
    const b = vid.buffered;
    if (b.length) parts.push(`${Math.max(0, b.end(b.length - 1) - vid.currentTime).toFixed(1)}s buffered`);
    const q = vid.getVideoPlaybackQuality ? vid.getVideoPlaybackQuality() : null;
    if (q && q.droppedVideoFrames) parts.push(`${q.droppedVideoFrames} dropped`);
    if (perf.note) parts.push(perf.note);
    $("#wstat").textContent = parts.join(" · ");
    $("#wsharp").hidden = !upscaling() || P.mode !== "sharp";
  }
  function paint() {
    if (upscaling() && vid.readyState >= 2) {
      try { up.draw(vid, P.mode === "sharp" ? +P.sharp : 0); } catch (e) { console.warn(e); }
    }
  }
  // If the upscaler can't keep up on this device, step down (sharpen → plain → off) for now.
  function checkPerf() {
    if (perf.seen < 120 || !upscaling()) return;
    const share = perf.missed / perf.seen;
    if (share > 0.08) {
      const next = P.mode === "sharp" ? "smooth" : "off";
      perf.note = next === "off" ? "upscaling off: too slow here" : "sharpening off: too slow here";
      P.mode = next; $("#wup").value = next; layout();
    }
    perf.seen = perf.missed = 0;
  }
  // draw each new video frame
  function loop(now, meta) {
    if (!alive) return;
    paint();
    if (meta && upscaling()) {
      if (perf.last != null && meta.presentedFrames > perf.last) {
        perf.seen += meta.presentedFrames - perf.last;
        perf.missed += meta.presentedFrames - perf.last - 1;
      }
      perf.last = meta.presentedFrames;
    } else perf.last = null;
    frameCb = vid.requestVideoFrameCallback ? vid.requestVideoFrameCallback(loop) : requestAnimationFrame(loop);
  }
  every(2000, () => { stat(); checkPerf(); });
  const ro = new ResizeObserver(layout); ro.observe(tv);
  vid.addEventListener("loadedmetadata", layout);
  vid.addEventListener("resize", layout);
  vid.addEventListener("playing", () => { show(""); $("#wplay").textContent = "❚❚"; });
  vid.addEventListener("pause", () => { $("#wplay").textContent = "▶"; });
  vid.addEventListener("waiting", () => (cur || rc) && show("Buffering…"));
  vid.addEventListener("error", () => {
    if (rc) return nextClip();
    if (cur) show("The channel stopped. Pick it again to retry (maybe no free tuner).");
  });
  vid.addEventListener("ended", () => rc && nextClip());
  loop();

  // Safari and iPhones play HLS themselves; elsewhere hls.js does it (loaded on first use).
  const NATIVE = !!vid.canPlayType("application/vnd.apple.mpegurl") && /Apple/.test(navigator.vendor || "");
  function hlsJs() {
    if (window.Hls) return Promise.resolve(window.Hls);
    return new Promise((ok, fail) => {
      const el = document.createElement("script");
      el.src = "static/hls.min.js"; el.onload = () => ok(window.Hls); el.onerror = () => fail(new Error("couldn't load the player"));
      document.head.appendChild(el);
    });
  }
  async function tune(num) {
    rc = null; vid.loop = false;
    cur = chans.find(c => c.num === num) || { num, name: num };
    location.replace(`#/watch/${encodeURIComponent(num)}`);
    show(`Tuning ${cur.num} ${cur.name}… (the picture starts after the ${DELAY_S}s delay)`);
    drawGuide(); drawInfo();
    const url = `api/watch/${encodeURIComponent(num)}/live.m3u8`;
    if (hls) { hls.destroy(); hls = null; }
    if (NATIVE) {
      vid.src = url;
    } else {
      const Hls = await hlsJs().catch(e => { show(e.message); return null; });
      if (!Hls || !alive || !cur || cur.num !== num) return;
      if (!Hls.isSupported()) { vid.src = url; }
      else {
        // About 3 pieces (6s) behind the newest: enough buffer to ride out a slow moment.
        hls = new Hls({ liveSyncDurationCount: 3, liveMaxLatencyDurationCount: 10, maxBufferLength: 20,
                        backBufferLength: 10, abrEwmaDefaultEstimate: 20e6, manifestLoadingTimeOut: 40000,
                        manifestLoadingMaxRetry: 2 });
        let retries = 0;
        hls.on(Hls.Events.ERROR, (_, d) => {
          if (!d.fatal) return;
          if (d.type === Hls.ErrorTypes.MEDIA_ERROR) hls.recoverMediaError();
          else if (d.type === Hls.ErrorTypes.NETWORK_ERROR && retries++ < 3) hls.startLoad();
          else show("The channel stopped. Pick it again to retry (maybe no free tuner).");
        });
        hls.on(Hls.Events.LEVEL_SWITCHED, stat);
        hls.loadSource(url);
        hls.attachMedia(vid);
      }
    }
    vid.play().catch(() => show("Press play to start."));
  }
  let DELAY_S = 6;

  function prog(p, now) {
    if (!p) return "";
    const share = clamp((now - p.start) / (p.stop - p.start), 0, 1);
    return `<div class="meter"><i style="width:${share * 100}%"></i></div>`;
  }
  function drawInfo() {
    const c = cur && chans.find(x => x.num === cur.num) || cur, now = Date.now() / 1000;
    if (rc) return drawRecapInfo();
    if (!c) { $("#wtop").innerHTML = ""; $("#winfo").innerHTML = ""; return; }
    const n = c.now;
    $("#wtop").innerHTML = `<div class="tv-chan">${c.logo ? `<img src="${esc(c.logo)}" alt="" referrerpolicy="no-referrer">` : ""}
      <div><b>${esc(c.num)} ${esc(c.name)}</b>${c.muted_now ? ` <span class="tag bad">ad muted</span>` : ""}
      <div>${esc(n ? n.title : "")}${n && n.subtitle ? ` <span class="dim">· ${esc(n.subtitle)}</span>` : ""}</div></div></div>`;
    $("#winfo").innerHTML = n ? `<div class="row" style="justify-content:space-between;margin:0">
        <div><b>${esc(n.title)}</b>${n.subtitle ? ` · ${esc(n.subtitle)}` : ""}</div>
        <span class="dim mono">${clock(n.start)} – ${clock(n.stop)}</span></div>
        ${prog(n, now)}
        ${n.desc ? `<p class="dim" style="margin:8px 0 0">${esc(n.desc)}</p>` : ""}
        ${c.next ? `<div class="dim" style="margin-top:8px">Next at ${clock(c.next.start)}: ${esc(c.next.title)}${c.next.subtitle ? ` · ${esc(c.next.subtitle)}` : ""}</div>` : ""}`
      : `<span class="dim">No guide information for ${esc(c.num)} ${esc(c.name)}.</span>`;
    const r = recapFor(c);
    if (r) $("#winfo").insertAdjacentHTML("afterbegin", `<div class="catchup">
        <span><b>Catch up</b> <span class="dim">· ${r.live ? "recap so far" : "recap"}: ${recapSummary(r)}</span></span>
        <span class="row" style="margin:0">${recapButtons(r)}</span></div>`);
  }
  function drawGuide() {
    const now = Date.now() / 1000;
    $("#wguide").innerHTML = chans.length ? chans.map(c => `<button class="ch${cur && cur.num === c.num ? " on" : ""}" data-num="${esc(c.num)}">
        <span class="ch-logo">${c.logo ? `<img src="${esc(c.logo)}" alt="" loading="lazy" referrerpolicy="no-referrer">` : ""}</span>
        <span class="ch-main"><span class="ch-name"><b>${esc(c.num)}</b> ${esc(c.name)}${c.open ? ` <span class="tag">${c.muted_now ? "muted" : "open"}</span>` : ""}</span>
        <span class="ch-now">${esc(c.now ? c.now.title : "–")}</span>${prog(c.now, now)}
        ${c.next ? `<span class="ch-next dim">${clock(c.next.start)} ${esc(c.next.title)}</span>` : ""}</span>
      </button>`).join("") : `<div class="empty">No channels (is the proxy running?)</div>`;
  }
  $("#wguide").addEventListener("click", e => { const b = e.target.closest("[data-num]"); if (b) tune(b.dataset.num); });

  // ---------- recaps ----------
  const teamsOf = t => { const m = /^(.+?) (?:at|vs\.?) (.+)$/.exec((t || "").trim()); return m ? [m[1].toLowerCase(), m[2].toLowerCase()] : null; };
  function recapFor(c) {
    const t = c && c.now && teamsOf(c.now.subtitle);
    return t && recapList.find(r => r.teams && r.teams[0].toLowerCase() === t[0] && r.teams[1].toLowerCase() === t[1]);
  }
  const pick = (r, level) => r.clips.filter(x => x.level <= level);
  const mmss = s => `${Math.floor(s / 60)}:${String(Math.round(s % 60)).padStart(2, "0")}`;
  const runtime = clips => mmss(clips.reduce((a, x) => a + x.seconds, 0));
  function recapSummary(r) { const all = pick(r, 2); return `${all.length} plays, ${runtime(all)}`; }
  function recapButtons(r) {
    const key = pick(r, 1), all = pick(r, 2);
    return `${key.length ? `<button class="ghost" data-recap="${esc(r.name)}" data-level="1" title="Scores, turnovers and missed kicks">Scores &amp; turnovers · ${runtime(key)}</button>` : ""}
      <button data-recap="${esc(r.name)}" data-level="2" title="Every key play">All highlights · ${runtime(all)}</button>`;
  }
  function drawRecaps() {
    const box = $("#wrecaps");
    box.hidden = !recapList.length;
    box.innerHTML = `<h3>Game recaps</h3>` + recapList.map(r => `<div class="recap-row">
        <div><b>${esc(r.teams ? `${r.teams[0]} at ${r.teams[1]}` : r.title)}</b>${r.live ? ` <span class="tag">live</span>` : ""}
          <div class="dim">${recapSummary(r)}</div></div>
        <div class="row" style="margin:4px 0 0">${recapButtons(r)}</div></div>`).join("");
  }
  function startRecap(name, level) {
    const r = recapList.find(x => x.name === name);
    if (!r) return;
    const clips = pick(r, level);
    if (!clips.length) return;
    if (hls) { hls.destroy(); hls = null; }
    rc = { name, level, clips, i: -1, from: cur && cur.num };
    nextClip();
  }
  function playClip() {
    const c = rc.clips[rc.i];
    show("");
    vid.src = `api/recap/${encodeURIComponent(rc.name)}/${encodeURIComponent(c.file)}`;
    vid.play().catch(() => show("Press play to start."));
    drawInfo();
  }
  async function nextClip(step = 1) {
    if (!rc) return;
    // a live game's recap grows: take in plays added since it started
    const r = recapList.find(x => x.name === rc.name);
    if (r) {
      const have = new Set(rc.clips.map(x => x.id));
      for (const x of pick(r, rc.level)) if (!have.has(x.id)) rc.clips.push(x);
    }
    rc.i += step;
    if (rc.i < 0) rc.i = 0;
    if (rc.i >= rc.clips.length) return endRecap();
    playClip();
  }
  function endRecap() {
    const back = rc && rc.from;
    rc = null;
    if (back) tune(back);
    else { vid.pause(); vid.removeAttribute("src"); vid.load(); show("Recap finished. Pick a channel."); drawInfo(); }
  }
  function drawRecapInfo() {
    const r = recapList.find(x => x.name === rc.name), c = rc.clips[rc.i];
    if (!c) return;
    const title = r && r.teams ? `${r.teams[0]} at ${r.teams[1]}` : rc.name;
    $("#wtop").innerHTML = `<div class="tv-chan"><div><b>Recap</b> <span class="tag">${rc.i + 1} / ${rc.clips.length}</span>
        <div>${esc(title)} <span class="dim">· ${esc(c.when)} · ${esc(c.score)}</span></div></div></div>`;
    $("#winfo").innerHTML = `<div class="catchup"><span><b>${esc(c.text)}</b></span>
        <span class="row" style="margin:0">
          <button class="ghost" id="rprev" title="Previous play (←)">◀ Previous</button>
          <button class="ghost" id="rnext" title="Next play (→)">Next ▶</button>
          <button id="rlive">${rc.from ? "Back to live" : "Close recap"}</button></span></div>
      <div class="recap-list">${rc.clips.map((x, i) => `<button class="ghost${i === rc.i ? " on" : ""}" data-clip="${i}">
        <span class="mono dim">${esc(x.when)}</span> ${esc(x.text)}</button>`).join("")}</div>`;
    $("#rprev").onclick = () => nextClip(-1);
    $("#rnext").onclick = () => nextClip(1);
    $("#rlive").onclick = endRecap;
  }
  main.querySelector(".watch").addEventListener("click", e => {
    const b = e.target.closest("[data-recap]");
    if (b) return startRecap(b.dataset.recap, +b.dataset.level);
    const k = e.target.closest("[data-clip]");
    if (k && rc) { rc.i = +k.dataset.clip; playClip(); }
  });
  async function refreshRecaps() {
    try {
      const d = await api("api/recaps");
      if (!alive || !Array.isArray(d)) return;
      recapList = d;
      drawRecaps();
      if (!rc) drawInfo();
    } catch (e) { /* next time */ }
  }

  async function refresh() {
    try {
      const d = await api("api/channels");
      if (!alive) return;
      if (d.error) $("#wguide").innerHTML = `<div class="empty">Can't reach the proxy: ${esc(d.error)}</div>`;
      chans = d.channels || []; DELAY_S = d.delay_seconds ?? DELAY_S;
      if (!d.error) drawGuide();
      drawInfo();
    } catch (e) { /* try again next time */ }
  }
  await refresh();
  refreshRecaps();
  every(15000, refresh);
  every(20000, refreshRecaps);
  if (arg) tune(decodeURIComponent(arg));

  // controls
  $("#wplay").onclick = () => vid.paused ? (cur ? vid.play() : null) : vid.pause();
  $("#wmute").onclick = () => { vid.muted = !vid.muted; };
  vid.addEventListener("volumechange", () => {
    $("#wmute").textContent = vid.muted || !vid.volume ? "🔇" : "🔊";
    P.volume = vid.volume; P.muted = vid.muted; saveWatchPrefs();
  });
  vid.dispatchEvent(new Event("volumechange"));
  $("#wvol").oninput = e => { vid.volume = +e.target.value; vid.muted = false; };
  $("#wup").onchange = e => { perf.note = ""; P.mode = e.target.value; Object.assign(WATCH_PREFS, P); saveWatchPrefs(); layout(); };
  $("#wsharp").oninput = e => { P.sharp = +e.target.value; Object.assign(WATCH_PREFS, P); saveWatchPrefs(); paint(); };
  const fullscreen = () => {
    if (document.fullscreenElement) return document.exitFullscreen();
    if (tv.requestFullscreen) return tv.requestFullscreen();
    if (vid.webkitEnterFullscreen) vid.webkitEnterFullscreen();   // iPhone: the browser's own player
  };
  $("#wfs").onclick = fullscreen;
  tv.addEventListener("dblclick", fullscreen);
  const wake = () => { tv.classList.remove("idle"); clearTimeout(idle); idle = setTimeout(() => tv.classList.add("idle"), 3000); };
  tv.addEventListener("mousemove", wake); tv.addEventListener("touchstart", wake, { passive: true }); wake();
  const keys = e => {
    if (e.target.closest("input, select, textarea")) return;
    const i = chans.findIndex(c => cur && c.num === cur.num);
    if (e.key === " ") { e.preventDefault(); $("#wplay").click(); }
    else if (e.key === "f") fullscreen();
    else if (e.key === "m") $("#wmute").click();
    else if (rc && (e.key === "ArrowRight" || e.key === "ArrowLeft")) { e.preventDefault(); nextClip(e.key === "ArrowRight" ? 1 : -1); }
    else if ((e.key === "ArrowUp" || e.key === "ArrowDown") && chans.length) {
      e.preventDefault();
      tune(chans[(Math.max(i, 0) + (e.key === "ArrowUp" ? -1 : 1) + chans.length) % chans.length].num);
    }
  };
  document.addEventListener("keydown", keys);

  cleanup = () => {
    alive = false;
    ro.disconnect();
    document.removeEventListener("keydown", keys);
    if (frameCb != null && vid.cancelVideoFrameCallback) vid.cancelVideoFrameCallback(frameCb);
    if (hls) hls.destroy();
    vid.pause(); vid.removeAttribute("src"); vid.load();   // lets the tuner go
  };
}
