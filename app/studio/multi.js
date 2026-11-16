"use strict";
// Multi-view: one channel per tuner at once. The main channel is big on the left with the
// sound. When it goes to a commercial break, the sound and the big spot move to a side
// channel that isn't in one, and go back to the main channel as soon as its break ends.
// Side channels in a break are hidden until it ends.
//
// Breaks come from the proxy (api/breaks) as clock times when its viewers reach them; the
// studio stamps the same clock on each piece of the stream (EXT-X-PROGRAM-DATE-TIME), so
// each tile knows whether what it's showing right now is inside a break.

const BREAK_LONG = 15;   // seconds: shorter mutes (and sponsor lines) don't move the sound

async function multiView(arg) {
  main.innerHTML = `<div class="multi">
    <div class="mv-bar">
      <span id="mvslots" class="row" style="margin:0"></span>
      <span class="grow"></span>
      <span class="dim" id="mvnote"></span>
      <button class="ghost" id="mvmute" title="Sound (m)">🔊</button>
      <input type="range" id="mvvol" min="0" max="1" step="0.05" title="Volume">
      <button class="ghost" id="mvcap" title="Captions (c)">CC</button>
      <button class="ghost" id="mvfs" title="Full screen (f)">⛶</button>
    </div>
    <div class="mv-grid" id="mvgrid"></div>
    <p class="dim mv-help">The main channel has the sound. During its commercial breaks the sound and the big
      spot go to a side channel that isn't in a break, and come back when the main channel's game does.
      Side channels in a break are hidden. Click a side channel to make it the main one.</p>
  </div>`;
  const grid = $("#mvgrid");
  const P = Object.assign({ volume: 1, muted: false, cc: false }, WATCH_PREFS);
  let chans = [], tuners = null, alive = true, focus = null, soundBlocked = false;
  const NATIVE = !!document.createElement("video").canPlayType("application/vnd.apple.mpegurl") && /Apple/.test(navigator.vendor || "");

  const saved = (arg ? decodeURIComponent(arg) : (WATCH_PREFS.multi || "")).split(",").filter(Boolean);
  let tiles = [];

  function makeTile(i) {
    const el = document.createElement("div");
    el.className = "mv-tile";
    el.innerHTML = `<video playsinline muted></video>
      <div class="mv-label"></div><div class="tv-msg"></div>`;
    grid.appendChild(el);
    const t = { i, el, video: $("video", el), msg: $(".tv-msg", el), num: null, hls: null, breaks: [], known: null, failed: false };
    t.video.textTracks.addEventListener("addtrack", () => applySound());
    t.video.addEventListener("playing", () => { t.failed = false; say(t, ""); });
    t.video.addEventListener("waiting", () => t.num && say(t, "Buffering…"));
    t.video.addEventListener("error", () => t.num && fail(t));
    el.addEventListener("click", e => {
      if (e.target.closest("button")) return;
      if (soundBlocked) { soundBlocked = false; applySound(); return; }
      if (t.i !== 0 && t.num) swap(0, t.i);
    });
    el.addEventListener("dblclick", fullscreen);
    return t;
  }
  const say = (t, m) => { t.msg.textContent = m || ""; t.msg.hidden = !m; };
  function fail(t) {
    t.failed = true;
    say(t, `${t.num} stopped (maybe no free tuner). Pick it again to retry.`);
  }

  const keep = () => tiles.map(t => t.num).filter(Boolean).join(",");
  async function tune(t, num) {
    if (t.hls) { t.hls.destroy(); t.hls = null; }
    t.num = num || null; t.breaks = []; t.known = null; t.failed = false;
    t.video.pause(); t.video.removeAttribute("src"); t.video.load();
    if (!t.num) { say(t, ""); return; }
    const c = chans.find(x => x.num === num);
    say(t, `Tuning ${num}${c ? " " + c.name : ""}…`);
    const url = `api/watch/${encodeURIComponent(num)}/live.m3u8?keep=${encodeURIComponent(keep())}`;
    t.video.muted = true;
    if (NATIVE) t.video.src = url;
    else {
      const Hls = await loadHls().catch(e => { say(t, e.message); return null; });
      if (!Hls || !alive || t.num !== num) return;
      if (!Hls.isSupported()) t.video.src = url;
      else {
        // capLevelToPlayerSize: small tiles take the 720p version
        const h = t.hls = new Hls({ liveSyncDurationCount: 3, liveMaxLatencyDurationCount: 10, maxBufferLength: 20,
                                    backBufferLength: 10, capLevelToPlayerSize: true, manifestLoadingTimeOut: 40000,
                                    manifestLoadingMaxRetry: 2 });
        let retries = 0;
        h.on(Hls.Events.ERROR, (_, d) => {
          if (!d.fatal) return;
          if (d.type === Hls.ErrorTypes.MEDIA_ERROR) h.recoverMediaError();
          else if (d.type === Hls.ErrorTypes.NETWORK_ERROR && retries++ < 3) h.startLoad();
          else fail(t);
        });
        h.loadSource(url);
        h.attachMedia(t.video);
      }
    }
    t.video.play().catch(() => {});
    applySound();
  }
  function loadHls() {
    if (window.Hls) return Promise.resolve(window.Hls);
    return new Promise((ok, no) => {
      const el = document.createElement("script");
      el.src = "static/hls.min.js"; el.onload = () => ok(window.Hls); el.onerror = () => no(new Error("couldn't load the player"));
      document.head.appendChild(el);
    });
  }

  // ---------- breaks and focus ----------
  // The clock time of what a tile shows now (from the stream's stamps), or null.
  function playhead(t) {
    if (t.hls) { const d = t.hls.playingDate; return d ? d.getTime() / 1000 : null; }
    const v = t.video;
    if (v.getStartDate) { const d = v.getStartDate().getTime(); if (d) return d / 1000 + v.currentTime; }
    return null;
  }
  // The long break a tile is in now, or null.
  function inBreak(t) {
    const p = t.num && t.known != null ? playhead(t) : null;
    if (p == null) return null;
    return t.breaks.find(([s, e]) => p >= s && (e == null || p < e) && (e ?? t.known) - s >= BREAK_LONG) || null;
  }
  const watchable = t => t.num && !t.failed && !inBreak(t);

  function decide() {
    const m = tiles[0], sides = tiles.slice(1).filter(t => t.num);
    let f;
    if (m.num && !m.failed && !inBreak(m)) f = m;   // main is back: it takes over, whatever the sides do
    else if (focus && focus !== m && watchable(focus)) f = focus;
    else f = sides.find(watchable) || (m.num ? m : sides[0] || m);
    const changed = f !== focus;
    focus = f;
    layoutTiles();
    if (changed) applySound();
    drawLabels();
  }
  function layoutTiles() {
    const shown = tiles.filter(t => t === focus || (t.num && (t === tiles[0] || !inBreak(t))));
    const others = shown.filter(t => t !== focus);
    grid.classList.toggle("solo", !others.length);
    for (const t of tiles) {
      const on = shown.includes(t);
      t.el.hidden = !on;   // hidden ones keep playing, so they're ready to come back
      t.el.classList.toggle("off", !t.num);
      t.el.classList.toggle("focus", t === focus);
      if (t === focus) { t.el.style.gridRow = `1 / span ${Math.max(1, others.length)}`; t.el.style.order = 0; }
      else { t.el.style.gridRow = ""; t.el.style.order = 1 + t.i; }
    }
    const hidden = tiles.filter(t => t.num && !shown.includes(t)).map(t => t.num);
    $("#mvnote").textContent = hidden.length ? `${hidden.join(", ")} hidden during a break` : "";
  }
  function applySound() {
    for (const t of tiles) {
      const v = t.video, on = t === focus && !soundBlocked;
      v.volume = P.volume;
      v.muted = !on || P.muted;
      for (const tr of v.textTracks) {
        if (tr.kind === "captions" || tr.kind === "subtitles") tr.mode = t === focus && P.cc ? "showing" : "disabled";
      }
      if (on && !P.muted && v.paused && t.num) v.play().catch(() => {});
      // Without a click on the page first, browsers stop a video that's unmuted: ask for one.
      if (on && !P.muted) setTimeout(() => {
        if (alive && v.paused && t.num && !v.muted) { v.muted = true; v.play().catch(() => {}); soundBlocked = true; drawLabels(); }
      }, 300);
    }
    $("#mvmute").textContent = P.muted || !P.volume ? "🔇" : "🔊";
    $("#mvcap").classList.toggle("on", !!P.cc);
  }
  function drawLabels() {
    for (const t of tiles) {
      const c = t.num && chans.find(x => x.num === t.num), b = inBreak(t);
      const lab = $(".mv-label", t.el);
      if (!t.num) { lab.innerHTML = ""; say(t, t.i === 0 ? "Pick a main channel above." : ""); continue; }
      const left = b && b[1] != null ? ` ${Math.max(0, Math.round(b[1] - playhead(t)))}s` : "";
      lab.innerHTML = `<b>${esc(t.num)} ${esc(c ? c.name : "")}</b>
        ${t.i === 0 ? `<span class="tag">main</span>` : ""}
        ${t === focus && !P.muted ? `<span class="tag good">${soundBlocked ? "click for sound" : "sound"}</span>` : ""}
        ${b ? `<span class="tag bad">break${left}</span>` : ""}
        <div class="mv-now">${esc(c && c.now ? c.now.title : "")}${c && c.now && c.now.subtitle ? ` · ${esc(c.now.subtitle)}` : ""}</div>`;
    }
  }

  // ---------- choosing channels ----------
  let slotsKey = "";
  function drawSlots() {
    const key = JSON.stringify([tiles.map(t => t.num), chans.map(c => [c.num, c.name, c.now && c.now.title])]);
    if (key === slotsKey) return;   // redrawing would close a list that's open
    slotsKey = key;
    const opts = sel => `<option value="">${sel === 0 ? "Main: pick…" : "— none —"}</option>` +
      chans.map(c => `<option value="${esc(c.num)}">${esc(c.num)} ${esc(c.name)}${c.now ? " · " + esc(c.now.title) : ""}</option>`).join("");
    $("#mvslots").innerHTML = tiles.map(t => `<label class="mv-slot"><span class="dim">${t.i === 0 ? "Main" : "Side"}</span>
      <select data-slot="${t.i}">${opts(t.i)}</select></label>`).join("");
    for (const t of tiles) $(`select[data-slot="${t.i}"]`).value = t.num || "";
  }
  $("#mvslots").addEventListener("change", e => {
    const s = e.target.closest("select[data-slot]");
    if (!s) return;
    const t = tiles[+s.dataset.slot], num = s.value || null;
    const other = num && tiles.find(x => x !== t && x.num === num);
    if (other) return swap(t.i, other.i);   // already on: trade places
    tune(t, num).then(save);
  });
  function swap(a, b) {
    // Trade the players themselves (no retuning): slot order is what's main and what's side.
    [tiles[a], tiles[b]] = [tiles[b], tiles[a]];
    tiles.forEach((t, i) => { t.i = i; });
    focus = null;
    decide(); drawSlots(); save();
  }
  function save() {
    WATCH_PREFS.multi = tiles.map(t => t.num || "").join(",").replace(/,+$/, "");
    saveWatchPrefs();
    location.replace(`#/multi/${encodeURIComponent(WATCH_PREFS.multi)}`);
  }

  // ---------- data ----------
  async function refreshChannels() {
    try {
      const d = await api("api/channels");
      if (!alive) return;
      chans = d.channels || [];
      if (tuners == null) {
        tuners = d.tuners || null;
        const n = tuners || 2;
        if (!tuners) $("#mvnote").textContent = "The proxy doesn't know how many tuners there are (set TUNER_COUNT); showing 2.";
        for (let i = 0; i < n; i++) tiles.push(makeTile(i));
        const pick = saved.slice(0, n).map(num => chans.some(c => c.num === num) ? num : null);
        pick.forEach((num, i) => { tiles[i].num = num; });   // all named before any asks, so none stops another
        drawSlots();
        await Promise.all(pick.map((num, i) => num ? tune(tiles[i], num) : null));
        drawSlots();
        decide();
      } else drawSlots();
      if (d.error) $("#mvnote").textContent = `Can't reach the proxy: ${d.error}`;
    } catch (e) { /* next time */ }
  }
  async function refreshBreaks() {
    try {
      const d = await api("api/breaks");
      if (!alive || !d.channels) return;
      for (const t of tiles) {
        const b = t.num && d.channels[t.num];
        t.breaks = b ? b.breaks : []; t.known = b ? b.known_until : null;
      }
    } catch (e) { /* next time */ }
  }
  await refreshChannels();
  refreshBreaks();
  every(15000, refreshChannels);
  every(2000, refreshBreaks);
  every(500, () => tiles.length && decide());

  // ---------- controls ----------
  $("#mvvol").value = P.volume;
  const setSound = (volume, muted) => {
    P.volume = volume; P.muted = muted;
    WATCH_PREFS.volume = volume; WATCH_PREFS.muted = muted; saveWatchPrefs();
    soundBlocked = false; applySound(); drawLabels();
  };
  $("#mvmute").onclick = () => setSound(P.volume, !P.muted);
  $("#mvvol").oninput = e => setSound(+e.target.value, false);
  $("#mvcap").onclick = () => { P.cc = !P.cc; WATCH_PREFS.cc = P.cc; saveWatchPrefs(); applySound(); };
  const box = $(".multi");
  function fullscreen() {
    if (document.fullscreenElement) return document.exitFullscreen();
    if (box.requestFullscreen) box.requestFullscreen();
  }
  $("#mvfs").onclick = fullscreen;
  const keys = e => {
    if (e.target.closest("input, select, textarea")) return;
    if (e.key === "f") fullscreen();
    else if (e.key === "m") $("#mvmute").click();
    else if (e.key === "c") $("#mvcap").click();
  };
  document.addEventListener("keydown", keys);
  applySound();

  cleanup = () => {
    alive = false;
    document.removeEventListener("keydown", keys);
    for (const t of tiles) {
      if (t.hls) t.hls.destroy();
      t.video.pause(); t.video.removeAttribute("src"); t.video.load();   // lets the tuners go
    }
  };
}
