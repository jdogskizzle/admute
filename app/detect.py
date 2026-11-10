"""Ad detection from the decoded picture and sound.

The main signal is the on-screen graphics. Most stations keep a logo in one corner during
shows and drop it during ads, and sports add a scorebug. The detector learns where these
are by finding edges that stay put over many minutes (up to two graphics, in different
corners), saves that per channel, and calls it an ad once all of them have been gone for a
few seconds. Each graphic is recognised by matching its shape against a learned picture of
it, which still works when it is see-through and the background is bright (a translucent
bug over ice). Black frames and silence, which usually mark the cut into and out of a
break, are used to place the start and end of the mute.
"""

import csv
import logging
import math
import os
import struct
import time
import zlib
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import scene as scene_mod

import numpy as np

log = logging.getLogger("admute")

W, H, FPS = 384, 216, 5
AUDIO_RATE = 16000   # what speech-to-text wants


def _env(name, default):
    return float(os.environ.get(name, default))


EDGE = _env("EDGE", 24)                        # brightness step that counts as an edge
MASK_T = _env("MASK_T", 0.6)                   # share of frames a pixel must be an edge to count as logo
LEARN_MIN = _env("LEARN_MIN", 5) * 60 * FPS    # minutes of video before the logo is trusted
LEARN_TAU = _env("LEARN_TAU", 15) * 60 * FPS   # how slowly the learned logo changes
# Shape match of a graphic against its learned picture, 0..1. On a hockey game, play scored
# mostly 0.5-1 (even replays over bright ice) and ads mostly under 0.05.
LOGO_ON = _env("LOGO_ON", 0.35)                # above this: the graphic is on screen
LOGO_OFF = _env("LOGO_OFF", 0.3)               # below this: it's gone
# Until a graphic's picture is learned, edge scores are used: fractions of its usual score.
EDGE_ON, EDGE_OFF = 0.85, 0.75
GRAPHICS = int(_env("GRAPHICS", 2))            # how many graphics (corners) to follow
PICTURE_MIN = 30 * FPS                         # frames of a graphic needed before its picture is used
SURFACE = _env("SURFACE", 1)                   # 0 turns off the playing-surface check
MERGE = _env("MERGE", 15)                      # an ad starting this soon after one ended continues the break
AD_MIN = _env("AD_MIN", 4)                     # seconds without the logo before muting
SHOW_MIN = _env("SHOW_MIN", 3)                 # seconds with the logo back before unmuting
AD_MAX = _env("AD_MAX", 300)                   # give up on a "break" this long (show with no logo)
ARM = _env("ARM", 20)                          # logo must be on screen this long before muting is armed
SNAP = _env("SNAP", 3)                         # look this far back for a black frame or silence
SILENCE = 32768 * 10 ** (_env("SILENCE_DB", -50) / 20)

# The settings the mute decisions use. A survey's trace can be replayed with other values
# (see Decider), which is how the studio tries settings without decoding the video again.
SURFACE_AT = _env("SURFACE_AT", 0.5)           # playing surface: where between "usual in ads" (0) and "usual in show" (1) it counts
SURFACE_HOLD = _env("SURFACE_HOLD", 2)         # ...and it must stay on screen this long (ads flash colours too)
# Splice marks: some channels cut to a break with a black frame and/or silence, and their
# own graphics and studio cut-ins never have one. Where a channel's breaks reliably start
# with a mark (learned), a lost logo only mutes from a mark, or after SPLICE_WAIT seconds.
SPLICE = _env("SPLICE", -1)                    # -1 learn per channel, 0 off, 1 always
SPLICE_WAIT = _env("SPLICE_WAIT", 20)
# Side-by-side ads: the network's fixed frame around a small game picture and the ad,
# learned from labelled examples (replay.py layouts) into DATA/layouts, shared by all channels.
LAYOUT = _env("LAYOUT", 1)                     # 0 ignores the learned layouts
LAYOUT_HOLD = _env("LAYOUT_HOLD", 0.6)         # seconds a layout must match before muting
# Scene check (scene.py): a classifier learned from reviewed recordings gives the chance each
# second is an ad. When one has been trained, it decides; the logo rules are the fallback.
SCENE = _env("SCENE", 1)                       # 0: ignore the scene classifier
SCENE_ON = _env("SCENE_ON", 0.9)               # mute when the recent average chance is above this
SCENE_OFF = _env("SCENE_OFF", 0.5)             # unmute when it drops below this
SCENE_WINDOW = _env("SCENE_WINDOW", 6)        # seconds averaged
# Sponsor billboards in the commentary (listen.py), muted during sports: needs the speech model.
SPEECH = _env("SPEECH", 1)                     # 0: don't listen
SETTINGS = dict(SCENE=SCENE, SCENE_ON=SCENE_ON, SCENE_OFF=SCENE_OFF, SCENE_WINDOW=SCENE_WINDOW, LAYOUT=LAYOUT, LAYOUT_HOLD=LAYOUT_HOLD, LOGO_ON=LOGO_ON, LOGO_OFF=LOGO_OFF, AD_MIN=AD_MIN, SHOW_MIN=SHOW_MIN, MERGE=MERGE,
                AD_MAX=AD_MAX, ARM=ARM, SNAP=SNAP, SURFACE=SURFACE, SURFACE_AT=SURFACE_AT,
                SURFACE_HOLD=SURFACE_HOLD, SPLICE=SPLICE, SPLICE_WAIT=SPLICE_WAIT)
SPLICE_BREAK = 60      # mutes at least this long count as real breaks when learning about marks
SPLICE_SHARE = 0.75    # share of them that must start with a mark to rely on it

_hh, _ww = int((H - 1) * 0.28), int((W - 1) * 0.28)
CORNERS = {
    "top-left": (slice(0, _hh), slice(0, _ww)),
    "top-right": (slice(0, _hh), slice(W - 1 - _ww, W - 1)),
    "bottom-left": (slice(H - 1 - _hh, H - 1), slice(0, _ww)),
    "bottom-right": (slice(H - 1 - _hh, H - 1), slice(W - 1 - _ww, W - 1)),
}


def write_png(path, img):
    h, w = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        f.write(png)
    os.replace(tmp, path)


def highpass(p):
    """Picture minus its 3x3 blur: keeps the outline of a graphic, drops the background."""
    h, w = p.shape
    pad = np.pad(p, 1, mode="edge")
    return p - sum(pad[i:i + h, j:j + w] for i in range(3) for j in range(3)) / 9


def normalize(p):
    return (p - p.mean()) / (p.std() + 1e-3)


_SCENE_POOL = ThreadPoolExecutor(1, thread_name_prefix="scene")


def edge_map(g):
    """Pixels with a sharp brightness step to the right or below (g: int16 picture)."""
    return (np.abs(g[:-1, 1:] - g[:-1, :-1]) + np.abs(g[1:, :-1] - g[:-1, :-1])) > EDGE


class Layout:
    """A side-by-side ad frame: edges that are there in every example of it. Its score is the
    share of them lit in a frame, relative to the examples; above `threshold` it's on screen."""

    def __init__(self, name, mask, level, threshold):
        self.name, self.mask, self.level, self.threshold = name, mask, level, threshold

    def score(self, edges):
        return float(edges[self.mask].mean()) / self.level

    @staticmethod
    def load_all():
        folder = Path(os.environ.get("DATA", "/data")) / "layouts"
        out = []
        for f in sorted(folder.glob("*.npz")):
            try:
                with np.load(f) as d:
                    out.append(Layout(f.stem, d["mask"].astype(bool), float(d["level"]), float(d["threshold"])))
            except Exception as exc:
                log.warning("couldn't load layout %s: %s", f.name, exc)
        return out


class Graphic:
    """One steady on-screen graphic (station logo or scorebug) in one corner."""

    def __init__(self, corner, mask, box, picture=None, n=0, level=0.95):
        self.corner, self.mask, self.box = corner, mask, box
        self.picture = np.zeros((box[1] - box[0], box[3] - box[2]), np.float32) if picture is None else picture
        self.n, self.level = n, level
        self.edge = self.shape = None

    def measure(self, g, edges, learn, hi=None):
        """Score this frame; `learn` lets a clearly visible graphic refine its picture.
        `hi` is the same frame at twice the size, for a sharper picture of small graphics."""
        self.edge = float(edges[self.mask].mean())
        y0, y1, x0, x1 = self.box
        patch = g[y0:y1, x0:x1] if hi is None else hi[2 * y0:2 * y1, 2 * x0:2 * x1]
        hp = highpass(patch.astype(np.float32))
        if hp.shape != self.picture.shape:  # picture was learned at another size: learn it again
            self.picture, self.n = np.zeros(hp.shape, np.float32), 0
        if learn and self.edge >= 0.8 * self.level:
            self.n += 1
            self.picture += max(1.0 / self.n, 1.0 / LEARN_TAU) * (hp - self.picture)
        if self.edge >= EDGE_ON * self.level:
            self.level = min(1.0, max(0.3, self.level + (self.edge - self.level) / (60 * FPS)))
        if self.n >= PICTURE_MIN:
            # A see-through graphic can show up darker than a bright background, so the
            # match can come out negative; either way it's there.
            self.shape = abs(float((normalize(hp) * normalize(self.picture)).mean()))
        else:
            self.shape = None

    def state(self, p=SETTINGS):
        """True (on screen), False (gone) or None (in between)."""
        return graphic_state(self.shape, self.edge / self.level, p)

    def score(self):
        return self.shape if self.shape is not None else min(1.0, self.edge / self.level)


def graphic_state(shape, edge, p):
    """On screen (True), gone (False) or in between (None), from the shape match, or from
    the edge score (relative to usual) while the graphic's picture isn't learned yet."""
    if shape is not None:
        v, on, off = shape, p["LOGO_ON"], p["LOGO_OFF"]
    else:
        v, on, off = edge, EDGE_ON, EDGE_OFF
    return True if v >= on else False if v <= off else None


class Surface:
    """The colour that fills most of the screen during the show, like ice or grass.

    Sports replays and close-ups often drop the on-screen graphics, but still show the
    playing surface, while ads almost never do. So while enough of the screen is this colour,
    a missing logo doesn't start a mute. It's only used on channels where it clearly tells
    the show from the ads (learned from both), and it never ends a mute.
    """
    Y_BINS, C_BINS = 3, 5  # odd number of colour bins, so neutral (white, grey) sits in the middle one

    def __init__(self, hist=None, show=0.0, ad=0.0, n_show=0, n_ad=0):
        self.hist = np.zeros(self.Y_BINS * self.C_BINS ** 2, np.float32) if hist is None else hist
        self.show, self.ad, self.n_show, self.n_ad = show, ad, n_show, n_ad
        self.share = 0.0

    def bins(self, y, uv):
        """Bin number per pixel of the lower two thirds of the picture (graphics live up top)."""
        yy = y[::2, ::2][H // 6:]
        c = uv[H // 6:].astype(np.int16)
        return ((yy.astype(np.int16) * self.Y_BINS) >> 8) * self.C_BINS ** 2 \
            + ((c[..., 0] * self.C_BINS) >> 8) * self.C_BINS + ((c[..., 1] * self.C_BINS) >> 8)

    def measure(self, y, uv, show, ad):
        b = self.bins(y, uv)
        top = int(self.hist.argmax())
        self.share = float(np.count_nonzero(b == top)) / b.size
        if show:  # graphics on screen: learn what the show looks like
            self.n_show += 1
            h = np.bincount(b.ravel(), minlength=self.hist.size) / b.size
            self.hist += max(1.0 / self.n_show, 1.0 / LEARN_TAU) * (h - self.hist)
            self.show += max(1.0 / self.n_show, 1.0 / LEARN_TAU) * (self.share - self.show)
        elif ad:  # well inside a break: learn how much of it ads show
            self.n_ad += 1
            self.ad += max(1.0 / self.n_ad, 1.0 / LEARN_TAU) * (self.share - self.ad)

    def learned(self):
        return self.n_show >= LEARN_MIN and self.n_ad >= 60 * FPS

    def threshold(self, p=SETTINGS):
        """Share of the screen that counts as "the show is on", or None while it isn't useful."""
        return surface_threshold(self.learned(), self.show, self.ad, p)

    def visible(self, p=SETTINGS):
        th = self.threshold(p)
        return th is not None and self.share >= th


def surface_threshold(learned, show, ad, p):
    if not p["SURFACE"] or not learned:
        return None
    if show < 0.3 or show - ad < 0.25:  # not a clear difference on this channel
        return None
    return ad + (show - ad) * p["SURFACE_AT"]


class Decider:
    """When to mute: all graphics gone for AD_MIN seconds (and the playing surface not on
    screen), until one is back for SHOW_MIN seconds. Black frames and silence place the cut.
    Kept apart from the picture analysis so a survey's trace can be replayed with other settings."""

    def __init__(self, p=SETTINGS, fmt=str, now=time.monotonic, event=None, keep=10, splice=(0, 0)):
        self.p, self.fmt, self.now, self.event, self.keep = p, fmt, now, event or (lambda m: None), keep
        self.logo, self.changed_at = True, None
        self.surface_at = None  # last time the playing surface was on screen (for SURFACE_HOLD)
        self.surface_since = None
        self.splice = list(splice)  # [breaks seen, breaks that started with a mark]
        self.loss_marked = None  # whether the break being muted started at a black frame or silence
        self.layout_since = None  # when a side-by-side ad layout came on screen
        self.scene_hist = deque(maxlen=120)  # (second, chance it's an ad), from the scene classifier
        # Not armed until the logo has actually been seen, so a show without it isn't muted.
        self.state, self.armed = "show", False
        self.intervals = []  # [start, end] mute ranges, in stream arrival time
        self.ads = 0
        self.blacks = deque(maxlen=300)
        self.silences = deque(maxlen=600)

    def step(self, t, states, surface=False, layout=False, scene=None):
        """states: on/gone/None per graphic for this frame; surface: playing surface on screen;
        layout: a learned side-by-side ad layout matches this frame; scene: the scene
        classifier's chance that this second is an ad (None without a classifier)."""
        p = self.p
        if layout and p["LAYOUT"]:
            if self.layout_since is None:
                self.layout_since = t
        else:
            self.layout_since = None
        sbs = self.layout_since is not None and t - self.layout_since >= p["LAYOUT_HOLD"]
        if scene is not None and p["SCENE"]:
            return self._scene_step(t, scene, sbs)
        if sbs:  # a side-by-side ad: whatever graphics the small game picture shows don't count
            states = [False] * max(1, len(states))
        if surface:
            if self.surface_since is None:
                self.surface_since = t
            if t - self.surface_since >= p["SURFACE_HOLD"]:
                self.surface_at = t
        else:
            self.surface_since = None
        if not states:
            return
        # On screen while any graphic is clearly there; gone once all of them clearly are.
        logo = True if True in states else False if all(v is False for v in states) else self.logo
        if logo != self.logo or self.changed_at is None:
            self.logo, self.changed_at = logo, t
        held = t - self.changed_at

        if self.state == "show":
            if logo:
                if not self.armed and held >= p["ARM"]:
                    self.armed = True
                    self.event(f"logo on screen since {self.fmt(self.changed_at)}, ad muting armed")
            elif sbs or (self.armed and t - max(self.changed_at, self.surface_at or 0) >= p["AD_MIN"]):
                gone = self.layout_since if sbs else max(self.changed_at, self.surface_at or 0)
                start = self._snap(gone)
                if sbs:
                    start = max(start, t - (_env("DELAY", 10) - 1))
                elif self.splice_needed():
                    marks = self._marks(gone - p["SNAP"], t)
                    if marks:  # mute from the cut (or a later one, after a station promo)
                        near = [m for m in marks if m <= gone + 0.5]
                        start = max(near) if near else min(marks)
                    elif t - gone >= p["SPLICE_WAIT"]:
                        start = gone
                    else:
                        return  # no cut yet: probably the channel's own graphic or studio
                    # what viewers haven't heard yet (DELAY behind, decoding about 1 s behind live)
                    start = max(start, t - (_env("DELAY", 10) - 1))
                self.loss_marked = bool(self._marks(gone - p["SNAP"], min(t, gone + 4)))
                self.state = "ad"
                prev = self.intervals[-1] if self.intervals else None
                if prev and start - prev[1] <= p["MERGE"]:
                    # Back-to-back ads with a logo-ish card between them: keep the break muted.
                    gap = start - prev[1]
                    prev[1] = math.inf
                    self.event(f"next ad after a {gap:.0f}s gap, staying muted")
                else:
                    self.intervals.append([start, math.inf])
                    if self.keep:
                        self.intervals = self.intervals[-self.keep:]
                    self.ads += 1
                    # Viewers hear `start` at start + DELAY; positive margin means the mute was in time.
                    margin = start + _env("DELAY", 10) - self.now()
                    note = f"{margin:.1f}s to spare" if margin >= 0 else f"first {-margin:.1f}s of the ad was heard"
                    self.event(f"ad start, muting from {self.fmt(start)} ({note})")
        else:
            cur = self.intervals[-1]
            if logo and held >= p["SHOW_MIN"]:
                cur[1] = max(cur[0], self._snap(self.changed_at))
                self.state = "show"
                if cur[1] - cur[0] >= SPLICE_BREAK and self.loss_marked is not None:
                    self.splice[0] += 1
                    self.splice[1] += self.loss_marked
                    if self.splice[0] == 5 or self.splice[0] % 20 == 0:
                        self.event(f"breaks starting at a black frame or silence: {self.splice[1]} of {self.splice[0]}"
                                   + (", so other logo losses wait for one" if self.splice_needed() else ""))
                self.event(f"ad end at {self.fmt(cur[1])}, muted {cur[1] - cur[0]:.0f}s")
            elif t - cur[0] >= p["AD_MAX"]:
                cur[1] = t
                self.state, self.armed = "show", False
                self.event(f"no logo for {p['AD_MAX']:.0f}s, probably a show without it; unmuted until the logo is back")

    def _scene_step(self, t, scene, layout):
        """Mute when the scene classifier is sure it's an ad over the last few seconds, unmute
        when it's sure it's the broadcast again. Boundaries go back to where the chance
        crossed one half (as far as the delay allows), then to a nearby black frame or silence."""
        p, sec = self.p, int(t)
        if not self.scene_hist or self.scene_hist[-1][0] != sec:
            self.scene_hist.append((sec, scene))
        win = [(s, q) for s, q in self.scene_hist if s > t - p["SCENE_WINDOW"]]
        avg = sum(q for _, q in win) / len(win)
        floor = t - (_env("DELAY", 10) - 1)  # viewers haven't heard anything after this yet
        sbs = layout  # a side-by-side layout has matched for LAYOUT_HOLD
        if self.state == "show":
            if avg >= p["SCENE_ON"] or sbs:
                first = next((s for s, q in win if q > 0.5), sec)
                start = max(self._snap(first), floor)
                self.state = "ad"
                prev = self.intervals[-1] if self.intervals else None
                if prev and start - prev[1] <= p["MERGE"]:
                    gap = start - prev[1]
                    prev[1] = math.inf
                    self.event(f"ad again after a {gap:.0f}s gap, staying muted")
                else:
                    self.intervals.append([start, math.inf])
                    if self.keep:
                        self.intervals = self.intervals[-self.keep:]
                    self.ads += 1
                    self.event(f"ad start, muting from {self.fmt(start)} (scene check {avg:.2f})")
        else:
            cur = self.intervals[-1]
            if not sbs and avg <= p["SCENE_OFF"]:
                last_ad = max((s for s, q in win if q > 0.5), default=sec - p["SCENE_WINDOW"])
                cur[1] = max(cur[0], self._snap(last_ad + 1), floor)
                self.state = "show"
                self.event(f"ad end at {self.fmt(cur[1])}, muted {cur[1] - cur[0]:.0f}s (scene check {avg:.2f})")
            elif t - cur[0] >= p["AD_MAX"]:
                cur[1] = t
                self.state = "show"
                self.event(f"muted for {p['AD_MAX']:.0f}s, unmuting")

    def splice_needed(self):
        mode = self.p["SPLICE"]
        if mode >= 0:
            return bool(mode)
        n, marked = self.splice
        return n >= 5 and marked >= SPLICE_SHARE * n

    def _marks(self, a, b):
        return sorted(m for m in (*self.blacks, *self.silences) if a <= m <= b)

    def _snap(self, t):
        """Move a boundary to the nearest black frame or silence just before it, if there is one."""
        marks = self._marks(t - self.p["SNAP"], t + 0.5)
        return min(marks, key=lambda m: abs(m - t)) if marks else t


# Per-frame trace a survey saves (trace.npz "frames"), one row per frame:
TRACE_COLS = ["t", "black", "surface_share", "surface_show", "surface_ad", "surface_learned"] + \
    [f"g{i}_{k}" for i in range(GRAPHICS) for k in ("shape", "edge")]


def scene_chances(trace, classifier):
    """The scene classifier's chance of an ad for each frame of a trace (None if it can't)."""
    if classifier is None or "scene_feats" not in trace or not len(trace["scene_sec"]):
        return None
    secs = trace["scene_sec"]
    q = classifier.p(trace["scene_feats"].astype(np.float32))
    t = trace["frames"][:, 0]
    i = np.searchsorted(secs, np.floor(t), side="right") - 1  # the latest second already checked
    out = np.where(i >= 0, q[np.clip(i, 0, len(q) - 1)], np.nan)
    return out


def replay_trace(trace, p, classifier="saved", title=None):
    """Mute ranges a survey would have made with settings p, from its saved trace (fast:
    no video). Learning isn't redone, so it's close to a full survey but not exact.
    classifier: the scene classifier to use ("saved": the trained one, None: none)."""
    p = {**SETTINGS, **p}
    if classifier == "saved":
        classifier = scene_mod.Classifier.load()
    if title is not None and not scene_mod.covers(classifier, title):
        classifier = None  # not a kind of program it learned from
    sp = scene_chances(trace, classifier)
    d = Decider(p, keep=None, splice=tuple(trace["splice"]) if "splice" in trace else (0, 0))
    f, sil, si = trace["frames"], trace["silences"], 0
    ng = (f.shape[1] - 6) // 2
    lay = trace.get("layouts")
    sbs = (lay >= trace["layout_thresholds"]).any(axis=1) if lay is not None and lay.size else np.zeros(len(f), bool)
    for k, row in enumerate(f.tolist()):
        t = row[0]
        while si < len(sil) and sil[si] <= t:
            d.silences.append(float(sil[si]))
            si += 1
        if row[1]:
            d.blacks.append(t)
        states = [graphic_state(None if math.isnan(row[6 + 2 * i]) else row[6 + 2 * i], row[7 + 2 * i], p)
                  for i in range(ng) if not math.isnan(row[7 + 2 * i])]
        th = surface_threshold(bool(row[5]), row[3], row[4], p)
        d.step(t, states, th is not None and row[2] >= th, bool(sbs[k]),
               None if sp is None or math.isnan(sp[k]) else float(sp[k]))
    end = f[-1, 0] if len(f) else 0.0
    return [(s, float(end) if e == math.inf else e) for s, e in d.intervals]


def wall(t):
    """Monotonic time -> wall clock string."""
    return time.strftime("%H:%M:%S", time.localtime(time.time() - (time.monotonic() - t)))


class AdDetector:
    def __init__(self, num, name, data_dir, fmt=None, now=None, keep=10, trace=False, live=False, title=None):
        """fmt turns a stream time into a readable label; now is the current time on the same
        clock (live: arrival time; replay: how far the recording has been decoded).
        trace: keep per-frame scores (surveys save them for the studio to replay)."""
        self.num, self.name = num, name
        self.fmt, self.now, self.keep = fmt or wall, now or time.monotonic, keep
        self.last_t = None
        self.anchor = None  # (PTS, detector time) of one picture: puts audio timestamps on this clock
        logos = data_dir / "logos"
        logdir = data_dir / "logs" / num
        logos.mkdir(parents=True, exist_ok=True)
        logdir.mkdir(parents=True, exist_ok=True)
        self.state_path = logos / f"{num}.npz"
        self.png_path = logos / f"{num}.png"
        self.events_path = data_dir / "logs" / "events.log"

        self.freq = np.zeros((H - 1, W - 1), np.float32)  # how often each pixel is an edge
        self.seen = 0
        self.graphics = []  # Graphic per followed corner, biggest first
        self.surface = Surface()
        if self.state_path.exists():
            try:
                with np.load(self.state_path) as d:
                    self.freq, self.seen = d["freq"].astype(np.float32), int(d["seen"])
                    for i in range(GRAPHICS):
                        if f"g{i}_corner" in d:
                            corner = str(d[f"g{i}_corner"])
                            ys, xs = CORNERS[corner]
                            mask = np.zeros(self.freq.shape, bool)
                            self.graphics.append(Graphic(corner, mask, tuple(int(v) for v in d[f"g{i}_box"]),
                                                         d[f"g{i}_picture"].astype(np.float32),
                                                         int(d[f"g{i}_n"]), float(d[f"g{i}_level"])))
                    if "surface_hist" in d:
                        self.surface = Surface(d["surface_hist"].astype(np.float32), float(d["surface_show"]),
                                               float(d["surface_ad"]), int(d["surface_n_show"]), int(d["surface_n_ad"]))
                    if "splice" in d:
                        self._splice = tuple(int(v) for v in d["splice"])
                    if not self.graphics and "level" in d:  # saved by an older version: one graphic
                        self._old_level = float(d["level"])
            except Exception as exc:
                log.warning("%s: couldn't load saved logo (%s), learning again", num, exc)
                self.graphics = []
        self._update_graphics()

        self.score = None
        self._splice = getattr(self, "_splice", (0, 0))
        self.decider = Decider(SETTINGS, self.fmt, lambda: self.now(), self._event, keep, self._splice)
        self.trace = ([], [], [], []) if trace else None  # (frame rows, silences, layout scores, scene features)
        self.layouts = Layout.load_all()
        self.scene_on = scene_mod.available() and SCENE
        self.classifier = scene_mod.Classifier.load() if self.scene_on else None
        self.scene_p, self._scene_sec, self._scene_job = None, None, None
        self._sync_scene = trace  # surveys wait for each result so they're repeatable; live never waits
        # Live TV: only during sports, per the guide. Recordings for the studio are games.
        # Recordings: the title from the file name. Only programs the classifier learned from use it.
        self.live, self._sports_at = live, 0.0
        self._listen_off_until = 0.0   # after speech-to-text fails, wait before trying again
        self._sports = not live and scene_mod.covers(self.classifier, title or scene_mod.show_title(name))
        self.listener = None
        if not live and not trace:   # renders of recordings (games); surveys don't need it
            self._listen(True)
        if self.classifier:
            log.info("%s: scene check ready (learned from %d games)%s", num, len(self.classifier.info.get("trained_on", [])),
                     "; used during sports, per the guide" if live else "")
        if self.layouts:
            log.info("%s: watching for %d side-by-side ad layout%s", num, len(self.layouts), "s" * (len(self.layouts) > 1))

        self._csv_file = open(logdir / f"{time.strftime('%Y%m%d-%H%M%S')}.csv", "w", newline="")
        self._csv = csv.writer(self._csv_file)
        self._csv.writerow(["time", "score", "score_min", "black", "silent", "logo", "state"])
        self._sec, self._scores, self._black, self._silent, self._rows = None, [], False, False, 0
        self._last_save = time.monotonic()

        if self.graphics:
            log.info("%s: using saved graphics: %s", num, self._describe())
        else:
            left = max(0, LEARN_MIN - self.seen) / FPS / 60
            log.info("%s: learning the logo, about %.0f more minutes of video needed", num, left)

    # --- inputs -------------------------------------------------------------

    def _scene(self, t, img, uvf):
        """Once a second: the scene check's features, and the chance it's an ad."""
        if not self.scene_on or uvf is None:
            return
        if self._scene_job is not None and self._scene_job.done():
            self._scene_result(*self._scene_job.result())
            self._scene_job = None
        sec = int(t)
        if sec == self._scene_sec or self._scene_job is not None:
            return  # live: skip a second rather than queue up if the CPU is busy
        self._scene_sec = sec
        rgb = scene_mod.picture(img, uvf)
        job = _SCENE_POOL.submit(lambda: (sec, scene_mod.features(rgb)))
        if self._sync_scene:
            self._scene_result(*job.result())
        else:
            self._scene_job = job

    def _scene_result(self, sec, f):
        if self.trace is not None:
            self.trace[3].append((sec, f))
        if self.classifier:
            self.scene_p = float(self.classifier.p(f))

    def frame(self, t, img, uv=None, hi=None, uvf=None):
        self.last_t = t
        g = img.astype(np.int16)
        black = np.count_nonzero(g > 40) < g.size * 0.01
        if black:
            self.decider.blacks.append(t)
        edges = edge_map(g)
        lay = [x.score(edges) for x in self.layouts]
        sbs = any(v >= x.threshold for v, x in zip(lay, self.layouts))

        if not black and self.state != "ad":
            self.seen += 1
            self.freq += max(1.0 / self.seen, 1.0 / LEARN_TAU) * (edges - self.freq)
            if not self.graphics or self.seen % (FPS * 30) == 0:
                had = [x.corner for x in self.graphics]
                self._update_graphics()
                now_ = [x.corner for x in self.graphics]
                if now_ != had and now_:
                    log.info("%s: following %s%s", self.num, self._describe(),
                             ", ad muting is on" if not had else "")
                elif now_ != had:
                    log.info("%s: lost the on-screen graphics, learning them again", self.num)
                    self.save()

        for x in self.graphics:
            x.measure(g, edges, learn=not black and self.state != "ad", hi=hi)
        self.score = max((x.score() for x in self.graphics), default=None)
        sf, share = self.surface, math.nan
        if uv is not None and not black and self.graphics:
            inside = self.state == "ad" and self.intervals and t - self.intervals[-1][0] > 10
            sf.measure(img, uv, show=self.logo and self.state == "show", ad=inside)
            share = sf.share
        self._scene(t, img, uvf)
        if self.live and time.monotonic() - self._sports_at > 60:
            self._sports_at = time.monotonic()
            g = scene_mod.guide()
            titles = g.sports_now(self.num, time.time()) if g else []
            self._listen(bool(titles))
            title = next((x for x in titles if scene_mod.covers(self.classifier, x)), None)
            was, self._sports = self._sports, title is not None
            if self.classifier and was != self._sports:
                log.info("%s: %s", self.num, f"{title} is on: scene check in use" if self._sports
                         else "logo rules in use (no program the scene check learned from)")
        self.decider.step(t, [x.state() for x in self.graphics], share >= (sf.threshold() or 2), sbs,
                          self.scene_p if self._sports else None)
        if self.trace is not None:
            row = [t, float(black), share, sf.show, sf.ad, float(sf.learned())]
            for i in range(GRAPHICS):
                x = self.graphics[i] if i < len(self.graphics) else None
                row += [math.nan, math.nan] if x is None or x.edge is None else \
                       [math.nan if x.shape is None else x.shape, x.edge / x.level]
            self.trace[0].append(row)
            self.trace[2].append(lay)
        self._log_second(t, black)
        if time.monotonic() - self._last_save > 120:
            self.save()

    def _listen(self, on):
        """Start or stop listening for sponsor billboards (sports only)."""
        import listen
        on = on and bool(SPEECH) and os.path.isdir(listen.SPEECH_MODEL) and time.monotonic() >= self._listen_off_until
        if on and self.listener is None:
            self.listener = listen.Listener(AUDIO_RATE, lambda: self.now(), _env("DELAY", 10), self._event,
                                            self.fmt, sync=not self.live)
            log.info("%s: listening for sponsor billboards", self.num)
        elif not on and self.listener is not None:
            self.listener.close()
            self.listener = None
            log.info("%s: stopped listening for sponsor billboards", self.num)

    def audio(self, t, samples):
        if self.listener is not None:
            try:
                self.listener.audio(t, samples, self.intervals)
            except Exception as exc:
                # e.g. the GPU ran out of memory (it's shared): try again in 10 minutes, fresh
                log.warning("%s: speech-to-text failed, trying again in 10 minutes: %s", self.num, exc)
                self.listener.close()
                self.listener = None
                self._listen_off_until = time.monotonic() + 600
                import listen
                listen.drop()
        n = AUDIO_RATE // 10
        for i in range(0, len(samples) - n + 1, n):
            w = samples[i:i + n].astype(np.float32)
            if math.sqrt(float(np.mean(w * w))) < SILENCE:
                self.decider.silences.append(t + i / AUDIO_RATE)
                if self.trace is not None:
                    self.trace[1].append(t + i / AUDIO_RATE)
                self._silent = True

    def muted(self, t):
        return any(s <= t < e for s, e in self.intervals) or (self.listener is not None and self.listener.muted(t))

    def save_trace(self, path):
        rows, sil, lay, _ = self.trace
        np.savez_compressed(path, frames=np.array(rows, np.float32).reshape(-1, len(TRACE_COLS)),
                            layouts=np.array(lay, np.float32).reshape(len(rows), len(self.layouts)),
                            layout_names=np.array([x.name for x in self.layouts]),
                            layout_thresholds=np.array([x.threshold for x in self.layouts], np.float32),
                            scene_sec=np.array([s for s, _ in self.trace[3]], np.int32),
                            scene_feats=np.array([f for _, f in self.trace[3]], np.float16).reshape(-1, 512),
                            silences=np.array(sorted(sil), np.float64), cols=np.array(TRACE_COLS),
                            splice=np.array(self._splice))

    # The decisions live in self.decider; these are its current state.
    intervals = property(lambda self: self.decider.intervals)
    state = property(lambda self: self.decider.state)
    logo = property(lambda self: self.decider.logo)
    armed = property(lambda self: self.decider.armed)
    ads = property(lambda self: self.decider.ads)

    # --- logic --------------------------------------------------------------

    @property
    def mask(self):
        """All followed graphics together (for the preview picture and status)."""
        if not self.graphics:
            return None
        m = np.zeros(self.freq.shape, bool)
        for x in self.graphics:
            m |= x.mask
        return m

    @property
    def corner(self):
        return " + ".join(x.corner for x in self.graphics) or None

    def _describe(self):
        return ", ".join(f"{x.corner} ({x.mask.sum()} px{'' if x.n >= PICTURE_MIN else ', picture not learned yet'})"
                         for x in self.graphics)

    def _update_graphics(self):
        if self.seen < LEARN_MIN:
            if not self.graphics:
                return
        found = []
        for name, (ys, xs) in CORNERS.items():
            sub = self.freq[ys, xs] > MASK_T
            # Long straight lines are letterbox/pillarbox edges or tickers, not a logo.
            cols, rows = sub.mean(0) > 0.5, sub.mean(1) > 0.5
            sub[:, cols] = False
            sub[rows, :] = False
            n = int(sub.sum())
            if n >= 80:  # smaller than this is usually a bit of static scenery, not a logo
                mask = np.zeros(self.freq.shape, bool)
                mask[ys, xs] = sub
                found.append((n, name, mask))
        found.sort(key=lambda f: -f[0])
        old = {x.corner: x for x in self.graphics}
        keep = []
        for n, name, mask in found[:GRAPHICS]:
            yy, xx = np.nonzero(mask)
            ys, xs = CORNERS[name]
            box = (max(ys.start, int(yy.min()) - 3), min(ys.stop, int(yy.max()) + 4),
                   max(xs.start, int(xx.min()) - 3), min(xs.stop, int(xx.max()) + 4))
            x = old.get(name)
            if x and x.box[0] <= box[0] and x.box[1] >= box[1] and x.box[2] <= box[2] and x.box[3] >= box[3]:
                x.mask = mask  # same graphic: keep its learned picture
            else:
                if x:  # it grew past its box: learn the picture again for the bigger box
                    box = (min(box[0], x.box[0]), max(box[1], x.box[1]), min(box[2], x.box[2]), max(box[3], x.box[3]))
                x = Graphic(name, mask, box, level=getattr(self, "_old_level", 0.95))
            keep.append(x)
        if keep or self.seen >= LEARN_MIN:
            self.graphics = keep

    # --- output -------------------------------------------------------------

    def _event(self, msg):
        line = f"{self.num} {self.name}: {msg}"
        log.info(line)
        with open(self.events_path, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")

    def _log_second(self, t, black):
        sec = int(t)
        if self._sec is not None and sec != self._sec:
            sc = self._scores
            self._csv.writerow([self.fmt(self._sec), f"{np.mean(sc):.3f}" if sc else "", f"{min(sc):.3f}" if sc else "",
                                int(self._black), int(self._silent), int(self.logo), self.state if sc else "learning"])
            self._scores, self._black, self._silent = [], False, False
            self._rows += 1
            if self._rows % 10 == 0:
                self._csv_file.flush()
        self._sec = sec
        self._black |= black
        if self.score is not None:
            self._scores.append(self.score)

    def save(self):
        self._last_save = time.monotonic()
        tmp = self.state_path.with_name(f"{self.num}.tmp.npz")
        with open(tmp, "wb") as f:
            extra = {}
            for i, x in enumerate(self.graphics):
                extra.update({f"g{i}_corner": x.corner, f"g{i}_box": np.array(x.box), f"g{i}_picture": x.picture,
                              f"g{i}_n": x.n, f"g{i}_level": x.level})
            sf = self.surface
            np.savez_compressed(f, freq=self.freq, seen=self.seen, surface_hist=sf.hist, surface_show=sf.show,
                                surface_ad=sf.ad, surface_n_show=sf.n_show, surface_n_ad=sf.n_ad,
                                splice=np.array(self.decider.splice), **extra)
        os.replace(tmp, self.state_path)
        img = np.clip(self.freq * 160, 0, 160).astype(np.uint8)
        if self.mask is not None:
            img[self.mask] = 255
        write_png(self.png_path, img)

    def close(self):
        if self.listener is not None:
            self.listener.close()
        try:
            self.save()
        finally:
            self._csv_file.close()

    def status(self):
        return {
            "logo_learned": bool(self.graphics),
            "learning_minutes_left": round(max(0, LEARN_MIN - self.seen) / FPS / 60, 1),
            "logo_score": None if self.score is None else round(self.score, 3),
            "graphics": [{"corner": x.corner, "pixels": int(x.mask.sum()),
                          "picture_learned": x.n >= PICTURE_MIN,
                          "score": None if x.edge is None else round(x.score(), 3)} for x in self.graphics],
            "surface": {"share": round(self.surface.share, 2), "usual_in_show": round(self.surface.show, 2),
                        "usual_in_ads": round(self.surface.ad, 2), "in_use": self.surface.threshold() is not None},
            "state": self.state,
            "armed": self.armed,
            "ads_muted": self.ads,
            "sponsor_lines_muted": len(self.listener.mutes) if self.listener else 0,
        }
