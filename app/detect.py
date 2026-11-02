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

import numpy as np

log = logging.getLogger("admute")

W, H, FPS = 384, 216, 5
AUDIO_RATE = 8000


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

    def state(self):
        """True (on screen), False (gone) or None (in between)."""
        if self.shape is not None:
            v, on, off = self.shape, LOGO_ON, LOGO_OFF
        else:
            v, on, off = self.edge / self.level, EDGE_ON, EDGE_OFF
        return True if v >= on else False if v <= off else None

    def score(self):
        return self.shape if self.shape is not None else min(1.0, self.edge / self.level)


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

    def threshold(self):
        """Share of the screen that counts as "the show is on", or None while it isn't useful."""
        if not SURFACE or self.n_show < LEARN_MIN or self.n_ad < 60 * FPS:
            return None
        if self.show < 0.3 or self.show - self.ad < 0.25:  # not a clear difference on this channel
            return None
        return (self.show + self.ad) / 2

    def visible(self):
        th = self.threshold()
        return th is not None and self.share >= th


def wall(t):
    """Monotonic time -> wall clock string."""
    return time.strftime("%H:%M:%S", time.localtime(time.time() - (time.monotonic() - t)))


class AdDetector:
    def __init__(self, num, name, data_dir, fmt=None, now=None, keep=10):
        """fmt turns a stream time into a readable label; now is the current time on the same
        clock (live: arrival time; replay: how far the recording has been decoded)."""
        self.num, self.name = num, name
        self.fmt, self.now, self.keep = fmt or wall, now or time.monotonic, keep
        self.last_t = None
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
                    if not self.graphics and "level" in d:  # saved by an older version: one graphic
                        self._old_level = float(d["level"])
            except Exception as exc:
                log.warning("%s: couldn't load saved logo (%s), learning again", num, exc)
                self.graphics = []
        self._update_graphics()

        self.score = None
        self.logo, self.changed_at = True, None
        self.surface_at = None  # last time the playing surface was on screen
        # Not armed until the logo has actually been seen, so a show without it isn't muted.
        self.state, self.armed = "show", False
        self.intervals = []  # [start, end] mute ranges, in stream arrival time
        self.ads = 0
        self.blacks = deque(maxlen=300)
        self.silences = deque(maxlen=600)

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

    def frame(self, t, img, uv=None, hi=None):
        self.last_t = t
        g = img.astype(np.int16)
        black = np.count_nonzero(g > 40) < g.size * 0.01
        if black:
            self.blacks.append(t)
        edges = (np.abs(g[:-1, 1:] - g[:-1, :-1]) + np.abs(g[1:, :-1] - g[:-1, :-1])) > EDGE

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
        if uv is not None and not black and self.graphics:
            inside = self.state == "ad" and self.intervals and t - self.intervals[-1][0] > 10
            self.surface.measure(img, uv, show=self.logo and self.state == "show", ad=inside)
            if self.surface.visible():
                self.surface_at = t
        self._decide(t)
        self._log_second(t, black)
        if time.monotonic() - self._last_save > 120:
            self.save()

    def audio(self, t, samples):
        n = AUDIO_RATE // 10
        for i in range(0, len(samples) - n + 1, n):
            w = samples[i:i + n].astype(np.float32)
            if math.sqrt(float(np.mean(w * w))) < SILENCE:
                self.silences.append(t + i / AUDIO_RATE)
                self._silent = True

    def muted(self, t):
        return any(s <= t < e for s, e in self.intervals)

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

    def _decide(self, t):
        if not self.graphics:
            return
        # On screen while any graphic is clearly there; gone once all of them clearly are.
        states = [x.state() for x in self.graphics]
        logo = True if True in states else False if all(v is False for v in states) else self.logo
        if logo != self.logo or self.changed_at is None:
            self.logo, self.changed_at = logo, t
        held = t - self.changed_at

        if self.state == "show":
            if logo:
                if not self.armed and held >= ARM:
                    self.armed = True
                    self._event(f"logo on screen since {self.fmt(self.changed_at)}, ad muting armed")
            elif self.armed and t - max(self.changed_at, self.surface_at or 0) >= AD_MIN:
                start = self._snap(max(self.changed_at, self.surface_at or 0))
                self.state = "ad"
                prev = self.intervals[-1] if self.intervals else None
                if prev and start - prev[1] <= MERGE:
                    # Back-to-back ads with a logo-ish card between them: keep the break muted.
                    gap = start - prev[1]
                    prev[1] = math.inf
                    self._event(f"next ad after a {gap:.0f}s gap, staying muted")
                else:
                    self.intervals.append([start, math.inf])
                    if self.keep:
                        self.intervals = self.intervals[-self.keep:]
                    self.ads += 1
                    # Viewers hear `start` at start + DELAY; positive margin means the mute was in time.
                    margin = start + _env("DELAY", 10) - self.now()
                    note = f"{margin:.1f}s to spare" if margin >= 0 else f"first {-margin:.1f}s of the ad was heard"
                    self._event(f"ad start, muting from {self.fmt(start)} ({note})")
        else:
            cur = self.intervals[-1]
            if logo and held >= SHOW_MIN:
                cur[1] = max(cur[0], self._snap(self.changed_at))
                self.state = "show"
                self._event(f"ad end at {self.fmt(cur[1])}, muted {cur[1] - cur[0]:.0f}s")
            elif t - cur[0] >= AD_MAX:
                cur[1] = t
                self.state, self.armed = "show", False
                self._event(f"no logo for {AD_MAX:.0f}s, probably a show without it; unmuted until the logo is back")

    def _snap(self, t):
        """Move a boundary to the nearest black frame or silence just before it, if there is one."""
        marks = [m for m in (*self.blacks, *self.silences) if t - SNAP <= m <= t + 0.5]
        return min(marks, key=lambda m: abs(m - t)) if marks else t

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
                                surface_ad=sf.ad, surface_n_show=sf.n_show, surface_n_ad=sf.n_ad, **extra)
        os.replace(tmp, self.state_path)
        img = np.clip(self.freq * 160, 0, 160).astype(np.uint8)
        if self.mask is not None:
            img[self.mask] = 255
        write_png(self.png_path, img)

    def close(self):
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
        }
