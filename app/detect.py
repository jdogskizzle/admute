"""Ad detection from the decoded picture and sound.

The main signal is the station logo. Most stations keep a logo in one corner during shows
and drop it during ads. The detector learns where the logo is by finding edges that stay
put over many minutes, saves that per channel, and calls it an ad once the logo has been
gone for a few seconds. Black frames and silence, which usually mark the cut into and out
of a break, are used to place the start and end of the mute.
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
# Thresholds are fractions of the channel's usual logo score (often about 0.95). Ad graphics
# that happen to sit in the logo corner (QR codes, end cards) scored up to about 0.66.
LOGO_ON = _env("LOGO_ON", 0.85)                # above this share of the usual score: logo is on screen
LOGO_OFF = _env("LOGO_OFF", 0.75)              # below this share: logo is gone
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
        self.level = 0.95  # the channel's usual logo score while the logo is on screen
        if self.state_path.exists():
            try:
                with np.load(self.state_path) as d:
                    self.freq, self.seen = d["freq"].astype(np.float32), int(d["seen"])
                    if "level" in d:
                        self.level = float(d["level"])
            except Exception as exc:
                log.warning("%s: couldn't load saved logo (%s), learning again", num, exc)
        self.mask, self.corner = None, None
        self._update_mask()

        self.score = None
        self.logo, self.changed_at = True, None
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

        if self.mask is not None:
            log.info("%s: using saved logo in the %s corner (%d px)", num, self.corner, self.mask.sum())
        else:
            left = max(0, LEARN_MIN - self.seen) / FPS / 60
            log.info("%s: learning the logo, about %.0f more minutes of video needed", num, left)

    # --- inputs -------------------------------------------------------------

    def frame(self, t, img):
        self.last_t = t
        g = img.astype(np.int16)
        black = np.count_nonzero(g > 40) < g.size * 0.01
        if black:
            self.blacks.append(t)
        edges = (np.abs(g[:-1, 1:] - g[:-1, :-1]) + np.abs(g[1:, :-1] - g[:-1, :-1])) > EDGE

        if not black and self.state != "ad":
            self.seen += 1
            self.freq += max(1.0 / self.seen, 1.0 / LEARN_TAU) * (edges - self.freq)
            if self.mask is None or self.seen % (FPS * 30) == 0:
                had = self.mask is not None
                self._update_mask()
                if self.mask is not None and not had:
                    log.info("%s: logo learned in the %s corner (%d px), ad muting is on",
                             self.num, self.corner, self.mask.sum())
                    self.save()

        self.score = float(edges[self.mask].mean()) if self.mask is not None else None
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

    def _update_mask(self):
        if self.seen < LEARN_MIN:
            self.mask = None
            return
        best, best_n = None, 0
        for name, (ys, xs) in CORNERS.items():
            sub = self.freq[ys, xs] > MASK_T
            # Long straight lines are letterbox/pillarbox edges or tickers, not a logo.
            cols, rows = sub.mean(0) > 0.5, sub.mean(1) > 0.5
            sub[:, cols] = False
            sub[rows, :] = False
            n = int(sub.sum())
            if n > best_n:
                best, best_n = (name, ys, xs, sub), n
        if best_n < 80:  # smaller than this is usually a bit of static scenery, not a logo
            self.mask = None
            return
        name, ys, xs, sub = best
        mask = np.zeros(self.freq.shape, bool)
        mask[ys, xs] = sub
        self.mask, self.corner = mask, name

    def _decide(self, t):
        s = self.score
        if s is None:
            return
        on, off = LOGO_ON * self.level, LOGO_OFF * self.level
        logo = True if s >= on else False if s <= off else self.logo
        if logo and s >= on:  # follow slow drift in how strongly the logo shows up
            self.level = min(1.0, max(0.3, self.level + (s - self.level) / (60 * FPS)))
        if logo != self.logo or self.changed_at is None:
            self.logo, self.changed_at = logo, t
        held = t - self.changed_at

        if self.state == "show":
            if logo:
                if not self.armed and held >= ARM:
                    self.armed = True
                    self._event(f"logo on screen since {self.fmt(self.changed_at)}, ad muting armed")
            elif held >= AD_MIN and self.armed:
                start = self._snap(self.changed_at)
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
            np.savez_compressed(f, freq=self.freq, seen=self.seen, level=self.level)
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
            "logo_learned": self.mask is not None,
            "logo_corner": self.corner,
            "learning_minutes_left": round(max(0, LEARN_MIN - self.seen) / FPS / 60, 1),
            "logo_score": None if self.score is None else round(self.score, 3),
            "logo_usual_score": round(self.level, 3),
            "state": self.state,
            "armed": self.armed,
            "ads_muted": self.ads,
        }
