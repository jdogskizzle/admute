"""Listening to a recording's sound: what's said, and which sounds come back.

- Speech: the commentary as words with times (faster-whisper on the GPU). Sponsor
  billboards and slogans ("The NFL powered by Verizon 5G") are ads: admute could mute just
  those while the picture plays on. The commentators naming a feature's sponsor, or a brand
  in a segment's name ("coming up on the Verizon halftime show"), is left alone.
- Fingerprints: 20 a second, 32 bits each, from how the loudness of 33 frequency bands
  changes (the Philips/Haitsma method). The same sound gives mostly the same values, so
  sounds that repeat (the league's music jingle, network promos, ads) can be found across
  recordings by matching values that line up over a few seconds.

Times are seconds from the start of the recording, as the studio's video shows them.
"""

import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from pipeline import FFMPEG

RATE = 16000
HOP = 800                  # samples between fingerprints: 20 a second
FPS = RATE // HOP
WIN = 6144                 # 384 ms windows: long next to the step, so timing jitter barely matters
SPEECH_MODEL = os.environ.get("SPEECH_MODEL", "/opt/models/speech")
SPEECH_DEVICE = os.environ.get("SPEECH_DEVICE", "cuda")


def audio(path):
    """The recording's first audio track, mono 16 kHz float, sample 0 at the start of the
    recording (gaps filled with silence, so times stay right through dropouts)."""
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-i", str(path), "-map", "0:a:0", "-vn",
           "-af", "aresample=async=1000:first_pts=0", "-ac", "1", "-ar", str(RATE), "-f", "s16le", "pipe:1"]
    raw = subprocess.run(cmd, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, check=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768


# --- speech ---------------------------------------------------------------------------

_model = None


def model():
    global _model
    if _model is None:
        import logging
        logging.getLogger("faster_whisper").setLevel(logging.WARNING)   # not a line per pass
        from faster_whisper import WhisperModel
        try:
            compute = "int8_float16" if SPEECH_DEVICE == "cuda" else "int8"
            _model = WhisperModel(SPEECH_MODEL, device=SPEECH_DEVICE, compute_type=compute)
        except RuntimeError as exc:
            if SPEECH_DEVICE == "cpu" or "out of memory" not in str(exc):
                raise
            print("  the GPU is out of memory (shared with other apps): using the CPU, slower", flush=True)
            _model = WhisperModel(SPEECH_MODEL, device="cpu", compute_type="int8",
                                  cpu_threads=max(1, (os.cpu_count() or 4) // 2))
    return _model


def transcribe(samples, progress=None):
    """[[start, end, word], ...] for everything said."""
    segs, _ = model().transcribe(samples, language="en", word_timestamps=True, vad_filter=True,
                                 condition_on_previous_text=False, beam_size=1)
    words = []
    for s in segs:
        for w in s.words or []:
            words.append([round(w.start, 2), round(w.end, 2), w.word.strip()])
        if progress:
            progress(s.end)
    return words


def sentences(words, gap=0.6):
    """Words grouped into sentences: split after . ? ! or at a pause. Lists of words."""
    out, cur = [], []
    for w in words:
        if cur and (w[0] - cur[-1][1] >= gap or cur[-1][2][-1:] in ".?!"):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def text_of(ws):
    return " ".join(w[2] for w in ws)


def clause(ws, i, j, back=6.0, ahead=2.0, pause=0.3):
    """Words i..j (the sponsor phrase) widened to its clause: back to a pause or a comma (at
    most `back` seconds), on to the brand name after it (at most `ahead` seconds)."""
    first, last = ws[i][0], ws[j][1]
    while i > 0 and ws[i][0] - ws[i - 1][1] < pause and ws[i - 1][2][-1:] not in ",;:" \
            and first - ws[i - 1][0] < back:
        i -= 1
    while j + 1 < len(ws) and ws[j + 1][0] - ws[j][1] < pause and ws[j][2][-1:] not in ".?!,;:" \
            and ws[j + 1][1] - last < ahead:
        j += 1
    return ws[i:j + 1]


# Sponsor lines are of two kinds. Billboards: the broadcast's own sponsor tags and slogans,
# read by an announcer, the same words every time ("The NFL powered by Verizon 5G", "The NFL
# on CBS is sponsored by GEICO", "You're watching the Allstate halftime", "American Express
# is the official payments partner of the NFL"): these are ads, muted. Credits: the
# commentators naming a feature's sponsor ("aerial coverage brought to you by Taco Bell",
# "first half highlights presented by Target"): left on, like a brand in a segment's name
# ("coming up on the Verizon halftime show").
BY = r"(?:is |are )?(?:powered|sponsored|brought to you|presented|delivered) by"
NETWORK = r"(?:cbs|fox|nbc|abc|espn|prime(?: video)?|amazon)"
BILLBOARD = re.compile("|".join([
    rf"\bthe nfl(?: on {NETWORK})?,? {BY}",                                   # the NFL (on CBS) powered by ...
    rf"\bon {NETWORK}(?: sports)? {BY}",                                       # ... on CBS is sponsored by ...
    rf"\b(?:sunday|monday|thursday|saturday) night football,? {BY}",           # Sunday Night Football is brought to you by
    # The X halftime show is brought to you by ... (as a line of its own, not "their halftime
    # show, powered by ...")
    rf"(?:^|\bthe |[.?!] )(?:[\w&'.-]+ ){{0,3}}halftime (?:show|report),? {BY}",
    r"\b(?:you're watching|you are watching|this has been|presents) the (?:[\w&'.-]+ ){1,3}halftime",
    r"\bofficial \w+(?: \w+){0,2} (?:partner|sponsor|provider)s? of\b",      # ... is the official payments partner of
    r"\b(?:is|are) the official \w+(?: \w+){0,2} of the nfl\b",
    r"\buse (?:the )?(?:promo |offer )?code\b",                               # promo codes
    r"\bpowered by \w+ 5g\b",                                                 # slogans
]), re.I)
CREDIT = re.compile(r"\b(brought to you by|presented by|sponsored by|powered by|delivered by|courtesy of"
                    r"|in partnership with)\b", re.I)
# A brand on a segment's name, e.g. "the Verizon halftime report": left alone.
SEGMENT = re.compile(r"\b(?:[A-Z][\w&'.-]+ ){1,3}(?:halftime|skycam|sky cam|report|update|kickoff"
                     r"|pregame|postgame|starting lineups?|player of the game|play of the game|drive chart)\b")
LONG = 10.0      # seconds: a longer "sentence" is run-on speech, so only the clause is muted


def inside(t0, t1, spans):
    """How much of [t0, t1] lies in the spans (0..1)."""
    if t1 <= t0:
        return 0.0
    return sum(max(0.0, min(t1, b) - max(t0, a)) for a, b in spans) / (t1 - t0)


def find_reads(words, breaks):
    """Sponsor lines in the program (not in the ad breaks). Each is {start, end, text, context,
    kind}: "billboard" (an ad: muted), "credit" (a feature's sponsor named: left on) or
    "segment" (a brand in a segment's name: left on). A billboard split over two sentences
    ("The NFL." "Powered by Verizon 5G.") is found across the split."""
    sents = sentences(words)
    out = []
    for n, ws in enumerate(sents):
        if inside(ws[0][0], ws[-1][1], breaks) > 0.5:
            continue
        prev = sents[n - 1] if n and ws[0][0] - sents[n - 1][-1][1] < 1.5 else []
        tail = prev[-4:]
        full = text_of(ws)
        joined = text_of(tail + ws)
        starts = np.cumsum([0] + [len(w[2]) + 1 for w in tail + ws])   # character offset of each word
        m = BILLBOARD.search(joined)
        if m and m.end() > starts[len(tail)]:                          # it reaches into this sentence
            i = int(np.searchsorted(starts, m.start(), "right") - 1)
            j = int(np.searchsorted(starts, m.end() - 1, "right") - 1)
            span = (prev if i < len(tail) else []) + ws
            if span[-1][1] - span[0][0] > LONG:
                span = clause(tail + ws, i, j)
            out.append({"start": span[0][0], "end": span[-1][1], "text": text_of(span), "context": full[:400],
                        "kind": "billboard"})
            continue
        m = CREDIT.search(full)
        if m or SEGMENT.search(full):
            out.append({"start": ws[0][0], "end": ws[-1][1], "text": full[:300], "context": full[:400],
                        "kind": "credit" if m else "segment"})
    # a billboard that also reached back into the previous sentence replaces that one's entry
    keep = []
    for r in out:
        if keep and r["kind"] == "billboard" and keep[-1]["kind"] != "billboard" and r["start"] <= keep[-1]["start"]:
            keep.pop()
        keep.append(r)
    return keep


# --- fingerprints ---------------------------------------------------------------------

def fingerprint(samples):
    """(bits uint32, loudness dB float16), FPS a second."""
    n = max(0, (len(samples) - WIN) // HOP + 1)
    edges = np.geomspace(300, 3000, 34)
    freqs = np.fft.rfftfreq(WIN, 1 / RATE)
    band = np.searchsorted(edges, freqs) - 1        # band of each FFT bin (-1 / 33: outside)
    keep = (band >= 0) & (band < 33)
    win = np.hanning(WIN).astype(np.float32)
    energy = np.empty((n, 33), np.float32)
    loud = np.empty(n, np.float32)
    step = 20000
    for a in range(0, n, step):
        b = min(n, a + step)
        idx = np.arange(a, b)[:, None] * HOP + np.arange(WIN)[None, :]
        spec = np.abs(np.fft.rfft(samples[idx] * win, axis=1)) ** 2
        e = np.zeros((b - a, 33), np.float32)
        for k in range(33):
            e[:, k] = spec[:, keep & (band == k)].sum(axis=1)
        energy[a:b] = e
        loud[a:b] = 10 * np.log10(spec.mean(axis=1) + 1e-12)
    d = energy[:, :-1] - energy[:, 1:]                 # across frequency
    bit = np.zeros((n, 32), bool)
    bit[1:] = (d[1:] - d[:-1]) > 0                     # and its change over time
    bits = np.packbits(bit, axis=1, bitorder="little").view("<u4")[:, 0]
    return bits, loud.astype(np.float16)


def matches(prints, min_len=2.0, max_count=120, quiet=-75.0, min_hits=10):
    """Stretches of sound heard more than once. prints: [(name, bits, loud)].
    Returns [(i, a0, a1, j, b0, b1, hits)]: recording i from a0 to a1 (seconds) sounds like
    recording j from b0 to b1.

    Two airings of the same sound differ in about 1 bit in 8, so whole 32-bit values rarely
    match; each 16-bit half matches about 1 time in 10. That's plenty when the matches line up
    (same time offset) over a few seconds, which chance almost never does."""
    offs = np.cumsum([0] + [len(p[1]) for p in prints])
    bits = np.concatenate([p[1] for p in prints])
    loud = np.concatenate([p[2] for p in prints]).astype(np.float32)
    rec = np.repeat(np.arange(len(prints)), [len(p[1]) for p in prints])
    ok = loud > quiet
    pa, pb = [], []
    for half in (bits & 0xFFFF, bits >> 16):
        idx = np.nonzero(ok)[0]
        vals = half[idx].astype(np.int64)
        order = np.lexsort((idx, vals))
        idx, vals = idx[order], vals[order]
        _, counts = np.unique(vals, return_counts=True)
        rare = ~np.repeat(counts > max_count, counts)   # values heard everywhere don't count
        idx, vals = idx[rare], vals[rare]
        for k in range(1, max_count):
            same = vals[k:] == vals[:-k]
            if not same.any():
                break
            a, b = idx[:-k][same], idx[k:][same]
            far = (rec[a] != rec[b]) | (b - a > 10 * FPS)    # not the same moment in one recording
            pa.append(a[far])
            pb.append(b[far])
    a, b = np.concatenate(pa), np.concatenate(pb)
    diag = (b - a + 1) // 2                                 # one line, allowing a frame of jitter
    order = np.lexsort((a, diag))
    a, b, diag = a[order], b[order], diag[order]
    out = []
    brk = np.nonzero((np.diff(diag) != 0) | (np.diff(a) > 2 * FPS))[0] + 1
    lo_, hi_ = np.r_[0, brk], np.r_[brk, len(a)]
    keep = (hi_ - lo_) >= min_hits
    for lo, hi in zip(lo_[keep], hi_[keep]):
        hits = hi - lo
        a0, a1 = a[lo], a[hi - 1]
        if (a1 - a0) < min_len * FPS or hits < 0.1 * (a1 - a0):
            continue
        b0, b1 = b[lo], b[hi - 1]
        i, j = rec[a0], rec[b0]
        if rec[a1] != i or rec[b1] != j:
            continue
        out.append((int(i), (a0 - offs[i]) / FPS, (a1 - offs[i]) / FPS,
                    int(j), (b0 - offs[j]) / FPS, (b1 - offs[j]) / FPS, int(hits)))
    return same_broadcast(out)


def same_broadcast(found, most=180.0):
    """Drop matches between two recordings of the same broadcast (a doubleheader's second
    game recorded while the first was still on): many minutes matching at one time offset,
    which no repeated sound does."""
    total = {}
    for i, a0, a1, j, b0, b1, _ in found:
        key = (i, j, round(b0 - a0))
        total[key] = total.get(key, 0) + (a1 - a0)
    near = lambda i, j, d: max(total.get((i, j, d + k), 0) for k in (-1, 0, 1))
    return [m for m in found if near(m[0], m[3], round(m[4] - m[1])) <= most]


def edge_matches(prints, breaks, span=12.0, w=2.0, close=0.3):
    """Sounds shared by break edges (the league's jingle, often under the commentary, so too
    faint for matches()). Compares the sound around every break start and end with every
    other's, sliding over each other. prints: [(name, bits, loud)], breaks: {name: [(a, b)]}.
    Returns matches like matches(), for pairs whose best 2-second stretch differs in under
    `close` of its bits (unrelated sound: about 0.4)."""
    win, half = int(w * FPS), int(span * FPS)
    edges = []   # (recording index, centre frame, bits around it)
    for r, (name, bits, _) in enumerate(prints):
        for a, b in breaks.get(name, []):
            for t in (a, b):
                c = int(t * FPS)
                if c - half >= 0 and c + half <= len(bits):
                    edges.append((r, c, bits[c - half:c + half]))
    out = []
    n = 2 * half - win + 1
    for x in range(len(edges)):
        rx, cx, A = edges[x]
        for y in range(x + 1, len(edges)):
            ry, cy, B = edges[y]
            if rx == ry and abs(cx - cy) < 2 * half:
                continue
            D = np.bitwise_count(A[:, None] ^ B[None, :]).astype(np.int16)   # every frame against every frame
            S = np.zeros((n, n), np.int32)
            for k in range(win):                                              # 2-second sums along each diagonal
                S += D[k:k + n, k:k + n]
            i, j = np.unravel_index(int(S.argmin()), S.shape)
            ber = S[i, j] / (32 * win)
            if ber < close:
                ta = (cx - half + i) / FPS
                tb = (cy - half + j) / FPS
                out.append((rx, ta, ta + w, ry, tb, tb + w, int(round((1 - ber) * 100))))
    return out


def group(found, names):
    """Matches joined into sounds: each a list of places [(name, start, end)] it was heard."""
    places = []   # (rec, start, end)
    links = []
    for i, a0, a1, j, b0, b1, _ in found:
        places += [(i, a0, a1), (j, b0, b1)]
        links.append((len(places) - 2, len(places) - 1))
    parent = list(range(len(places)))

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(x, y):
        parent[root(x)] = root(y)

    for x, y in links:
        join(x, y)
    # the same stretch of one recording, matched more than once: one place
    order = sorted(range(len(places)), key=lambda k: places[k][:2])
    for p, q in zip(order, order[1:]):
        (ri, s0, e0), (rj, s1, e1) = places[p], places[q]
        if ri != rj:
            continue
        if min(e0, e1) - s1 >= 0.6 * min(e0 - s0, e1 - s1):
            join(p, q)
        elif max(e0 - s0, e1 - s1) <= 5 and s1 - s0 <= 4:   # short pieces of one sound at one edge
            join(p, q)
    sounds = {}
    for k, pl in enumerate(places):
        sounds.setdefault(root(k), []).append(pl)
    out = []
    for pls in sounds.values():
        merged = []
        for r, s, e in sorted(pls):
            if merged and merged[-1][0] == r and s <= merged[-1][2] + 1:
                merged[-1][2] = max(merged[-1][2], e)
            else:
                merged.append([r, s, e])
        out.append([(names[r], round(s, 1), round(e, 1)) for r, s, e in merged])
    return out


def where(start, end, breaks, edge=4.0):
    """Where a sound falls against the breaks: start (at a break's start), end (at a break's
    end), in (inside a break), out (in the program), across (over a break's edge)."""
    part = inside(start, end, breaks)
    for a, b in breaks:
        if part > 0.8 and a - edge <= start <= a + edge:
            return "start"
        if part > 0.8 and b - edge <= end <= b + edge:
            return "end"
        if (abs(end - a) <= edge or abs(start - b) <= edge) and part < 0.2:
            return "edge"
    if part > 0.8:
        return "in"
    if part < 0.2:
        return "out"
    return "across"


def guess(places):
    """What a repeated sound probably is, from where it's heard."""
    kinds = [p["where"] for p in places]
    n = len(kinds)
    length = np.median([p["end"] - p["start"] for p in places])
    near = sum(k in ("edge", "across") for k in kinds)
    if near >= max(2, n / 2) and length <= 20:
        return "jingle"
    if sum(k == "end" for k in kinds) >= max(2, n / 2):
        return "closer"
    if sum(k in ("in", "start", "end") for k in kinds) >= n / 2:
        return "ad"
    return "program"


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


# --- live: sponsor billboards as they're said -----------------------------------------

LIVE_WINDOW = float(os.environ.get("SPEECH_WINDOW", "10"))   # seconds of sound each pass looks at
LIVE_STEP = float(os.environ.get("SPEECH_STEP", "1"))        # seconds between passes
_live = {"count": 0, "unload": None}
_lock = threading.Lock()
_pool = None


def _worker():
    """One thread for every channel's passes: the model runs one pass at a time anyway."""
    global _pool
    if _pool is None:
        from concurrent.futures import ThreadPoolExecutor
        _pool = ThreadPoolExecutor(1, thread_name_prefix="speech")
    return _pool


def _unload():
    """Free the model's GPU memory once no channel is listening (after a minute, so a channel
    change doesn't reload it)."""
    global _model
    with _lock:
        if _live["count"] == 0:
            _model = None
            import gc
            gc.collect()


class Listener:
    """Live TV: mutes sponsor billboards in the commentary while the picture plays on.

    Every LIVE_STEP seconds the last LIVE_WINDOW seconds of sound are transcribed, and any
    billboard in them (find_reads) is muted from its first word to its last. A line is
    usually caught as its "brought to you by" is said, a second or two after it starts, so
    with viewers DELAY seconds behind there's time to mute it from the start; later passes
    carry the mute on to the end of the line as it's heard.

    rate: samples a second of what audio() gets. sync (replays): passes run in the calling
    thread, and a pass counts as taking the time it took, so results land when they would live."""

    def __init__(self, rate, now, delay, event=None, fmt=str, sync=False):
        self.rate, self.now, self.delay, self.sync = rate, now, delay, sync
        self.event, self.fmt = event or (lambda m: None), fmt
        self.buf, self.buf_t = np.zeros(0, np.float32), None   # recent sound, and the time of its first sample
        self.mutes = []      # [start, end, text]
        self.job, self.last = None, None
        self.late = 0
        self.closed = False
        with _lock:
            _live["count"] += 1
            if _live["unload"]:
                _live["unload"].cancel()
                _live["unload"] = None

    def audio(self, t, samples, breaks=()):
        """Sound starting at t (int16). breaks: the ad breaks being muted (no reads looked for there)."""
        x = samples.astype(np.float32) / 32768
        if self.rate != RATE:
            x = np.interp(np.arange(0, len(x) * RATE / self.rate) * self.rate / RATE, np.arange(len(x)), x).astype(np.float32)
        if self.buf_t is None or abs(self.buf_t + len(self.buf) / RATE - t) > 0.5:   # a gap: start over
            self.buf, self.buf_t = x, t
        else:
            self.buf = np.concatenate([self.buf, x])
        extra = len(self.buf) - int((LIVE_WINDOW + 2) * RATE)
        if extra > 0:
            self.buf, self.buf_t = self.buf[extra:], self.buf_t + extra / RATE
        end = self.buf_t + len(self.buf) / RATE
        if self.job is not None:
            if not self.job.done():
                return
            self._result(*self.job.result())
            self.job = None
        if self.last is not None and end - self.last < LIVE_STEP:
            return
        if any(a <= end - 1 and end - 1 < b for a, b in breaks):
            return   # in an ad break: already muted
        self.last = end
        n = int(min(LIVE_WINDOW, len(self.buf) / RATE) * RATE)
        if n < RATE * 2:
            return
        t0, chunk, spans = end - n / RATE, self.buf[-n:].copy(), [tuple(b) for b in breaks]
        if self.sync:
            self._result(*self._pass(t0, chunk, spans))
        else:
            self.job = _worker().submit(self._pass, t0, chunk, spans)

    def _pass(self, t0, chunk, breaks):
        began = time.monotonic()
        with _lock:
            m = model()
        segs, _ = m.transcribe(chunk, language="en", word_timestamps=True, vad_filter=True,
                               condition_on_previous_text=False, beam_size=1)
        words = [[t0 + w.start, t0 + w.end, w.word.strip()] for s in segs for w in (s.words or [])]
        reads = [r for r in find_reads(words, breaks) if r["kind"] == "billboard"]
        return reads, time.monotonic() - began

    def _result(self, reads, took):
        # Viewers have heard everything up to (now - delay); one second more goes for the
        # decoding the detector's clock runs behind live.
        floor = self.now() + (took if self.sync else 0) - (self.delay - 1)
        for r in reads:
            a, b = r["start"] - 0.4, r["end"] + 0.3   # word times from a short window run a little late
            if b <= floor:
                continue
            same = next((m for m in self.mutes if a <= m[1] + 1 and b >= m[0] - 1), None)
            if same:
                if b > same[1]:
                    same[1] = b
                    same[2] = r["text"] if len(r["text"]) > len(same[2]) else same[2]
                continue
            start = max(a, floor)
            heard = start - a
            if heard > 0.3:
                self.late += 1
            self.mutes.append([start, b, r["text"]])
            self.mutes = self.mutes[-50:]
            self.event(f"sponsor line at {self.fmt(a)}, muted"
                       + (f" (first {heard:.1f}s heard)" if heard > 0.3 else "") + f": {r['text'][:120]}")

    def muted(self, t):
        return any(a <= t < b for a, b, _ in self.mutes)

    def close(self):
        if self.closed:
            return
        self.closed = True
        with _lock:
            _live["count"] -= 1
            if _live["count"] == 0:
                _live["unload"] = threading.Timer(60, _unload)
                _live["unload"].daemon = True
                _live["unload"].start()
