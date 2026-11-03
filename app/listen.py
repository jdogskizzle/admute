"""Listening to a recording's sound: what's said, and which sounds come back.

- Speech: the commentary as words with times (faster-whisper on the GPU). Sponsor reads
  are whole sentences like "...is brought to you by...": admute could mute just those while
  the picture plays on. A brand on a segment's name ("Verizon halftime") is left alone.
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


# Phrases that make a sentence a sponsor read.
READ = re.compile(r"\b(brought to you by|presented by|sponsored by|powered by|courtesy of|in partnership with"
                  r"|(?:is|are) the official \w+(?: \w+){0,3} of|official (?:sponsor|partner)"
                  r"|proud (?:sponsor|partner)|(?:is|are) proud to)\b", re.I)
# A brand on a segment's name, e.g. "the Verizon halftime report": left alone.
SEGMENT = re.compile(r"\b(?:[A-Z][\w&'.-]+ ){1,3}(?:halftime|skycam|sky cam|report|update|kickoff"
                     r"|pregame|postgame|starting lineups?|player of the game|play of the game|drive chart|red zone)\b")
SHORT = 3.0      # seconds: a sponsor phrase in a shorter sentence is just a mention


def inside(t0, t1, spans):
    """How much of [t0, t1] lies in the spans (0..1)."""
    if t1 <= t0:
        return 0.0
    return sum(max(0.0, min(t1, b) - max(t0, a)) for a, b in spans) / (t1 - t0)


def find_reads(words, breaks):
    """Sponsor reads in the program (not in the ad breaks). Each is {start, end, text, context,
    kind}: "read" (would be muted) or "mention" (a sponsor phrase in a short line, left on),
    and "segment" for a brand on a segment's name (left on)."""
    out = []
    for ws in sentences(words):
        if inside(ws[0][0], ws[-1][1], breaks) > 0.5:
            continue
        full = text_of(ws)
        starts = np.cumsum([0] + [len(w[2]) + 1 for w in ws])   # character offset of each word
        m = READ.search(full)
        if m:
            i = int(np.searchsorted(starts, m.start(), "right") - 1)
            j = int(np.searchsorted(starts, m.end() - 1, "right") - 1)
            part = clause(ws, i, j)
            kind = "read" if part[-1][1] - part[0][0] >= SHORT else "mention"
        elif SEGMENT.search(full):
            part, kind = ws, "segment"
        else:
            continue
        out.append({"start": part[0][0], "end": part[-1][1], "text": text_of(part), "context": full[:400],
                    "kind": kind})
    return out


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
    return out


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
