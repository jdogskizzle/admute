"""Run the ad detector, and muting, on a recording instead of live TV.

  replay survey REC                 scan a whole recording: per-second timeline, list of
                                    breaks, timeline picture and picture sheets (1 frame/10s)
  replay render REC START LENGTH    write a clip with muting applied to the review library
  replay listen REC                 transcribe the commentary and fingerprint the sound (GPU)
  replay sounds                     find sponsor reads and repeated sounds (jingles, promos,
                                    ads) in every listened-to recording

START and LENGTH are like 1:20:00 and 10:00. REC is a path inside the container; the ./replay
wrapper translates host paths using the volumes in docker-compose.yaml.

Replays use the same pipeline as live TV, clocked by the recording's own timestamps. The
detector only gets to look DELAY seconds ahead (minus REPLAY_LAG for decoding), exactly
like live, so a clip sounds the way it would have if you'd watched it live.
"""

import argparse
import asyncio
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from detect import H, MASK_T, SETTINGS, W, AdDetector, edge_map, write_png
from pipeline import FFMPEG, DetectorProc, Pipeline
from ts import PKT, SilentFrames

DATA = Path(os.environ.get("DATA", "/data")) / "replay"
REVIEW = Path(os.environ.get("REVIEW", "/review"))
DELAY = float(os.environ.get("DELAY", "10"))
LAG = float(os.environ.get("REPLAY_LAG", "1.0"))   # live decoding runs about this far behind the tuner
# Recordings decode on the CPU by default: MPEG-2 is cheap, and several surveys at once
# would queue on the GPU's single video decoder. REPLAY_GPU=1 decodes on the GPU instead.
GPU = os.environ.get("REPLAY_GPU", "0") == "1"
LEARN_PASS = 15 * 60                               # survey: learn the logo from this much video first

log = logging.getLogger("admute")


def hms(t):
    t = max(0, int(round(t)))
    return f"{t // 3600}:{t // 60 % 60:02d}:{t % 60:02d}"


def parse_time(s):
    total = 0.0
    for part in s.split(":"):
        total = total * 60 + float(part)
    return total


def slug(path):
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("_")[:80]


# --- the recording's clock --------------------------------------------------

def pcrs(data):
    """(pid, seconds) for every packet in data that carries a PCR."""
    n = len(data) // PKT
    if not n:
        return []
    a = np.frombuffer(data, np.uint8, n * PKT).reshape(n, PKT)
    has = (a[:, 0] == 0x47) & ((a[:, 3] & 0x20) != 0) & (a[:, 4] >= 7) & ((a[:, 5] & 0x10) != 0)
    out = []
    for r in a[has]:
        base = (int(r[6]) << 25) | (int(r[7]) << 17) | (int(r[8]) << 9) | (int(r[9]) << 1) | (int(r[10]) >> 7)
        out.append((((int(r[1]) & 0x1F) << 8) | int(r[2]), base / 90000.0))
    return out


class PcrClock:
    """Seconds since the start of the recording, from the stream's PCR. Smooths over wraps and splices."""

    def __init__(self, origin, pid):
        self.pid, self.last, self.offset, self.t = pid, None, -origin, 0.0

    def update(self, data):
        for pid, v in pcrs(data):
            if pid != self.pid:
                continue
            if self.last is not None and not -1 < v - self.last < 5:
                self.offset += self.last - v + 0.04
            self.last = v
            self.t = v + self.offset
        return self.t


def aligned(f, off):
    """Nearest offset at or after off where three packets in a row start with a sync byte."""
    f.seek(off)
    data = f.read(PKT * 64)
    for i in range(min(PKT, len(data))):
        if all(i + k * PKT < len(data) and data[i + k * PKT] == 0x47 for k in range(3)):
            return off + i
    return off


def pcr_at(f, off):
    off = aligned(f, off)
    f.seek(off)
    found = pcrs(f.read(PKT * 4000))
    return found[0] if found else (None, None)


def seek(path, target):
    """Byte offset where the recording reaches `target` seconds, plus the clock origin and PCR pid."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        pid, origin = pcr_at(f, 0)
        if origin is None:
            sys.exit("no PCR timestamps in this file; is it an MPEG-TS recording?")
        _, last = pcr_at(f, max(0, size - PKT * 8000))
        rate = size / max(1.0, last - origin)
        off = int(target * rate)
        for _ in range(6):
            off = min(max(0, off), size - PKT * 8000)
            _, here = pcr_at(f, off)
            err = target - (here - origin)
            if abs(err) < 0.5:
                break
            off += int(err * rate)
        return aligned(f, max(0, off)), origin, pid


# --- survey -----------------------------------------------------------------

async def detect_file(path, det, until=None, label=""):
    """Feed a recording (from its start) through the detector as fast as the GPU decodes."""
    proc = DetectorProc(det, GPU)
    await proc.start()
    began, shown = time.monotonic(), time.monotonic()
    with open(path, "rb") as f:
        while data := f.read(PKT * 5600):
            await proc.feed_wait(data, 0.0)
            pos = det.last_t or 0
            if time.monotonic() - shown > 30:
                shown = time.monotonic()
                print(f"  {label}{hms(pos)} done ({pos / (shown - began):.0f}x real time)", flush=True)
            if until and pos >= until:
                break
    await proc.finish()
    return time.monotonic() - began


def sheets(path, out):
    """Picture sheets: one frame every 10 s, 60 per sheet (10 minutes each), timestamped."""
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("sheet_*.jpg"):
        old.unlink()
    hw = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"] if GPU else []
    scale = "scale_cuda=256:144,hwdownload,format=nv12" if GPU else "scale=256:144"
    vf = (f"setpts=PTS-STARTPTS,fps=1/10,{scale},"
          "drawtext=text='%{pts\\:hms}':x=4:y=h-th-6:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.6,"
          "tile=6x10")
    return subprocess.Popen([FFMPEG, "-hide_banner", "-loglevel", "error", *hw, "-i", str(path),
                             "-vf", vf, "-q:v", "4", str(out / "sheet_%03d.jpg")])


def timeline(csv_path, png_path):
    """One pixel column per 5 s: logo score (top), ad state, black frames, silence."""
    rows = list(csv.DictReader(open(csv_path)))
    cols = (len(rows) + 4) // 5
    img = np.zeros((70, max(cols, 1)), np.uint8)
    for i in range(cols):
        chunk = rows[i * 5:(i + 1) * 5]
        scores = [float(r["score"]) for r in chunk if r["score"]]
        if scores:
            h = int(min(scores) * 40)
            img[40 - h:40, i] = 200
        if any(r["state"] == "ad" for r in chunk):
            img[44:56, i] = 255
        if any(r["black"] == "1" for r in chunk):
            img[58:63, i] = 150
        if any(r["silent"] == "1" for r in chunk):
            img[65:70, i] = 100
    for m in range(0, cols, 12 * 5):  # tick every 5 minutes
        img[40:43, m] = 255
    write_png(png_path, img)


async def survey(args):
    path = Path(args.recording)
    name = args.name or slug(path)
    work = DATA / name
    work.mkdir(parents=True, exist_ok=True)
    print(f"Surveying {path.name}\n  results in data/replay/{name}/", flush=True)
    sheet_proc = sheets(path, work / "sheets")

    if args.fresh:
        for old in (work / "logos").glob("*"):
            old.unlink()

    def make(trace=False):
        d = AdDetector(name, path.stem, work, fmt=hms, keep=None, trace=trace)
        d.now = lambda: (d.last_t or 0) + LAG
        return d

    if not (work / "logos" / f"{name}.npz").exists():
        print(f"  learning the logo from the first {LEARN_PASS // 60} minutes", flush=True)
        det = make()
        await detect_file(path, det, until=LEARN_PASS, label="learning: ")
        det.close()
        if det.mask is None:
            print("  no logo found in the first pass; the main pass will keep trying", flush=True)

    det = make(trace=True)
    took = await detect_file(path, det, label="")
    det.close()
    det.save_trace(work / "trace.npz")
    sheet_proc.wait()

    csv_path = Path(det._csv_file.name)
    shutil.copy(csv_path, work / "timeline.csv")
    timeline(work / "timeline.csv", work / "timeline.png")
    breaks = [(s, e) for s, e in det.intervals]
    lines = [f"{path.name}", f"length {hms(det.last_t or 0)}, surveyed in {took / 60:.1f} min",
             f"graphics: {det._describe() or 'not found'}", "",
             f"{len(breaks)} breaks muted:"]
    for s, e in breaks:
        end = e if e != float("inf") else (det.last_t or s)
        lines.append(f"  {hms(s)} - {hms(end)}  ({end - s:.0f}s)")
    (work / "breaks.txt").write_text("\n".join(lines) + "\n")
    (work / "result.json").write_text(json.dumps({
        "recording": str(path), "length": det.last_t or 0, "took": took, "finished": time.time(),
        "graphics": det._describe(), "settings": SETTINGS,
        "breaks": [[s, e if e != float("inf") else (det.last_t or s)] for s, e in breaks]}, indent=1))
    print("\n" + "\n".join(lines), flush=True)


# --- side-by-side ad layouts ---------------------------------------------------

LAYOUTS = DATA.parent / "layouts"


def frames_lo(path, start, length, fps):
    """Pictures at the detector's size (brightness only), decoded on the CPU."""
    cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-ss", f"{max(0, start):.2f}", "-i", str(path)]
    if length:
        cmd += ["-t", f"{length:.2f}"]
    cmd += ["-map", "0:v:0", "-vf", f"fps={fps},scale={2 * W}:{2 * H}:flags=area,format=gray",
            "-f", "rawvideo", "pipe:1"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    size, n = 4 * W * H, 0
    try:
        while len(buf := p.stdout.read(size)) == size:
            hi = np.frombuffer(buf, np.uint8).reshape(2 * H, 2 * W)
            yield start + n / fps, hi.reshape(H, 2, W, 2).mean(axis=(1, 3)).astype(np.int16)
            n += 1
    finally:
        p.kill()
        p.wait()


def labelled():
    """(name, recording path, labels) for every survey with labels."""
    for d in sorted(DATA.iterdir()):
        try:
            labels = json.loads((d / "labels.json").read_text())
            rec = json.loads((d / "result.json").read_text())["recording"]
        except (OSError, ValueError, KeyError):
            continue
        if Path(rec).exists():
            yield d.name, Path(rec), labels


def learn_layouts(args):
    """Learn side-by-side ad layouts from labels marked "sbs", and check each one against
    everything else in the labelled recordings so it won't fire on the game."""
    from concurrent.futures import ThreadPoolExecutor
    recs = list(labelled())
    examples = [(name, path, a[0], a[1]) for name, path, labels in recs
                for a in labels.get("ads", []) if len(a) > 2 and a[2] == "sbs" and a[1] - a[0] >= 8]
    if not examples:
        sys.exit("No side-by-side ads labelled yet. Mark some in the studio (Labels tab), then run this again.")
    print(f"Learning from {len(examples)} labelled side-by-side ads", flush=True)

    def edges_of(ex):  # skip the first and last second: the frame animates in and out
        name, path, s, e = ex
        return [edge_map(g) for _, g in frames_lo(path, s + 1, e - s - 2, FPS_LEARN)]

    with ThreadPoolExecutor(8) as pool:
        ex_edges = list(pool.map(edges_of, examples))

    # Group examples that share a frame, by how alike their average edges are.
    groups = []
    for ex, E in zip(examples, ex_edges):
        if not E:
            continue
        m = np.mean(E, axis=0)
        for g in groups:
            if np.corrcoef(g["mean"].ravel(), m.ravel())[0, 1] > 0.5:
                g["examples"].append(ex)
                g["edges"] += E
                g["mean"] = np.mean(g["edges"], axis=0)
                break
        else:
            groups.append({"examples": [ex], "edges": list(E), "mean": m})

    candidates = []
    for g in groups:
        mask = g["mean"] > MASK_T
        if mask.sum() < 150:
            print(f"  skipped a layout from {len(g['examples'])} example(s): too few steady edges", flush=True)
            continue
        pos = np.array([e[mask].mean() for e in g["edges"]])
        candidates.append({**g, "mask": mask, "level": float(np.median(pos)), "pos": pos})
    if not candidates:
        sys.exit("No usable layouts found.")

    # Everything that isn't a labelled side-by-side ad, one frame a second, must score lower.
    def negatives(rec):
        name, path, labels = rec
        sbs = [(a[0] - 2, a[1] + 2) for a in labels.get("ads", []) if len(a) > 2 and a[2] == "sbs"]
        out = []
        for t, g in frames_lo(path, 0, None, 1):
            if not any(s <= t <= e for s, e in sbs):
                ed = edge_map(g)
                out.append([ed[c["mask"]].mean() for c in candidates])
        print(f"  checked {len(out)} other frames of {name}", flush=True)
        return np.array(out).reshape(-1, len(candidates))

    print(f"Checking {len(candidates)} layout(s) against the rest of {len(recs)} recordings "
          "(a few minutes)", flush=True)
    with ThreadPoolExecutor(len(recs)) as pool:
        neg = np.concatenate(list(pool.map(negatives, recs)))

    LAYOUTS.mkdir(parents=True, exist_ok=True)
    for old in LAYOUTS.glob("*"):
        old.unlink()
    kept = 0
    for i, c in enumerate(candidates):
        lv = c["level"]
        pos_lo = float(np.percentile(c["pos"], 10)) / lv
        neg_hi = float(np.percentile(neg[:, i], 99.95)) / lv if len(neg) else 0.0
        where = ", ".join(sorted({f"{n[:40]} {hms(s)}" for n, _, s, _ in c["examples"]}))
        print(f"\nLayout {i + 1}: {len(c['examples'])} example(s): {where}\n  {int(c['mask'].sum())} edge pixels;"
              f" examples score {pos_lo:.2f}+ (relative), everything else up to {neg_hi:.2f}", flush=True)
        if pos_lo <= neg_hi + 0.1:
            print("  not used: it doesn't stand apart from the rest of the broadcast", flush=True)
            continue
        th = (pos_lo + neg_hi) / 2
        kept += 1
        name = f"layout-{kept}"
        np.savez_compressed(LAYOUTS / f"{name}.npz", mask=c["mask"], level=lv, threshold=th,
                            examples=np.array([f"{n} {s:.1f}-{e:.1f}" for n, _, s, e in c["examples"]]))
        img = (np.clip(c["mean"], 0, 1) * 140).astype(np.uint8)
        img[c["mask"]] = 255
        write_png(LAYOUTS / f"{name}.png", img)
        print(f"  saved as {name}, matches above {th:.2f}", flush=True)
    print(f"\n{kept} layout(s) saved. Survey again to see them in use.", flush=True)


FPS_LEARN = 5


# --- scene classifier -----------------------------------------------------------

def reviewed_scenes():
    """(name, features per second, ad or not per second) for every reviewed recording whose
    survey saved scene features."""
    out = []
    for d in sorted(DATA.iterdir()):
        try:
            labels = json.loads((d / "labels.json").read_text())
        except (OSError, ValueError):
            continue
        if not labels.get("reviewed") or not (d / "trace.npz").exists():
            continue
        with np.load(d / "trace.npz") as tr:
            if "scene_feats" not in tr or not len(tr["scene_sec"]):
                print(f"  {d.name}: surveyed before the scene check existed; survey it again to use it", flush=True)
                continue
            secs, feats = tr["scene_sec"], tr["scene_feats"].astype(np.float32)
            trace = {k: tr[k] for k in tr.files}
        ads = [a for a in labels.get("ads", [])]
        y = np.array([any(a[0] <= s + 0.5 < a[1] for a in ads) for s in secs])
        out.append((d.name, feats, y, trace, ads))
    return out


def train_scene(args):
    """Learn the scene classifier from every reviewed recording, after checking how well it
    does on each one when that one is left out of the training."""
    from detect import replay_trace
    from scene import Classifier
    games = reviewed_scenes()
    if not games:
        sys.exit("No reviewed recordings with scene features yet. Review a game in the studio, and survey it "
                 "again if it was surveyed before the scene check existed.")
    print(f"Training on {len(games)} reviewed recording(s): " + ", ".join(g[0] for g in games), flush=True)

    def score(breaks, ads, length):
        n = int(length) + 1
        m, a = np.zeros(n, bool), np.zeros(n, bool)
        for s, e in breaks:
            m[int(s):int(e)] = True
        for x in ads:
            a[int(x[0]):int(x[1])] = True
        wrong = 0
        for s, e in breaks:
            seg = a[int(s):max(int(s) + 1, int(e))]
            wrong += seg.mean() < 0.5
        return int((a & ~m).sum()), int((m & ~a).sum()), int(wrong)

    report = {}
    if len(games) > 1:
        print("\nEach recording scored by a classifier trained on the others:", flush=True)
        for i, (name, X, y, trace, ads) in enumerate(games):
            rest = [g for j, g in enumerate(games) if j != i]
            clf = Classifier.train(np.concatenate([g[1] for g in rest]), np.concatenate([g[2] for g in rest]))
            length = float(trace["frames"][-1, 0])
            old = score(replay_trace(trace, {}, classifier=None), ads, length)
            new = score(replay_trace(trace, {}, classifier=clf), ads, length)
            report[name] = {"before": old, "after": new}
            print(f"  {name[:60]}\n    logo rules:  ads heard {old[0]}s, game muted {old[1]}s, wrong mutes {old[2]}"
                  f"\n    scene check: ads heard {new[0]}s, game muted {new[1]}s, wrong mutes {new[2]}", flush=True)
    clf = Classifier.train(np.concatenate([g[1] for g in games]), np.concatenate([g[2] for g in games]))
    from scene import show_title
    titles = sorted({show_title(json.loads((DATA / g[0] / "result.json").read_text())["recording"].rsplit("/", 1)[-1][:-3])
                     for g in games})
    clf.info = {"trained_on": [g[0] for g in games], "titles": titles, "seconds": int(sum(len(g[2]) for g in games)),
                "made": time.time(), "held_out": report}
    clf.save()
    print(f"\nSaved the scene classifier ({clf.info['seconds']} seconds of video). It's used for: "
          + ", ".join(titles) + " (other programs keep the logo rules).", flush=True)
    print("Live TV picks it up the next time a channel opens; surveys and the studio use it right away.", flush=True)


# --- render -----------------------------------------------------------------

async def render(args):
    path = Path(args.recording)
    name = args.name or slug(path)
    start, length = parse_time(args.start), parse_time(args.length)
    end = start + length
    warm = min(args.warmup, start)
    work = DATA / name / "render"
    (work / "logos").mkdir(parents=True, exist_ok=True)
    learned = DATA / name / "logos" / f"{name}.npz"
    if learned.exists():
        shutil.copy(learned, work / "logos" / f"{name}.npz")
    else:
        print("No survey for this recording yet, so the logo starts unlearned (no muting for about "
              "5 minutes). Run survey first for a fair test.", flush=True)

    REVIEW.mkdir(parents=True, exist_ok=True)
    out_path = REVIEW / f"{name} {hms(start).replace(':', '.')} +{int(length // 60)}m.ts"
    off, origin, pid = seek(path, start - warm)
    clock = PcrClock(origin, pid)

    pipe = Pipeline(name, SilentFrames(FFMPEG))
    det = AdDetector(name, path.stem, work, fmt=hms, keep=None)
    det.now = lambda: (det.last_t or 0) + LAG
    pipe.detector = det
    proc = DetectorProc(det, GPU)
    await proc.start()

    written = 0

    def flush(cutoff, out):
        nonlocal written
        for e in pipe.take(cutoff):
            if start <= e.t < end:
                out.write(e.pkt)
                written += PKT

    began = time.monotonic()
    print(f"Rendering {hms(start)} to {hms(end)} of {path.name} (detector warms up from {hms(start - warm)})",
          flush=True)
    with open(path, "rb") as f, open(out_path, "wb") as out:
        f.seek(off)
        while data := f.read(PKT * 350):
            t = clock.update(data)
            fed = pipe.ingest(data, t)
            if fed:
                await proc.feed_wait(fed, t)
            if det.last_t is not None:
                # Live, a packet goes out DELAY s after it arrives, when decoding has reached
                # about arrival + DELAY - LAG. Hold packets back until the detector is that far.
                flush(min(t, det.last_t - DELAY + LAG), out)
            if t >= end + DELAY + 2:
                break
        await proc.finish()
        flush(float("inf"), out)
    det.close()

    took = time.monotonic() - began
    print(f"\nWrote {out_path.name} ({written / 1e6:.0f} MB) in {took:.0f}s", flush=True)
    clip = [(max(s, start), min(e, end)) for s, e in det.intervals if e > start and s < end]
    if clip:
        print("Muted in this clip (clip time / recording time):")
        for s, e in clip:
            print(f"  {hms(s - start)} - {hms(e - start)}   ({hms(s)} - {hms(e)}, {e - s:.0f}s)")
    else:
        print("Nothing was muted in this clip.")


# --- listening: sponsor reads and repeated sounds -------------------------------------

def breaks_of(work):
    """Where the ads are in a recording: its labels once reviewed, else its survey's mutes."""
    labels = read_json(work / "labels.json") or {}
    if labels.get("ads") and labels.get("reviewed"):
        return [tuple(a[:2]) for a in labels["ads"]], "labels"
    res = read_json(work / "result.json") or {}
    return [tuple(b) for b in res.get("breaks", [])], "survey" if res else "none"


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def listen(args):
    import listen as L
    path = Path(args.recording)
    name = args.name or slug(path)
    work = DATA / name
    work.mkdir(parents=True, exist_ok=True)
    began = time.monotonic()
    print(f"Listening to {path.name}\n  reading the sound...", flush=True)
    samples = L.audio(path)
    length = len(samples) / L.RATE
    print(f"  {hms(length)} of sound; fingerprinting...", flush=True)
    bits, loud = L.fingerprint(samples)
    np.savez_compressed(work / "prints.npz", bits=bits, loud=loud, fps=L.FPS)
    if args.prints:
        print(f"\nFingerprinted in {(time.monotonic() - began) / 60:.1f} min", flush=True)
        return
    print("  transcribing...", flush=True)
    shown = [0.0]

    def progress(t):
        if t - shown[0] >= 600:
            shown[0] = t
            print(f"  {hms(t)} done ({t / max(1, time.monotonic() - began):.0f}x real time)", flush=True)

    words = L.transcribe(samples, progress)
    L.save_json(work / "speech.json", {"recording": str(path), "length": length, "words": words,
                                       "finished": time.time()})
    print(f"\nListened in {(time.monotonic() - began) / 60:.1f} min: {len(words)} words", flush=True)


def sounds(args):
    """Sponsor reads per recording (reads.json) and repeated sounds across all (data/sounds)."""
    import listen as L
    prints, names, breaks = [], [], {}
    for d in sorted(DATA.iterdir()):
        speech = read_json(d / "speech.json")
        if speech is None:
            continue
        br, src = breaks_of(d)
        breaks[d.name] = br
        reads = L.find_reads(speech["words"], br)
        L.save_json(d / "reads.json", {"reads": reads, "breaks_from": src, "found": time.time()})
        n = {k: sum(r["kind"] == k for r in reads) for k in ("billboard", "credit", "segment")}
        print(f"{d.name}: {n['billboard']} sponsor billboards (muted), {n['credit']} sponsor credits and "
              f"{n['segment']} brand-named segments (left on); breaks from {src}", flush=True)
        if (d / "prints.npz").exists():
            with np.load(d / "prints.npz") as p:
                prints.append((d.name, p["bits"], p["loud"]))
            names.append(d.name)
    if len(prints) < 1:
        print("Nothing listened to yet: run replay listen first.")
        return
    print(f"\nLooking for repeated sounds in {len(prints)} recordings...", flush=True)
    found = L.matches(prints)
    edge = L.edge_matches(prints, breaks)
    print(f"  {len(found)} repeats, {len(edge)} matches between break edges", flush=True)
    found += edge
    words = {n: (read_json(DATA / n / "speech.json") or {}).get("words", []) for n in names}
    out = []
    for places in L.group(found, names):
        pl = [{"name": n, "start": s, "end": e, "where": L.where(s, e, breaks.get(n, []))} for n, s, e in places]
        first = pl[0]
        text = " ".join(w[2] for w in words[first["name"]] if first["start"] <= w[0] < first["end"])
        out.append({"id": f"{first['name']}@{first['start']:.0f}", "guess": L.guess(pl), "places": pl,
                    "length": round(float(np.median([p["end"] - p["start"] for p in pl])), 1),
                    "recordings": len({p["name"] for p in pl}), "text": text[:300]})
    out.sort(key=lambda s: (-len(s["places"]), -s["length"]))
    L.save_json(DATA.parent / "sounds" / "sounds.json", {"sounds": out, "recordings": names, "found": time.time()})
    kinds = {}
    for s in out:
        kinds[s["guess"]] = kinds.get(s["guess"], 0) + 1
    print(f"{len(out)} repeated sounds: " + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items())), flush=True)


def main():
    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("survey", help="scan a whole recording")
    s.add_argument("recording")
    s.add_argument("--name", help="short name for the results folder (default: from the file name)")
    s.add_argument("--fresh", action="store_true", help="forget what an earlier survey learned and learn again")
    r = sub.add_parser("render", help="render a clip with muting into the review library")
    r.add_argument("recording")
    r.add_argument("start", help="e.g. 1:20:00")
    r.add_argument("length", help="e.g. 10:00")
    r.add_argument("--name")
    r.add_argument("--warmup", type=float, default=30, help="seconds of detection before the clip starts")
    sub.add_parser("layouts", help="learn side-by-side ad layouts from labelled examples")
    sub.add_parser("train", help="learn the scene classifier from reviewed recordings")
    li = sub.add_parser("listen", help="transcribe and fingerprint a recording's sound")
    li.add_argument("recording")
    li.add_argument("--name")
    li.add_argument("--prints", action="store_true", help="only redo the fingerprints")
    sub.add_parser("sounds", help="find sponsor reads and repeated sounds in listened-to recordings")
    args = ap.parse_args()
    if args.cmd == "listen":
        return listen(args)
    if args.cmd == "sounds":
        return sounds(args)
    if args.cmd == "layouts":
        return learn_layouts(args)
    if args.cmd == "train":
        return train_scene(args)
    asyncio.run(survey(args) if args.cmd == "survey" else render(args))


if __name__ == "__main__":
    main()
