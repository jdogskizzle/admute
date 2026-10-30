"""Run the ad detector, and muting, on a recording instead of live TV.

  replay survey REC                 scan a whole recording: per-second timeline, list of
                                    breaks, timeline picture and picture sheets (1 frame/10s)
  replay render REC START LENGTH    write a clip with muting applied to the review library

START and LENGTH are like 1:20:00 and 10:00. REC is a path inside the container; the ./replay
wrapper translates host paths using the volumes in docker-compose.yaml.

Replays use the same pipeline as live TV, clocked by the recording's own timestamps. The
detector only gets to look DELAY seconds ahead (minus REPLAY_LAG for decoding), exactly
like live, so a clip sounds the way it would have if you'd watched it live.
"""

import argparse
import asyncio
import csv
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from detect import AdDetector, write_png
from pipeline import FFMPEG, DetectorProc, Pipeline
from ts import PKT, SilentFrames

DATA = Path(os.environ.get("DATA", "/data")) / "replay"
REVIEW = Path(os.environ.get("REVIEW", "/review"))
DELAY = float(os.environ.get("DELAY", "10"))
LAG = float(os.environ.get("REPLAY_LAG", "1.0"))   # live decoding runs about this far behind the tuner
GPU = os.environ.get("DETECT_GPU", "1") == "1"
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
          "drawtext=text='%{pts\\:hms}':x=4:y=4:fontsize=16:fontcolor=white:box=1:boxcolor=black@0.6,"
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

    def make():
        d = AdDetector(name, path.stem, work, fmt=hms, keep=None)
        d.now = lambda: (d.last_t or 0) + LAG
        return d

    if not (work / "logos" / f"{name}.npz").exists():
        print(f"  learning the logo from the first {LEARN_PASS // 60} minutes", flush=True)
        det = make()
        await detect_file(path, det, until=LEARN_PASS, label="learning: ")
        det.close()
        if det.mask is None:
            print("  no logo found in the first pass; the main pass will keep trying", flush=True)

    det = make()
    took = await detect_file(path, det, label="")
    det.close()
    sheet_proc.wait()

    csv_path = Path(det._csv_file.name)
    shutil.copy(csv_path, work / "timeline.csv")
    timeline(work / "timeline.csv", work / "timeline.png")
    breaks = [(s, e) for s, e in det.intervals]
    lines = [f"{path.name}", f"length {hms(det.last_t or 0)}, surveyed in {took / 60:.1f} min",
             f"logo: {det.corner or 'not found'}, usual score {det.level:.2f}", "",
             f"{len(breaks)} breaks muted:"]
    for s, e in breaks:
        end = e if e != float("inf") else (det.last_t or s)
        lines.append(f"  {hms(s)} - {hms(end)}  ({end - s:.0f}s)")
    (work / "breaks.txt").write_text("\n".join(lines) + "\n")
    print("\n" + "\n".join(lines), flush=True)


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


def main():
    logging.basicConfig(level=logging.INFO, format="  %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("survey", help="scan a whole recording")
    s.add_argument("recording")
    s.add_argument("--name", help="short name for the results folder (default: from the file name)")
    r = sub.add_parser("render", help="render a clip with muting into the review library")
    r.add_argument("recording")
    r.add_argument("start", help="e.g. 1:20:00")
    r.add_argument("length", help="e.g. 10:00")
    r.add_argument("--name")
    r.add_argument("--warmup", type=float, default=30, help="seconds of detection before the clip starts")
    args = ap.parse_args()
    asyncio.run(survey(args) if args.cmd == "survey" else render(args))


if __name__ == "__main__":
    main()
