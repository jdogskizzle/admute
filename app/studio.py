"""admute studio: a web dashboard for checking and tuning the ad detector on recordings.

- Lists the recordings, runs surveys and clip renders (app/replay.py) as background jobs.
- Shows a survey's timeline (graphic scores, playing surface, black frames, silence, what
  was muted) next to the video, which it streams to the browser from any point.
- Labels: mark where the ads really are, then score the detector against them.
- Tuning: replays a survey's saved trace with other settings in about a second, no video
  decoding, and scores the result on every labelled recording.
- Live: what the admute proxy is doing right now.
- Sounds: sponsor reads found in the commentary and sounds that repeat across recordings
  (jingles, network promos, ads), to confirm what each one is.

Sign-in is formlogin (password in /data/studio/login.json, set with python -m formlogin passwd). Everything machine-specific
comes from the environment set in docker-compose.yaml.
"""

import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

import formlogin
import formlogin.httpserver

sys.path.insert(0, str(Path(__file__).parent))
from detect import SETTINGS, replay_trace  # noqa: E402
from pipeline import FFMPEG  # noqa: E402
from replay import DATA as REPLAY, REVIEW, hms, slug  # noqa: E402

PORT = int(os.environ.get("PORT", "8000"))
RECORDINGS = Path(os.environ.get("RECORDINGS", "/recordings"))
STATE = Path(os.environ.get("DATA", "/data")) / "studio"
LIVE_URL = os.environ.get("LIVE_URL", "").rstrip("/")  # the admute proxy, for the Live tab
GPU = os.environ.get("DETECT_GPU", "1") == "1"
MAX_JOBS = int(os.environ.get("STUDIO_MAX_JOBS", "3"))
FFPROBE = str(Path(FFMPEG).with_name("ffprobe"))
WEB = Path(__file__).parent / "studio"
REPLAY_PY = Path(__file__).parent / "replay.py"
# Settings the tuning panel can change, with the range its sliders cover.
TUNABLE = {
    "LOGO_ON": (0.05, 0.9, 0.01), "LOGO_OFF": (0.02, 0.8, 0.01), "AD_MIN": (1, 10, 0.5),
    "SHOW_MIN": (1, 10, 0.5), "MERGE": (0, 60, 1), "ARM": (0, 60, 1), "SNAP": (0, 6, 0.5),
    "SURFACE": (0, 1, 1), "SURFACE_AT": (0.1, 0.9, 0.05), "SURFACE_HOLD": (0, 5, 0.5),
    "SCENE": (0, 1, 1), "SCENE_ON": (0.5, 0.98, 0.02), "SCENE_OFF": (0.05, 0.6, 0.05), "SCENE_WINDOW": (3, 20, 1),
    "SPLICE": (-1, 1, 1), "SPLICE_WAIT": (5, 60, 1), "LAYOUT": (0, 1, 1), "LAYOUT_HOLD": (0, 3, 0.2),
    "AD_MAX": (60, 600, 10),
}


# --- recordings and surveys ----------------------------------------------------

def recordings():
    """Every .ts recording, newest first, with what's known about it."""
    out = []
    for p in RECORDINGS.rglob("*.ts"):
        try:
            st = p.stat()
        except OSError:
            continue
        name = slug(p)
        work = REPLAY / name
        res = read_json(work / "result.json")
        labels = read_json(work / "labels.json") or {}
        r = {"name": name, "title": p.stem, "path": str(p.relative_to(RECORDINGS)), "size": st.st_size,
             "modified": st.st_mtime, "recording": time.time() - st.st_mtime < 120,
             "surveyed": res is not None, "labelled": bool(labels.get("ads")) and labels.get("reviewed", False),
             "label_count": len(labels.get("ads", [])), "job": JOBS.active_for(name),
             "checked": len(labels.get("review") or {}), "missed_checked": bool(labels.get("missed_checked")),
             "listened": (work / "speech.json").exists()}
        if res:
            r.update(length=res["length"], breaks=len(res["breaks"]), graphics=res.get("graphics"),
                     surveyed_at=res.get("finished"))
            if labels.get("ads"):
                r["score"] = score(res["breaks"], labels["ads"], res["length"])
        out.append(r)
    out.sort(key=lambda r: -r["modified"])
    return out


def find(name):
    """Recording path for a survey name, only ever from the list of real recordings."""
    for p in RECORDINGS.rglob("*.ts"):
        if slug(p) == name:
            return p
    return None


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


_traces = {}


def trace(name):
    path = REPLAY / name / "trace.npz"
    try:
        mtime = path.stat().st_mtime
    except OSError:
        return None
    hit = _traces.get(name)
    if not hit or hit[0] != mtime:
        with np.load(path) as d:
            hit = (mtime, {k: d[k] for k in ("frames", "silences", "splice", "layouts", "layout_names",
                                             "layout_thresholds", "scene_sec", "scene_feats") if k in d})
        _traces[name] = hit
    return hit[1]


def timeline(name):
    """Per-second series for the chart, from the trace."""
    tr = trace(name)
    if tr is None:
        return None
    f = tr["frames"]
    if not len(f):
        return {"length": 0}
    sec = f[:, 0].astype(np.int64)
    n = int(sec.max()) + 1
    cnt = np.maximum(np.bincount(sec, minlength=n), 1)

    def mean(col):
        v = (f[:, col] if isinstance(col, int) else col).astype(np.float64)
        ok = ~np.isnan(v)
        s = np.bincount(sec[ok], v[ok], minlength=n)
        c = np.bincount(sec[ok], minlength=n)
        return [None if c[i] == 0 else round(float(s[i] / c[i]), 3) for i in range(n)]

    def any_(col):
        return (np.bincount(sec, f[:, col] > 0, minlength=n) > 0).astype(int).tolist()

    graphics = []
    for i in range((f.shape[1] - 6) // 2):
        shape, edge = f[:, 6 + 2 * i].copy(), f[:, 7 + 2 * i]
        learned = ~np.isnan(shape)
        shape[~learned] = edge[~learned]  # before the picture is learned, the edge score is what counts
        graphics.append({"score": mean(shape),
                         "learned": (np.bincount(sec, learned, minlength=n) * 2 > cnt).astype(int).tolist()})
    sil = np.zeros(n, int)
    s = tr["silences"].astype(np.int64)
    s = s[(s >= 0) & (s < n)]
    sil[s] = 1
    lays = []
    if "layouts" in tr and tr["layouts"].size:
        for i, nm in enumerate(tr["layout_names"]):
            lays.append({"name": str(nm), "threshold": float(tr["layout_thresholds"][i]),
                         "score": mean(tr["layouts"][:, i])})
    scene = None
    if "scene_feats" in tr and len(tr["scene_sec"]):
        from scene import Classifier
        c = Classifier.load()
        if c:
            q = c.p(tr["scene_feats"].astype(np.float32))
            scene = [None] * n
            for sec_, v in zip(tr["scene_sec"].tolist(), q.tolist()):
                if 0 <= sec_ < n:
                    scene[sec_] = round(v, 3)
    return {"length": n, "graphics": graphics, "layouts": lays, "scene": scene, "surface": mean(2), "surface_show": mean(3),
            "surface_ad": mean(4), "surface_learned": any_(5), "black": any_(1), "silent": sil.tolist()}


def merged(spans):
    """Sorted labels with overlaps joined. A label is [start, end] or [start, end, "sbs"]
    (a side-by-side ad, which the layout learning uses)."""
    out = []
    for a in sorted(spans):
        if out and a[0] <= out[-1][1]:
            out[-1][1] = max(out[-1][1], a[1])
            if len(a) > 2 and len(out[-1]) == 2:
                out[-1].append(a[2])
        else:
            out.append(list(a))
    return out


def score(breaks, ads, length):
    """How the mutes compare with the labelled ads, in whole seconds."""
    n = int(math.ceil(length)) + 1
    muted, truth = np.zeros(n, bool), np.zeros(n, bool)
    for s, e in breaks:
        muted[max(0, int(round(s))):max(0, int(round(e)))] = True
    ads = [a[:2] for a in ads]
    for s, e in ads:
        truth[max(0, int(round(s))):max(0, int(round(e)))] = True
    false_mutes = [[s, e] for s, e in breaks if truth[int(round(s)):int(round(e))].mean() < 0.5]
    missed, late, early = [], [], []
    for s, e in ads:
        seg = muted[int(round(s)):int(round(e))]
        if not seg.size or seg.mean() < 0.5:
            missed.append([s, e])
            continue
        first = int(np.argmax(seg))  # seconds of the ad heard before the mute began
        if first > 1:
            late.append([s, e, first])
        tail = int(np.argmax(seg[::-1]))  # seconds of the ad heard after it unmuted
        if tail > 1:
            early.append([s, e, tail])
    return {"ad_heard": int((truth & ~muted).sum()), "show_muted": int((muted & ~truth).sum()),
            "false_mutes": false_mutes, "missed": missed, "late": late, "early_end": early,
            "ads": len(ads), "breaks": len(breaks)}


def simulate(name, settings):
    tr = trace(name)
    if tr is None:
        return None
    from scene import show_title
    path = find(name)
    breaks = [[round(s, 2), round(e, 2)] for s, e in replay_trace(tr, settings, title=show_title(path.stem) if path else None)]
    out = {"breaks": breaks}
    labels = read_json(REPLAY / name / "labels.json") or {}
    if labels.get("ads"):
        length = float(tr["frames"][-1, 0]) if len(tr["frames"]) else 0
        out["score"] = score(breaks, labels["ads"], length)
    return out


def score_all(settings):
    """Tuned settings against every labelled recording, next to how the survey did."""
    rows = []
    for r in recordings():
        work = REPLAY / r["name"]
        labels = read_json(work / "labels.json") or {}
        res = read_json(work / "result.json")
        if not (labels.get("ads") and labels.get("reviewed") and res and (work / "trace.npz").exists()):
            continue
        sim = simulate(r["name"], settings)
        rows.append({"name": r["name"], "title": r["title"], "now": score(res["breaks"], labels["ads"], res["length"]),
                     "tuned": sim["score"]})
    return rows


def layouts():
    out = []
    for f in sorted((REPLAY.parent / "layouts").glob("*.npz")):
        with np.load(f) as d:
            out.append({"name": f.stem, "threshold": float(d["threshold"]), "pixels": int(d["mask"].sum()),
                        "examples": [str(x) for x in d["examples"]], "modified": f.stat().st_mtime})
    return out


# --- sounds: sponsor reads and repeated sounds -----------------------------------------

SOUNDS = REPLAY.parent / "sounds"
READ_VERDICTS = ("mute", "keep")
SOUND_VERDICTS = ("jingle", "closer", "ad", "program", "ignore")


def sounds_page():
    verdicts = read_json(SOUNDS / "verdicts.json") or {}
    recs = []
    for r in recordings():
        reads = (read_json(REPLAY / r["name"] / "reads.json") or {}).get("reads", [])
        # answers are keyed by the line's start second; a line found again may start a little
        # earlier or later, so an answer inside it still counts
        mine = [(int(k.rsplit("@", 1)[1]), k) for k in verdicts
                if k.startswith(f"read:{r['name']}@") and k.rsplit("@", 1)[1].isdigit()]
        for x in reads:
            key = f"read:{r['name']}@{round(x['start'])}"
            if key not in verdicts:
                key = next((k for sec, k in mine if x["start"] - 1 <= sec <= x["end"]), key)
            x["key"] = key
        recs.append({"name": r["name"], "title": r["title"], "listened": r["listened"], "job": r["job"],
                     "reads": reads})
    found = read_json(SOUNDS / "sounds.json") or {}
    # a sound found again may get another id; an answer follows it by where it was heard
    answered = [(k, v.get("places") or []) for k, v in verdicts.items() if k.startswith("sound:")]
    for snd in found.get("sounds", []):
        key = f"sound:{snd['id']}"
        if key not in verdicts:
            for k, places in answered:
                if any(p["name"] == q["name"] and p["start"] < q["end"] and q["start"] < p["end"]
                       for p in snd["places"] for q in places):
                    key = k
                    break
        snd["key"] = key
    return {"recordings": recs, "sounds": found.get("sounds", []), "found": found.get("found"),
            "verdicts": verdicts}


def set_verdict(key, value):
    """key: "read:<recording>@<second>" or "sound:<id>"; value "" clears it."""
    kind, _, rest = str(key).partition(":")
    allowed = READ_VERDICTS if kind == "read" else SOUND_VERDICTS if kind == "sound" else ()
    if not rest or (value and value not in allowed):
        raise ValueError("bad verdict")
    verdicts = read_json(SOUNDS / "verdicts.json") or {}
    if value:
        verdicts[key] = {"value": value, "at": time.time()}
        if kind == "sound":  # where it was heard, so the verdict can follow it if ids change
            for snd in (read_json(SOUNDS / "sounds.json") or {}).get("sounds", []):
                if snd["id"] == rest:
                    verdicts[key]["places"] = snd["places"]
    else:
        verdicts.pop(key, None)
    SOUNDS.mkdir(parents=True, exist_ok=True)
    tmp = SOUNDS / "verdicts.tmp"
    tmp.write_text(json.dumps(verdicts, indent=1))
    os.replace(tmp, SOUNDS / "verdicts.json")
    return verdicts


# --- jobs ------------------------------------------------------------------------

class Jobs:
    """Surveys and renders, run as app/replay.py in this container, a few at a time."""

    def __init__(self):
        self.jobs, self.lock, self.n = [], threading.Lock(), 0
        (STATE / "jobs").mkdir(parents=True, exist_ok=True)

    def start(self, kind, name, **kw):
        whole = kind in ("layouts", "train", "sounds")  # jobs about all recordings, not one
        path = None if whole else find(name)
        if path is None and not whole:
            raise ValueError("no such recording")
        if whole and any(j["kind"] == kind and j["state"] == "running" for j in self.jobs):
            raise ValueError("that's already running")
        if kind in ("survey", "listen") and self.active_for(name):
            raise ValueError("a job is already running for this recording")
        if kind == "listen":
            args, what = ["listen", str(path)], "Listen (speech + fingerprints)"
        elif kind == "sounds":
            args, what = ["sounds"], "Find sponsor reads and repeated sounds"
        elif kind == "survey":
            if self.active_for(name):
                raise ValueError("a job is already running for this recording")
            args = ["survey", str(path)] + (["--fresh"] if kw.get("fresh") else [])
            what = "Survey" + (" (learn again)" if kw.get("fresh") else "")
        elif kind == "layouts":
            args, what = ["layouts"], "Learn side-by-side layouts"
        elif kind == "train":
            args, what = ["train"], "Train the scene check"
        elif kind == "render":
            start, length = float(kw["start"]), float(kw.get("length", 600))
            if not 0 <= start or not 10 <= length <= 3600:
                raise ValueError("bad clip range")
            args = ["render", str(path), hms(start), hms(length)]
            what = f"Clip {hms(start)} +{int(length // 60)}m"
        else:
            raise ValueError("unknown job")
        env = dict(os.environ)
        for k, v in (kw.get("settings") or {}).items():
            if k in TUNABLE:
                env[k] = str(float(v))
        with self.lock:
            if sum(j["state"] == "running" for j in self.jobs) >= MAX_JOBS:
                raise ValueError(f"{MAX_JOBS} jobs are already running; wait for one to finish")
            self.n += 1
            job = {"id": f"{int(time.time())}-{self.n}", "kind": kind, "name": name, "what": what,
                   "title": path.stem if path else "all labelled recordings", "state": "running", "started": time.time(), "ended": None}
            log = STATE / "jobs" / f"{job['id']}.log"
            job["log"] = str(log)
            proc = subprocess.Popen([sys.executable, "-u", str(REPLAY_PY), *args], stdout=open(log, "wb"),
                                    stderr=subprocess.STDOUT, env=env, start_new_session=True)
            job["proc"] = proc
            self.jobs.append(job)
            self.jobs = self.jobs[-50:]
        threading.Thread(target=self._wait, args=(job,), daemon=True).start()
        return self.public(job)

    def _wait(self, job):
        code = job["proc"].wait()
        job["state"] = "done" if code == 0 else "failed"
        job["ended"] = time.time()

    def stop(self, job_id):
        for j in self.jobs:
            if j["id"] == job_id and j["state"] == "running":
                j["proc"].terminate()
                j["state"] = "stopped"
                return True
        return False

    def active_for(self, name):
        for j in self.jobs:
            if j["name"] == name and j["state"] == "running":
                return self.public(j, tail=False)
        return None

    @staticmethod
    def public(j, tail=True):
        out = {k: v for k, v in j.items() if k not in ("proc", "log")}
        try:
            text = Path(j["log"]).read_bytes()[-20000:].decode(errors="replace")
        except OSError:
            text = ""
        # Progress: the survey prints "  1:23:45 done (20x real time)"
        m = re.findall(r"^\s+(learning: )?(\d+:\d\d:\d\d) done", text, re.M)
        if m:
            out["progress"] = ("learning " if m[-1][0] else "") + m[-1][1]
        if tail:
            out["log"] = "\n".join(text.splitlines()[-60:])
        return out

    def list(self):
        return [self.public(j) for j in reversed(self.jobs)]


JOBS = None


# --- video -------------------------------------------------------------------------

_probe = {}
_players = []  # ffmpeg processes streaming video; one viewer, so a new one stops the old


def interlaced(path):
    if path not in _probe:
        try:
            out = subprocess.run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-show_entries",
                                  "stream=field_order", "-of", "csv=p=0", str(path)],
                                 capture_output=True, text=True, timeout=20).stdout
            _probe[path] = out.strip() not in ("", "progressive", "unknown")
        except (OSError, subprocess.TimeoutExpired):
            _probe[path] = False
    return _probe[path]


def video_cmd(path, start, height):
    deint = interlaced(path)
    if GPU:
        hw = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        vf = ("yadif_cuda=0:-1:1," if deint else "") + f"scale_cuda=-2:{height}"
        venc = ["-c:v", "h264_nvenc", "-preset", "p3", "-tune", "ll", "-b:v", "5M", "-maxrate", "8M", "-bufsize", "8M"]
    else:
        hw = []
        vf = ("yadif," if deint else "") + f"scale=-2:{height}"
        venc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
    return [FFMPEG, "-hide_banner", "-loglevel", "error", "-ss", f"{start:.2f}", *hw, "-i", str(path),
            "-map", "0:v:0", "-map", "0:a:0", "-vf", vf, *venc, "-g", "60",
            "-c:a", "aac", "-ac", "2", "-b:a", "160k",
            "-f", "mp4", "-movflags", "frag_keyframe+empty_moov+default_base_moof", "pipe:1"]


# HLS, for iPhones and Safari: they won't play the endless mp4 stream above (no length, no
# byte ranges). The playlist lists the whole recording in SEG-second pieces, so the player's
# own scrubber covers all of it. One ffmpeg run makes the pieces in order from wherever the
# player asks; asking for a piece far from it restarts it there. It pauses when it gets too
# far ahead of the player, and old pieces are deleted.
SEG = 6
HLS_AHEAD = 10   # pieces made before the player asks for them
HLS_KEEP = 4     # pieces kept behind the one the player last asked for

_durations = {}


def duration(path):
    key = (path, path.stat().st_mtime)
    if key not in _durations:
        out = subprocess.run([FFPROBE, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                             capture_output=True, text=True, timeout=30).stdout
        _durations[key] = float(out.strip() or 0)
    return _durations[key]


def hls_playlist(path, height):
    n = math.ceil(duration(path) / SEG)
    last = duration(path) - (n - 1) * SEG
    lines = ["#EXTM3U", "#EXT-X-VERSION:3", f"#EXT-X-TARGETDURATION:{SEG + 1}", "#EXT-X-MEDIA-SEQUENCE:0",
             "#EXT-X-PLAYLIST-TYPE:VOD", "#EXT-X-INDEPENDENT-SEGMENTS"]
    for i in range(n):
        lines += [f"#EXTINF:{SEG if i < n - 1 else max(last, 0.1):.3f},", f"hls/{i}.ts?h={height}"]
    return "\n".join(lines + ["#EXT-X-ENDLIST", ""]).encode()


class HlsMaker:
    def __init__(self):
        self.lock = threading.Lock()
        self.proc = self.key = None
        self.first = self.asked = 0
        self.dir = Path(os.environ.get("HLS_TMP", "/tmp")) / "admute-hls"
        threading.Thread(target=self._pace, daemon=True).start()

    def _made(self):
        """Index of the newest finished piece (-1 if none)."""
        done = [int(f.stem) for f in self.dir.glob("*.ts") if f.stem.isdigit()]
        return max(done, default=-1)

    def _start(self, path, height, n):
        self._stop()
        self.dir.mkdir(parents=True, exist_ok=True)
        for f in self.dir.iterdir():
            f.unlink(missing_ok=True)
        deint = interlaced(path)
        if GPU:
            hw = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
            vf = ("yadif_cuda=0:-1:1," if deint else "") + f"scale_cuda=-2:{height}"
            venc = ["-c:v", "h264_nvenc", "-preset", "p3", "-b:v", "5M", "-maxrate", "8M", "-bufsize", "8M", "-forced-idr", "1"]
        else:
            hw = []
            vf = ("yadif," if deint else "") + f"scale=-2:{height}"
            venc = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-sc_threshold", "0"]
        start = n * SEG
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-ss", f"{start}", *hw, "-i", str(path),
               "-map", "0:v:0", "-map", "0:a:0", "-vf", vf, *venc, "-g", "600",
               "-force_key_frames", f"expr:gte(t,n_forced*{SEG})",
               "-c:a", "aac", "-ac", "2", "-b:a", "160k", "-output_ts_offset", f"{start}",
               "-f", "hls", "-hls_time", str(SEG), "-hls_list_size", "0", "-start_number", str(n),
               "-hls_flags", "temp_file", "-hls_segment_filename", str(self.dir / "%d.ts"), str(self.dir / "out.m3u8")]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.key, self.first, self.asked = (path, height), n, n

    def _stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGCONT)  # in case it's paused
            self.proc.kill()
            self.proc.wait()
        self.proc = None

    def piece(self, path, height, n):
        """The bytes of piece n, made if needed (None if it couldn't be)."""
        f = self.dir / f"{n}.ts"
        with self.lock:
            running = self.proc is not None and self.proc.poll() is None
            coming = running and self.first <= n and self._made() < n <= self._made() + 3
            if self.key != (path, height) or not (f.exists() or coming):
                self._start(path, height, n)
            self.asked = n
            if self.proc and self.proc.poll() is None:
                self.proc.send_signal(signal.SIGCONT)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if f.exists():  # ffmpeg renames a piece into place once it's finished
                try:
                    return f.read_bytes()
                except FileNotFoundError:
                    return None
            if self.key != (path, height) or self.proc is None or (self.proc.poll() is not None and not f.exists()):
                return None
            time.sleep(0.05)
        return None

    def _pace(self):
        while True:
            time.sleep(0.5)
            with self.lock:
                if not self.proc or self.proc.poll() is not None:
                    continue
                made = self._made()
                self.proc.send_signal(signal.SIGSTOP if made - self.asked > HLS_AHEAD else signal.SIGCONT)
                for f in self.dir.glob("*.ts"):
                    if f.stem.isdigit() and int(f.stem) < self.asked - HLS_KEEP:
                        f.unlink(missing_ok=True)


HLS = HlsMaker()


# --- watching live TV -------------------------------------------------------------
# The proxy's output (muted, DELAY behind) as HLS in two versions: the broadcast's own size
# (1080 or 720 lines) at high quality, and 720p at about 5 Mb/s for slower connections; the
# player picks between them as the connection allows. Deinterlaced to 60 frames a second
# (sports on 1080i channels) and not enlarged: the player upscales on the viewer's side.
# One channel at a time, since each one holds a tuner.

LIVE_TMP = Path(os.environ.get("HLS_TMP", "/tmp")) / "admute-live"
LIVE_SEG = 2
WATCH_MBPS = float(os.environ.get("WATCH_MBPS", "20"))   # the full-quality version's peak bitrate


def watch_cmd(num, out):
    from urllib.parse import quote
    src = f"{LIVE_URL}/stream/{quote(num, safe='')}"
    if GPU:
        hw = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        graph = "[0:v:0]bwdif_cuda=1:-1:1,split=2[hi][x];[x]scale_cuda=-2:'min(720,ih)'[lo]"
        venc = ["-c:v", "h264_nvenc", "-preset", "p6", "-tune", "hq", "-rc", "vbr", "-bf", "3",
                "-spatial-aq", "1", "-temporal-aq", "1", "-rc-lookahead", "20",
                "-cq:v:0", "19", "-cq:v:1", "23", "-a53cc", "1"]   # keep the closed captions
    else:
        hw = []
        graph = "[0:v:0]bwdif=1:-1:1,split=2[hi][x];[x]scale=-2:'min(720,ih)'[lo]"
        venc = ["-c:v", "libx264", "-preset", "veryfast", "-crf:v:0", "19", "-crf:v:1", "23"]
    rates = ["-b:v:0", f"{WATCH_MBPS * 0.8:g}M", "-maxrate:v:0", f"{WATCH_MBPS:g}M", "-bufsize:v:0", f"{WATCH_MBPS * 1.6:g}M",
             "-b:v:1", "4M", "-maxrate:v:1", "5M", "-bufsize:v:1", "10M"]
    return [FFMPEG, "-hide_banner", "-loglevel", "error", "-fflags", "+discardcorrupt+genpts", *hw,
            "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "4", "-i", src,
            "-filter_complex", graph, "-map", "[hi]", "-map", "[lo]", "-map", "0:a:0", "-map", "0:a:0",
            *venc, *rates, "-g", "120", "-force_key_frames", f"expr:gte(t,n_forced*{LIVE_SEG})",
            # a reception dropout loses audio frames: fill the gap with silence, keeping sync
            "-af", "aresample=async=1", "-c:a", "aac", "-ac", "2", "-b:a", "192k",
            "-f", "hls", "-hls_time", str(LIVE_SEG), "-hls_list_size", "10",
            "-hls_flags", "delete_segments+temp_file+independent_segments",
            "-var_stream_map", "v:0,a:0 v:1,a:1", "-master_pl_name", "live.m3u8",
            "-hls_segment_filename", str(out / "%v" / "%d.ts"), str(out / "%v" / "index.m3u8")]


class LiveTV:
    """The one live channel being converted."""

    def __init__(self):
        self.lock = threading.Lock()
        self.proc, self.num, self.seen = None, None, 0.0
        threading.Thread(target=self._idle, daemon=True).start()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        self.proc = self.num = None

    def playlist(self, num):
        """The top playlist (listing both versions), once the first pieces are ready."""
        with self.lock:
            self.seen = time.monotonic()
            if self.num != num or self.proc is None or self.proc.poll() is not None:
                self.stop()
                shutil.rmtree(LIVE_TMP, ignore_errors=True)
                LIVE_TMP.mkdir(parents=True, exist_ok=True)
                self.proc = subprocess.Popen(watch_cmd(num, LIVE_TMP), stdin=subprocess.DEVNULL,
                                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.num = num
        deadline = time.monotonic() + 30   # the proxy holds DELAY seconds before sending anything
        while time.monotonic() < deadline:
            try:
                if (LIVE_TMP / "0" / "index.m3u8").read_text().count("#EXTINF") >= 2:
                    return (LIVE_TMP / "live.m3u8").read_text()
            except FileNotFoundError:
                pass
            if self.proc is None or self.proc.poll() is not None:
                return None
            time.sleep(0.2)
        return None

    def file(self, num, version, name):
        """A version's playlist or one of its pieces (None when gone or another channel's)."""
        self.seen = time.monotonic()
        if num != self.num:
            return None
        try:
            return (LIVE_TMP / version / name).read_bytes()
        except FileNotFoundError:
            return None

    def _idle(self):
        while True:
            time.sleep(5)
            with self.lock:
                if self.proc and time.monotonic() - self.seen > 20:   # the player went away
                    self.stop()


LIVE_TV = LiveTV()


# --- rapid recap ------------------------------------------------------------------------
# While a game is being recorded, its key plays are cut into a recap as they happen (see
# recap.py), so catching up in the Watch tab starts at once. Only football with ESPN data.

RECAPS = REPLAY.parent / "recap"
RECAP_EVERY = float(os.environ.get("RECAP_EVERY", "20"))   # seconds between looks at the feed


class RecapMaker:
    def __init__(self):
        self.errors = {}
        if os.environ.get("RECAP", "1") == "1":
            threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        import recap
        while True:
            now = time.time()
            for p in RECORDINGS.rglob("*.ts"):
                try:
                    if now - p.stat().st_mtime > 90 or not recap.league_of(p.stem) or not recap.teams_of(p.stem):
                        continue   # not being recorded now, or not a game the feed covers
                    r = recap.Recap(p, RECAPS / slug(p))
                    r.update(log=lambda m, n=p.stem: print(f"recap {n}: {m.strip()}", flush=True))
                    self.errors.pop(p.stem, None)
                except Exception as e:
                    if self.errors.get(p.stem) != str(e):
                        print(f"recap {p.stem}: {e}", flush=True)
                    self.errors[p.stem] = str(e)
            time.sleep(RECAP_EVERY)


RECAP_MAKER = RecapMaker()


def recaps():
    """Every recap, newest first, with its clips in order."""
    import recap
    out = []
    for d in RECAPS.glob("*/recap.json"):
        st = read_json(d)
        if not st or not st.get("clips"):
            continue
        rec = Path(st["recording"])
        try:
            live = time.time() - rec.stat().st_mtime < 90
        except OSError:
            live = False
        teams = recap.teams_of(rec.stem)
        clips = sorted(st["clips"].values(), key=lambda c: c["wall"])
        out.append({"name": d.parent.name, "title": rec.stem, "teams": teams, "live": live,
                    "updated": d.stat().st_mtime, "lag": st.get("lag"),
                    "clips": [{k: c.get(k) for k in ("id", "kind", "level", "when", "score", "text", "wall",
                                                     "situation", "before")}
                              | {"seconds": round(c["end"] - c["start"], 1), "file": c["clip"]} for c in clips]})
    return sorted(out, key=lambda r: -r["updated"])


def channels():
    if not LIVE_URL:
        return {"configured": False, "channels": []}
    try:
        with urllib.request.urlopen(LIVE_URL + "/channels", timeout=5) as r:
            return {"configured": True, **json.loads(r.read())}
    except Exception as e:
        return {"configured": True, "channels": [], "error": str(e)}


# --- live --------------------------------------------------------------------------

def live():
    out = {"configured": bool(LIVE_URL)}
    if LIVE_URL:
        try:
            with urllib.request.urlopen(LIVE_URL + "/status", timeout=3) as r:
                out["status"] = json.loads(r.read())
        except Exception as e:
            out["error"] = str(e)
    try:
        with open(Path(os.environ.get("DATA", "/data")) / "logs" / "events.log", "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 30000))
            out["events"] = f.read().decode(errors="replace").splitlines()[-150:][::-1]
    except OSError:
        out["events"] = []
    return out


# --- web ---------------------------------------------------------------------------

STATIC = {"app.js": "text/javascript", "watch.js": "text/javascript", "hls.min.js": "text/javascript", "style.css": "text/css", "index.html": "text/html; charset=utf-8"}


class Handler(BaseHTTPRequestHandler):
    def send(self, code, body, ctype="application/json", cache="no-store"):
        if not isinstance(body, bytes):
            body = json.dumps(body, allow_nan=False, default=float).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        for k, v in formlogin.SECURITY_HEADERS:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, ctype):
        """A file, with byte ranges (Safari won't play video without them)."""
        size = path.stat().st_size
        start, end = 0, size - 1
        m = re.fullmatch(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start, end = int(m.group(1)), min(int(m.group(2) or size - 1), size - 1)
            else:
                start = max(0, size - int(m.group(2)))
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
        self.send_response(206 if m and (m.group(1) or m.group(2)) else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if m and (m.group(1) or m.group(2)):
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "private, max-age=3600")
        for k, v in formlogin.SECURITY_HEADERS:
            self.send_header(k, v)
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0 and (chunk := f.read(min(left, 1 << 20))):
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        url = urlsplit(self.path)
        path, q = url.path, {k: v[-1] for k, v in parse_qs(url.query).items()}
        if formlogin.httpserver.handle(self, LOGIN, path):
            return
        try:
            self.get(path, q)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def get(self, path, q):
        if path in ("/", "/index.html"):
            return self.send(200, (WEB / "index.html").read_bytes(), STATIC["index.html"])
        if path.startswith("/static/") and path[8:] in STATIC:
            return self.send(200, (WEB / path[8:]).read_bytes(), STATIC[path[8:]])
        if path == "/api/recordings":
            return self.send(200, {"recordings": recordings(), "defaults": SETTINGS, "tunable": TUNABLE})
        if path == "/api/jobs":
            return self.send(200, JOBS.list())
        if path == "/api/live":
            return self.send(200, live())
        if path == "/api/recaps":
            return self.send(200, recaps())
        m = re.fullmatch(r"/api/recap/([\w.-]+)/([\w-]+\.mp4)", path)
        if m:
            f = RECAPS / m.group(1) / "clips" / m.group(2)
            return self.send_file(f, "video/mp4") if f.is_file() else self.send(404, {"error": "no such clip"})
        if path == "/api/channels":
            return self.send(200, channels())
        m = re.fullmatch(r"/api/watch/([\w.-]+)/(?:live\.m3u8|([01])/(index\.m3u8|\d+\.ts))", path)
        if m and LIVE_URL:
            num = m.group(1)
            if m.group(2):
                body = LIVE_TV.file(num, m.group(2), m.group(3))
                if body is None:
                    return self.send(404, {"error": "gone"})
                kind = "application/vnd.apple.mpegurl" if m.group(3).endswith("m3u8") else "video/mp2t"
                return self.send(200, body, kind)
            text = LIVE_TV.playlist(num)
            if text is None:
                return self.send(503, {"error": "the channel didn't start (no free tuner?)"})
            return self.send(200, text.encode(), "application/vnd.apple.mpegurl")
        if path == "/api/layouts":
            return self.send(200, layouts())
        if path == "/api/sounds":
            return self.send(200, sounds_page())
        if path == "/api/scene":
            from scene import Classifier
            c = Classifier.load()
            return self.send(200, {"trained": bool(c), **(c.info if c else {})})
        m = re.fullmatch(r"/api/layouts/(layout-\d+)\.png", path)
        if m and (REPLAY.parent / "layouts" / f"{m.group(1)}.png").exists():
            return self.send(200, (REPLAY.parent / "layouts" / f"{m.group(1)}.png").read_bytes(), "image/png")
        if path == "/api/score-all":
            return self.send(200, score_all(settings_from(q)))
        m = re.fullmatch(r"/api/rec/([\w.-]+)/(?:hls\.m3u8|hls/(\d+)\.ts)", path)
        if m:
            rec = find(m.group(1))
            if rec is None:
                return self.send(404, {"error": "no such recording"})
            height = int(q.get("h", 720)) if q.get("h", "720").isdigit() else 720
            height = height if height in (360, 540, 720, 1080) else 720
            if m.group(2) is None:
                return self.send(200, hls_playlist(rec, height), "application/vnd.apple.mpegurl")
            body = HLS.piece(rec, height, int(m.group(2)))
            if body is None:
                return self.send(503, {"error": "couldn't make that part of the video"})
            return self.send(200, body, "video/mp2t", cache="private, max-age=3600")
        m = re.fullmatch(r"/api/rec/([\w.-]+)(?:/(\w+))?(?:/(\d+)\.jpg)?", path)
        if not m:
            return self.send(404, {"error": "not found"})
        name, what, n = m.groups()
        work = REPLAY / name
        if find(name) is None:
            return self.send(404, {"error": "no such recording"})
        if what is None:
            return self.send(200, {"name": name, "title": find(name).stem, "result": read_json(work / "result.json"),
                                   "labels": read_json(work / "labels.json") or {"ads": [], "reviewed": False, "notes": ""},
                                   "timeline": timeline(name), "job": JOBS.active_for(name)})
        if what == "sim":
            sim = simulate(name, settings_from(q))
            return self.send(200 if sim else 404, sim or {"error": "survey this recording first"})
        if what == "sheet" and n is not None:
            f = work / "sheets" / f"sheet_{int(n):03d}.jpg"
            if f.exists():
                return self.send(200, f.read_bytes(), "image/jpeg", cache="max-age=600")
            return self.send(404, {"error": "no sheet"})
        if what == "thumb":
            return self.thumb(find(name), work, float(q.get("t", 0)))
        if what == "logo":
            f = work / "logos" / f"{name}.png"
            if f.exists():
                return self.send(200, f.read_bytes(), "image/png", cache="max-age=60")
            return self.send(404, {"error": "no logo"})
        if what == "video":
            return self.video(find(name), float(q.get("t", 0)), int(q.get("h", 720)))
        return self.send(404, {"error": "not found"})

    def thumb(self, path, work, t):
        """The exact frame at second t (cached), for the review's missed-ads grid."""
        t = max(0, int(t))
        f = work / "thumbs" / f"{t}.jpg"
        if not f.exists():
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_suffix(".tmp.jpg")
            subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-ss", str(t), "-i", str(path),
                            "-frames:v", "1", "-vf", "scale=192:108", "-q:v", "5", str(tmp)],
                           stdin=subprocess.DEVNULL, timeout=30)
            if not tmp.exists():
                return self.send(404, {"error": "no frame"})
            os.replace(tmp, f)
        return self.send(200, f.read_bytes(), "image/jpeg", cache="max-age=86400")

    def video(self, path, start, height):
        height = height if height in (360, 540, 720, 1080) else 720
        for p in _players:
            if p.poll() is None:
                p.kill()
        _players.clear()
        proc = subprocess.Popen(video_cmd(path, max(0.0, start), height), stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)
        _players.append(proc)
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Cache-Control", "no-store")
        for k, v in formlogin.SECURITY_HEADERS:
            self.send_header(k, v)
        self.end_headers()
        self.close_connection = True
        try:
            while chunk := proc.stdout.read(65536):
                self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            if proc.poll() is None:
                proc.kill()
            proc.wait()

    def do_POST(self):
        path = urlsplit(self.path).path
        if formlogin.httpserver.handle(self, LOGIN, path):
            return
        # JSON only: a plain cross-site form post can't set this content type.
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            return self.send(415, {"error": "send JSON"})
        try:
            req = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length") or 0), 1 << 20)) or b"{}")
            if not isinstance(req, dict):
                raise ValueError("expected an object")
            if path == "/api/jobs":
                return self.send(200, JOBS.start(req.get("kind"), str(req.get("name")), **{
                    k: req[k] for k in ("fresh", "start", "length", "settings") if k in req}))
            if path == "/api/sounds/verdict":
                return self.send(200, set_verdict(req.get("key"), str(req.get("value") or "")))
            if path == "/api/jobs/stop":
                return self.send(200, {"stopped": JOBS.stop(str(req.get("id")))})
            m = re.fullmatch(r"/api/rec/([\w.-]+)/labels", path)
            if m and find(m.group(1)) is not None:
                ads = merged([[float(a[0]), float(a[1])] + (["sbs"] if len(a) > 2 and a[2] == "sbs" else [])
                              for a in req.get("ads", []) if float(a[1]) > float(a[0]) >= 0])
                # Review mode's answers per mute (keyed by the mute's start second), kept so the
                # review can pick up where it left off.
                review = {}
                for k, v in (req.get("review") or {}).items():
                    if isinstance(v, dict) and v.get("verdict") in ("right", "wrong", "partly"):
                        span = v.get("span")
                        review[str(int(float(k)))] = {"verdict": v["verdict"], **(
                            {"span": [float(span[0]), float(span[1])]} if isinstance(span, list) and len(span) == 2 else {})}
                labels = {"ads": ads, "reviewed": bool(req.get("reviewed")), "notes": str(req.get("notes", ""))[:5000],
                          "review": review, "missed_checked": bool(req.get("missed_checked")), "updated": time.time()}
                f = REPLAY / m.group(1) / "labels.json"
                f.parent.mkdir(parents=True, exist_ok=True)
                tmp = f.with_suffix(".tmp")
                tmp.write_text(json.dumps(labels, indent=1))
                os.replace(tmp, f)
                return self.send(200, labels)
        except (ValueError, TypeError, KeyError) as e:
            return self.send(400, {"error": str(e)})
        self.send(404, {"error": "not found"})

    def log_message(self, *a):
        pass


def settings_from(q):
    out = {}
    for k, (lo, hi, _) in TUNABLE.items():
        if k in q:
            out[k] = min(hi, max(lo, float(q[k])))
    return out


if __name__ == "__main__":
    STATE.mkdir(parents=True, exist_ok=True)
    JOBS = Jobs()
    LOGIN = formlogin.Login.from_env(  # password: python -m formlogin passwd (see README)
        "STUDIO", "admute-studio", str(STATE), title="admute studio",
        accent=("#b4472f", "#e0785c"),
        icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'><rect x='1' y='3' width='14' height='10' rx='2' fill='%23e0785c'/></svg>")
    print(time.strftime("%F %T"), f"admute studio on :{PORT}", flush=True)
    ThreadingHTTPServer(("", PORT), Handler).serve_forever()
