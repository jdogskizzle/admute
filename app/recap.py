"""Rapid recap: the key plays of a game so far, cut from its recording while the game is on.

Plays come from ESPN's public scoreboard feed, each with the wall-clock time it was logged.
The recording's own clock says where that is in the file: the file was last written at its
last-modified time, so it started that long before. The broadcast shows the snap about
RECAP_LAG seconds after ESPN's time (measured on CBS, FOX, NBC and ESPN games: the
commentator's "touchdown" call comes ~23s after it on short plays).

Every update cuts any key play the recording already covers, so calling it every few
seconds during a game keeps the recap current, and opening it costs nothing. Clips are
H.264/AAC MP4s, each with a caption at the start; the whole recap is them in order.
"""

import json
import os
import threading
import re
import subprocess
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from pipeline import FFMPEG
from ts import PKT

FEED = os.environ.get("RECAP_FEED", "https://site.api.espn.com/apis/site/v2/sports")
LAG = float(os.environ.get("RECAP_LAG", "18"))     # first guess, seconds from ESPN's time to the snap on air
# Until the game's lag is known, clips cover every lag seen so far (FOX and CBS 16s, NBC 22s,
# ESPN 26s), and are cut again once it is.
LAG_RANGE = (12.0, 28.0)
PRE = float(os.environ.get("RECAP_PRE", "4"))      # seconds of the line-up shown before the snap
GPU = os.environ.get("RECAP_GPU", "0") == "1"      # encode clips with NVENC instead of x264
# Guide titles of the sports the feed covers, and the feed's name for each.
LEAGUES = {"NFL Football": "football/nfl", "College Football": "football/college-football"}
PERIODS = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "admute"})
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def league_of(title):
    return next((v for k, v in LEAGUES.items() if title.startswith(k)), None)


def find_event(league, away, home, when):
    """ESPN's id for the game between these teams around this time (a datetime), or None."""
    want = {away.lower(), home.lower()}
    for days in (0, -1, 1):
        day = (when + timedelta(days=days)).strftime("%Y%m%d")
        for e in get_json(f"{FEED}/{league}/scoreboard?dates={day}&limit=300").get("events", []):
            names = set()
            for c in e["competitions"][0]["competitors"]:
                t = c["team"]
                names |= {str(t.get(k, "")).lower() for k in ("displayName", "shortDisplayName", "name", "location")}
            if want <= names:
                return e["id"]
    return None


def teams_of(title):
    """("Tampa Bay Buccaneers", "Chicago Bears") from "... - Tampa Bay Buccaneers at Chicago Bears ..."."""
    part = title.rsplit(" - ", 1)[-1]
    part = re.sub(r"\s*\(.*?\)|\s+\d{4}-\d\d-\d\d.*$", "", part)
    m = re.match(r"(.+?) (?:at|vs\.?) (.+)$", part)
    return (m.group(1).strip(), m.group(2).strip()) if m else None


# --- which plays -----------------------------------------------------------------------

def _kind(p):
    """(kind, level, seconds after the snap to keep) for a key play, or None.
    Level 1: scores, turnovers, missed kicks. Level 2: big plays and sacks."""
    text, typ = _final(p.get("text", "")), p.get("type", {}).get("text", "")
    yards = p.get("statYardage") or 0
    if "No Play" in text or typ in ("Timeout", "End Period", "End of Half", "End of Game", "Two-minute warning"):
        return None
    kick = typ.startswith(("Kickoff", "Punt"))
    if "TOUCHDOWN" in text and "NULLIFIED" not in text:
        return "TD", 1, min(9 + abs(yards) / 6, 22)
    if typ == "Field Goal Good":
        return "FG", 1, 9
    if typ in ("Field Goal Missed", "Blocked Field Goal", "Blocked Punt"):
        return "MISS", 1, 10
    if "Safety" in typ or "SAFETY" in text:
        return "SAFETY", 1, 10
    if "Interception" in typ:
        return "INT", 1, min(9 + abs(yards) / 6, 20)
    if typ == "Fumble Recovery (Opponent)" or ("Fumble" in typ and "Opponent" in typ):
        return "FUMBLE", 1, 11
    if typ == "Sack":
        return "SACK", 2, 7
    if kick and yards >= 40:
        return "RETURN", 2, min(10 + yards / 8, 20)
    if not kick and yards >= 20:
        return "BIG", 2, min(7 + yards / 6, 18)
    return None


def _final(text):
    """A play's text as finally ruled: after a review that reversed the call, the new ruling."""
    return text.split("REVERSED.", 1)[1].strip() if "REVERSED." in text else text


def _headline(text, kind):
    """A short caption from a play's text: the formation, tacklers and kicking crew left out;
    for a kick return, the return."""
    text = re.sub(r"^\([^)]*\)\s*", "", _final(text).strip())
    sents = [x.strip() for x in re.split(r"\.\s+", text)
             if x.strip() and not re.search(r"reported in as eligible|^Injury|^Timeout", x.strip())]
    line = sents[0] if sents else text
    if kind == "RETURN" and len(sents) > 1:
        line = sents[1]
    elif kind == "FUMBLE" and "FUMBLES" not in line:
        line += ", " + next((x for x in sents if "FUMBLES" in x), "")
    line = re.sub(r",? (?:Center|Holder)-[\w.'-]+", "", line)
    line = re.sub(r"\s*\([^)]*\)|\s*\[[^]]*\]", "", line)      # tacklers
    line = re.split(r"\s*(?:PENALTY|The Replay Official)", line)[0]
    line = line.strip(" ,.")
    if kind == "TD" and "TOUCHDOWN" not in line:
        line += ", TOUCHDOWN"                               # an interception or fumble returned
    return line[:110]


def key_plays(summary):
    comp = summary["header"]["competitions"][0]
    side = {c["homeAway"]: c["team"]["abbreviation"] for c in comp["competitors"]}
    drives = summary.get("drives", {})
    plays = [p for d in drives.get("previous", []) + ([drives["current"]] if drives.get("current") else [])
             for p in d.get("plays", [])]
    out, seen = [], set()
    for p in plays:
        k = _kind(p)
        if not k or not p.get("wallclock") or p["id"] in seen:
            continue
        seen.add(p["id"])
        kind, level, after = k
        period = p.get("period", {}).get("number", 0)
        when = f"{PERIODS.get(period, 'OT')} {p.get('clock', {}).get('displayValue', '')}".strip()
        score = f"{side.get('away', '')} {p.get('awayScore', 0)}  {side.get('home', '')} {p.get('homeScore', 0)}"
        out.append({"id": p["id"], "wall": datetime.fromisoformat(p["wallclock"].replace("Z", "+00:00")).timestamp(),
                    "kind": kind, "level": level, "after": round(after, 1), "when": when, "score": score,
                    "text": _headline(p.get("text", ""), kind)})
    return sorted(out, key=lambda x: x["wall"])


# --- the recording's clock -------------------------------------------------------------

def recording_clock(path):
    """(wall time the recording started, its length in seconds), read from the file now."""
    from replay import pcr_at, since
    st = path.stat()
    with open(path, "rb") as f:
        _, first = pcr_at(f, 0)
        _, last = pcr_at(f, max(0, st.st_size - PKT * 8000))
    if first is None or last is None:
        raise ValueError(f"{path.name}: no timestamps found")
    length = since(last, first)
    return st.st_mtime - length, length


# --- finding the snap -------------------------------------------------------------------
# The lag from ESPN's time to the snap on air differs by game (about 21s on NBC, 15s on FOX),
# so each play's snap is looked for in the picture: in a wide shot of the field the players
# stand still at the line, then everything moves at once. Plays where that's clear set the
# game's lag; the rest go by it.

SW, SH, SFPS = 96, 54, 10


def _frames(path, start, length):
    """Small RGB frames (SFPS a second) of [start, start + length] of the recording."""
    from replay import seek
    off, _, _ = seek(path, max(0.0, start))
    p = subprocess.Popen([FFMPEG, "-v", "quiet", "-fflags", "+discardcorrupt", "-f", "mpegts", "-i", "pipe:0",
                          "-t", f"{length:.2f}", "-map", "0:v:0", "-vf",
                          f"bwdif=0:-1:1,fps={SFPS},scale={SW}:{SH}:flags=area,format=rgb24", "-f", "rawvideo", "pipe:1"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def feed():
        try:
            with open(path, "rb") as f:
                f.seek(off)
                while d := f.read(1 << 20):
                    p.stdin.write(d)
            p.stdin.close()
        except (BrokenPipeError, ValueError, OSError):
            pass
    threading.Thread(target=feed, daemon=True).start()
    data = p.stdout.read()
    p.wait()
    n = len(data) // (SW * SH * 3)
    return np.frombuffer(data[:n * SW * SH * 3], np.uint8).reshape(n, SH, SW, 3).astype(np.float32)


def snaps(path, start, length):
    """Times in [start, start + length] that look like a snap, each with how clear it is."""
    fr = _frames(path, start, length)
    if len(fr) < SFPS * 3:
        return []
    gray = fr.mean(axis=3)
    move = np.abs(np.diff(gray, axis=0)).mean(axis=(1, 2))      # move[i]: frame i to i + 1
    r, g, b = fr[..., 0], fr[..., 1], fr[..., 2]
    green = ((g >= r - 8) & (g > b + 25)).mean(axis=(1, 2))    # share that's grass (sunlit grass is yellowish)
    out, k = [], SFPS // 2                                      # half a second of stillness
    for i in range(k, len(move) - SFPS - 2):
        before, after = move[i - k:i], move[i + 2:i + 2 + SFPS]
        if before.max() > 4 or before.mean() > 3 or green[i - k:i + 1].mean() < 0.3:
            continue                                            # not a still wide shot
        if move[i] < before.mean() + 1 or move[i + 1] < 0.9 * move[i]:
            continue                                            # not the start of a rise
        if after.mean() < 6 or after.max() > 45:
            continue                                            # no burst (or a cut to another shot)
        out.append((start + (i + 1) / SFPS, float(after.mean() / (before.mean() + 1))))
    # Some productions (ESPN's) cut to the wide shot just as the ball is snapped: a cut into a
    # grass shot that's already moving counts too, as a weaker sign.
    for i in range(1, len(move) - SFPS - 2):
        after = move[i + 1:i + 1 + SFPS]
        into_grass = green[i + 1:i + 1 + SFPS].mean()
        cut = move[i] >= 30 and (move[i] >= 3 * max(move[i - 1], 1) or into_grass - green[max(0, i - 3):i + 1].mean() >= 0.3)
        if not cut or into_grass < 0.4:
            continue                                            # not a cut into a wide shot
        if after[:3].mean() < 4 or after.mean() < 6 or after.max() > 45:
            continue                                            # not moving right away
        out.append((start + (i + 1) / SFPS, 1.0))
    out.sort()
    # one per rise: the first of neighbours
    keep = []
    for t, c in out:
        if keep and t - keep[-1][0] < 1.0:
            continue
        keep.append((t, c))
    return keep


def game_lag(found):
    """The game's lag from the snap candidates of its plays (lists of lags): the 2-second span
    that most plays have a candidate in. None until three plays have one."""
    plays = [ls for ls in found if ls]
    if len(plays) < 3:
        return None
    best = None
    for x in sorted(l for ls in plays for l in ls):
        hits = [min(ls, key=lambda l: abs(l - x - 1)) for ls in plays if any(x <= l <= x + 2 for l in ls)]
        if best is None or len(hits) > len(best):
            best = hits
    return float(np.median(best))


# --- cutting --------------------------------------------------------------------------

def _escape(s):
    return s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace("%", "\\%")


def cut(path, start, length, out, caption):
    """Encode [start, start + length] seconds of the recording into an MP4 with a caption."""
    from replay import seek
    off, _, _ = seek(path, max(0.0, start))
    fade = min(0.4, length / 4)
    lines = [_escape(x) for x in caption]
    text = (f"drawtext=text='{lines[0]}':x=40:y=h-150:fontsize=h/24:fontcolor=white:box=1:boxcolor=black@0.6:"
            f"boxborderw=12:enable='lt(t,5)',"
            f"drawtext=text='{lines[1]}':x=40:y=h-90:fontsize=h/34:fontcolor=white:box=1:boxcolor=black@0.6:"
            f"boxborderw=10:enable='lt(t,5)'")
    vf = f"bwdif=1:-1:1,{text},fade=in:d={fade},fade=out:st={length - fade}:d={fade},format=yuv420p"
    enc = (["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "21", "-b:v", "0"] if GPU else
           ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"])
    tmp = out.with_suffix(".part.mp4")
    cmd = [FFMPEG, "-hide_banner", "-v", "error", "-fflags", "+discardcorrupt", "-f", "mpegts", "-i", "pipe:0",
           "-t", f"{length:.2f}", "-map", "0:v:0", "-map", "0:a:0", "-vf", vf,
           "-af", f"afade=in:d={fade},afade=out:st={length - fade}:d={fade}", *enc, "-g", "60", "-r", "60000/1001",
           "-c:a", "aac", "-b:a", "192k", "-ac", "2", "-ar", "48000", "-movflags", "+faststart", "-y", str(tmp)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        with open(path, "rb") as f:
            f.seek(off)
            while chunk := f.read(PKT * 5600):  # ffmpeg stops reading once it has `length`
                try:
                    p.stdin.write(chunk)
                except BrokenPipeError:
                    break
        p.stdin.close()
    except BrokenPipeError:
        pass
    if p.wait() != 0 or not tmp.exists():
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed cutting {out.name}")
    tmp.replace(out)


# --- the recap -------------------------------------------------------------------------

class Recap:
    """The recap of one recording, kept in folder `work` (recap.json and the clips)."""

    def __init__(self, path, work, title=None):
        self.path, self.work = Path(path), Path(work)
        self.title = title or self.path.stem
        self.file = self.work / "recap.json"
        try:
            self.state = json.loads(self.file.read_text())
        except (OSError, ValueError):
            self.state = {"recording": str(self.path), "event": None, "clips": {}}

    def _save(self):
        self.work.mkdir(parents=True, exist_ok=True)
        tmp = self.file.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.file)

    def event(self, start_wall):
        league = league_of(self.title)
        teams = teams_of(self.title)
        if not league or not teams:
            raise ValueError(f"not a game the recap knows: {self.title}")
        if not self.state.get("event"):
            ev = find_event(league, *teams, datetime.fromtimestamp(start_wall, timezone.utc))
            if not ev:
                raise ValueError(f"ESPN has no game for {teams[0]} at {teams[1]}")
            self.state.update(event=ev, league=league)
            self._save()
        return self.state["league"], self.state["event"]

    def update(self, log=print):
        """Cut every key play the recording already covers (and re-cut any whose snap moved as
        the game's lag became clearer). Returns the clips made."""
        start_wall, length = recording_clock(self.path)
        league, ev = self.event(start_wall)
        plays = key_plays(get_json(f"{FEED}/{league}/summary?event={ev}"))
        clips, found, made = self.state["clips"], self.state.setdefault("snaps", {}), []
        (self.work / "clips").mkdir(parents=True, exist_ok=True)
        for p in plays:   # look for each play's snap once the recording is well past it
            t = p["wall"] - start_wall
            if p["id"] not in found and t > 0 and t + 40 <= length:
                found[p["id"]] = [round(x - t, 2) for x, _ in snaps(self.path, t + 5, 30)]
        lag = game_lag([found[p["id"]] for p in plays if p["id"] in found])
        self.state["lag"] = None if lag is None else round(lag, 2)
        for p in plays:
            t = p["wall"] - start_wall
            if lag is None:   # not known yet: wide enough for any network
                near, snap = [], None
                a, b = t + LAG_RANGE[0] - PRE, t + LAG_RANGE[1] + p["after"]
            else:
                near = [l for l in found.get(p["id"], []) if abs(l - lag) <= 2.5]
                snap = t + (min(near, key=lambda l: abs(l - lag)) if near else lag)
                a, b = snap - PRE, snap + p["after"]
            if a < 0 or b > length - 2:
                continue  # before the recording, or not recorded yet
            old = clips.get(p["id"])
            if old and (snap is None or (old.get("snap") is not None and abs(old["snap"] - snap) < 1.5)):
                continue
            out = self.work / "clips" / f"{p['id']}.mp4"
            began = time.monotonic()
            cut(self.path, a, b - a, out, [p["text"], f"{p['when']}   {p['score']}"])
            clips[p["id"]] = {**p, "start": round(a, 2), "end": round(b, 2), "snap": snap and round(snap, 2),
                              "seen": bool(near), "clip": out.name, "took": round(time.monotonic() - began, 1)}
            made.append(clips[p["id"]])
            self._save()
            log(f"  {p['when']:9s} {p['kind']:6s} {p['text'][:64]}  ({'snap seen' if near else 'by lag' if lag else 'lag not known yet'}"
                f"{', re-cut' if old else ''})")
        self._save()
        return made

    def playlist(self, level=2, since=None):
        """Clips in order: those at this level or more important, after wall time `since`."""
        return [c for c in sorted(self.state["clips"].values(), key=lambda c: c["wall"])
                if c["level"] <= level and (since is None or c["wall"] >= since)]

    def join(self, out, level=2):
        """The whole recap as one MP4 (the clips are encoded alike, so they're copied)."""
        clips = self.playlist(level)
        if not clips:
            return None
        lst = self.work / "join.txt"
        lst.write_text("".join(f"file '{self.work / 'clips' / c['clip']}'\n" for c in clips))
        subprocess.run([FFMPEG, "-hide_banner", "-v", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                        "-c", "copy", "-movflags", "+faststart", "-y", str(out)], check=True)
        lst.unlink()
        return out
