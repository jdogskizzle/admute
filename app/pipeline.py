"""The parts shared by live TV (admute.py) and recordings (replay.py).

Pipeline: delay buffer + audio muting. Packets go in with a time, come out once that time
is older than the cutoff, with any audio frame inside a mute range swapped for silence.
Live TV uses arrival time and a wall-clock cutoff; replay uses the recording's own clock
and a cutoff that follows how far the detector has got, so both make the same decisions.
Whether an audio frame is muted goes by its own timestamp (PTS), on the detector's clock:
a moment's sound arrives about half a second after its picture, so going by arrival time
would put every mute edge off by that much.

DetectorProc: the ffmpeg that turns the stream into small gray frames and mono audio.
"""

import asyncio
import logging
import os
import re
from collections import deque

import numpy as np

from detect import AUDIO_RATE, FPS, H, W
from ts import PKT, PTS_WRAP, Demux, Entry, pts_seconds, silence

log = logging.getLogger("admute")

FFMPEG = os.environ.get("FFMPEG", "/usr/lib/jellyfin-ffmpeg/ffmpeg")


class Pipeline:
    def __init__(self, num, silent):
        self.num = num
        self.silent = silent
        self.buffer = deque()
        self.demux = Demux(silent)
        self.detector = None
        self.forced = []      # manual mute ranges
        self.muting = False
        self._rest = b""

    def ingest(self, chunk, t):
        """Add stream bytes that arrived at time t. Returns the packet-aligned bytes taken."""
        data = self._rest + chunk if self._rest else chunk
        off, end, start = 0, len(data) - PKT, None
        while off <= end:
            if data[off] != 0x47:  # lost packet alignment; find the next sync byte
                nxt = data.find(b"\x47", off + 1)
                off = nxt if nxt >= 0 else len(data)
                continue
            if start is None:
                start = off
            e = Entry(t, bytearray(data[off:off + PKT]))
            self.demux.feed(e)
            self.buffer.append(e)
            off += PKT
        self._rest = data[off:]
        return data[start:off] if start is not None else b""

    def take(self, cutoff):
        """Remove and return (time, packet) for everything at or before cutoff, muted as needed."""
        det, buf, out = self.detector, self.buffer, []
        anchor = det.anchor if det is not None else None
        while buf and buf[0].t <= cutoff:
            e = buf.popleft()
            if e.frames:
                mute = False
                for frame in e.frames:
                    pts = frame[2]
                    t = pts_seconds(pts, anchor) if anchor and pts is not None else e.t
                    if (det is not None and det.muted(t)) or any(s <= e.t < f for s, f in self.forced):
                        silence(frame, self.silent)
                        mute = True
                    if anchor and pts is not None and abs(t - anchor[1]) > 3600:
                        anchor = det.anchor = (pts, t)  # keep it near, for streams longer than the PTS wrap
                if mute != self.muting:
                    self.muting = mute
                    log.info("%s: audio %s", self.num, "muted" if mute else "unmuted")
            out.append(e)
        return out

    def newest(self):
        return self.buffer[-1].t if self.buffer else None


FIRST = re.compile(r"\] \[info\] n:\s*0 pts:\s*-?\d+ pts_time:(-?[\d.]+)")


class DetectorProc:
    """ffmpeg decoding the stream (on the GPU if asked) into frames and audio for the detector."""

    def __init__(self, det, gpu):
        self.det, self.gpu = det, gpu
        self.p, self.t0, self.tasks, self.dropping = None, None, [], False
        self.first = {}  # PTS (seconds) of the first picture ("v") and sound ("a") ffmpeg decoded
        self.timed = asyncio.Event()

    async def start(self):
        r, w = os.pipe()
        if self.gpu:
            hw = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
            # Decode on the GPU but shrink on the CPU, the same way as without it: the scene
            # check learned from pictures shrunk like that, and scale_cuda's differ noticeably.
            # Only FPS frames a second are copied off the GPU, so this is cheap.
            vf = f"fps={FPS},hwdownload,format=nv12,scale={2 * W}:{2 * H}:flags=area,format=nv12,showinfo"
        else:
            hw = []
            vf = f"fps={FPS},scale={2 * W}:{2 * H}:flags=area,format=nv12,showinfo"
        # showinfo/ashowinfo report the stream timestamps of the first picture and sound
        # (-copyts keeps them as broadcast); passthrough stops ffmpeg padding from zero.
        # aresample fills gaps in the sound (damaged frames), so counting samples keeps time
        # with the timestamps: the fps filter does the same for the picture.
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "level+info", "-nostats", "-fflags", "+discardcorrupt",
               "-copyts", *hw, "-f", "mpegts", "-i", "pipe:0",
               "-map", "0:v:0", "-vf", vf, "-fps_mode", "passthrough", "-f", "rawvideo", "pipe:1",
               "-map", "0:a:0", "-af", "ashowinfo,aresample=async=1000", "-ac", "1", "-ar", str(AUDIO_RATE), "-f", "s16le", f"pipe:{w}"]
        try:
            self.p = await asyncio.create_subprocess_exec(
                *cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE, pass_fds=(w,))
        finally:
            os.close(w)
        audio = asyncio.StreamReader(limit=1 << 20)
        await asyncio.get_running_loop().connect_read_pipe(
            lambda: asyncio.StreamReaderProtocol(audio), os.fdopen(r, "rb", buffering=0))
        self.tasks = [asyncio.create_task(c) for c in (self._video(), self._audio(audio), self._stderr())]
        log.info("%s: detector started (%s decode)", self.det.num, "GPU" if self.gpu else "CPU")

    def _ok(self):
        p = self.p
        return p and p.returncode is None and not p.stdin.is_closing()

    def feed(self, data, t):
        """Live: never block the tuner; skip data if ffmpeg falls far behind."""
        if not self._ok():
            return
        if self.t0 is None:
            self.t0 = t
        if self.p.stdin.transport.get_write_buffer_size() > 16 << 20:
            if not self.dropping:
                log.warning("%s: detector is falling behind, skipping data", self.det.num)
            self.dropping = True
            return
        self.dropping = False
        self.p.stdin.write(data)

    async def feed_wait(self, data, t):
        """Replay: wait for ffmpeg instead of skipping, so every frame is looked at."""
        if not self._ok():
            raise RuntimeError("detector ffmpeg exited")
        if self.t0 is None:
            self.t0 = t
        self.p.stdin.write(data)
        await self.p.stdin.drain()

    async def finish(self):
        """Replay: end of input. Let ffmpeg flush its last frames, then stop."""
        if self._ok():
            self.p.stdin.close()
        await self.p.wait()
        await asyncio.wait(self.tasks[:2], timeout=10)
        self.kill()

    async def _video(self):
        # nv12 at twice the analysis size: brightness, then colour at half size. The full
        # size is only used to recognise the small on-screen graphics.
        n, px = 0, 4 * W * H
        size = px * 3 // 2
        try:
            while True:
                buf = await self.p.stdout.readexactly(size)
                a = np.frombuffer(buf, np.uint8)
                hi = a[:px].reshape(2 * H, 2 * W)
                lo = hi.reshape(H, 2, W, 2).mean(axis=(1, 3)).astype(np.uint8)
                uvf = a[px:].reshape(H, W, 2)  # colour at the detector's size (for the scene check)
                self.det.frame(self.t0 + n / FPS, lo, uvf[::2, ::2], hi, uvf)
                n += 1
        except asyncio.IncompleteReadError:
            pass

    async def _audio(self, reader):
        n, pending = 0, b""
        block = AUDIO_RATE // 10 * 2
        # The sound usually starts a little before or after the first picture; place it by
        # their timestamps (wait briefly for both; without them, assume they start together).
        try:
            await asyncio.wait_for(self.timed.wait(), 10)
        except asyncio.TimeoutError:
            pass
        shift = self.first["a"] - self.first["v"] if self.timed.is_set() else 0.0
        if abs(shift) > 5:
            shift = 0.0
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                return
            pending += chunk
            k = len(pending) // block * block
            if k:
                samples = np.frombuffer(pending[:k], np.int16)
                self.det.audio(self.t0 + shift + n / AUDIO_RATE, samples)
                n += len(samples)
                pending = pending[k:]

    async def _stderr(self):
        shown = 0
        while line := await self.p.stderr.readline():
            text = line.decode(errors="replace").rstrip()
            if len(self.first) < 2 and "showinfo" in text:
                m = FIRST.search(text)
                if m and (kind := "a" if "ashowinfo" in text else "v") not in self.first:
                    self.first[kind] = float(m.group(1))
                    if kind == "v":
                        # Picture n is at t0 + n / FPS on the detector's clock, so this anchors
                        # stream timestamps to it.
                        self.det.anchor = (round(self.first["v"] * 90000) % PTS_WRAP, self.t0)
                    if len(self.first) == 2:
                        self.timed.set()
                        log.info("%s: muting by stream timestamps (sound starts %.2fs %s the picture)", self.det.num,
                                 abs(self.first["a"] - self.first["v"]),
                                 "before" if self.first["a"] < self.first["v"] else "after")
                continue
            if not ("[error]" in text or "[fatal]" in text):
                continue
            if "Invalid frame dimensions 0x0" in text or "Last message repeated" in text:
                continue  # harmless: stream joined mid-picture
            if shown < 20:
                log.warning("%s: ffmpeg: %s", self.det.num, text)
                shown += 1

    async def wait(self):
        try:
            await self.p.wait()
            await asyncio.wait(self.tasks, timeout=2)
        finally:
            self.kill()

    def kill(self):
        for t in self.tasks:
            t.cancel()
        if self.p and self.p.returncode is None:
            self.p.kill()
