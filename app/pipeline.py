"""The parts shared by live TV (admute.py) and recordings (replay.py).

Pipeline: delay buffer + audio muting. Packets go in with a time, come out once that time
is older than the cutoff, with any audio frame inside a mute range swapped for silence.
Live TV uses arrival time and a wall-clock cutoff; replay uses the recording's own clock
and a cutoff that follows how far the detector has got, so both make the same decisions.

DetectorProc: the ffmpeg that turns the stream into small gray frames and mono audio.
"""

import asyncio
import logging
import os
from collections import deque

import numpy as np

from detect import AUDIO_RATE, FPS, H, W
from ts import PKT, Demux, Entry, silence

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
        while buf and buf[0].t <= cutoff:
            e = buf.popleft()
            if e.frames:
                mute = (det is not None and det.muted(e.t)) or any(s <= e.t < f for s, f in self.forced)
                if mute:
                    for frame in e.frames:
                        silence(frame, self.silent)
                if mute != self.muting:
                    self.muting = mute
                    log.info("%s: audio %s", self.num, "muted" if mute else "unmuted")
            out.append(e)
        return out

    def newest(self):
        return self.buffer[-1].t if self.buffer else None


class DetectorProc:
    """ffmpeg decoding the stream (on the GPU if asked) into frames and audio for the detector."""

    def __init__(self, det, gpu):
        self.det, self.gpu = det, gpu
        self.p, self.t0, self.tasks, self.dropping = None, None, [], False

    async def start(self):
        r, w = os.pipe()
        if self.gpu:
            hw = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
            vf = f"fps={FPS},scale_cuda={W}:{H},hwdownload,format=nv12,format=gray"
        else:
            hw = []
            vf = f"fps={FPS},scale={W}:{H}:flags=area,format=gray"
        cmd = [FFMPEG, "-hide_banner", "-loglevel", "error", "-nostats", "-fflags", "+discardcorrupt",
               *hw, "-f", "mpegts", "-i", "pipe:0",
               "-map", "0:v:0", "-vf", vf, "-f", "rawvideo", "pipe:1",
               "-map", "0:a:0", "-ac", "1", "-ar", str(AUDIO_RATE), "-f", "s16le", f"pipe:{w}"]
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
        n, size = 0, W * H
        try:
            while True:
                buf = await self.p.stdout.readexactly(size)
                self.det.frame(self.t0 + n / FPS, np.frombuffer(buf, np.uint8).reshape(H, W))
                n += 1
        except asyncio.IncompleteReadError:
            pass

    async def _audio(self, reader):
        n, pending = 0, b""
        block = AUDIO_RATE // 10 * 2
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                return
            pending += chunk
            k = len(pending) // block * block
            if k:
                samples = np.frombuffer(pending[:k], np.int16)
                self.det.audio(self.t0 + n / AUDIO_RATE, samples)
                n += len(samples)
                pending = pending[k:]

    async def _stderr(self):
        shown = 0
        while line := await self.p.stderr.readline():
            text = line.decode(errors="replace").rstrip()
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
