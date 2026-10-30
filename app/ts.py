"""MPEG-TS handling: find the AC-3 audio tracks and swap their frames for silent ones.

Muting at this level means no re-encoding. Video passes through untouched, timestamps
don't move, and every audio track (main language and SAP) is silenced together.
"""

import logging
import subprocess
import threading
from collections import deque

log = logging.getLogger("admute")

PKT = 188
SYNC = b"\x0b\x77"
# AC-3 bitrates in kbps, indexed by frmsizecod >> 1. At 48 kHz a frame is 4 bytes per kbps.
BITRATES = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512, 576, 640]
# AC-3 acmod -> ffmpeg channel layout
LAYOUTS = {1: "FC", 2: "FL+FR", 3: "FL+FR+FC", 4: "FL+FR+BC", 5: "FL+FR+FC+BC",
           6: "FL+FR+SL+SR", 7: "FL+FR+FC+SL+SR"}


class Entry:
    """One TS packet waiting in the delay buffer."""
    __slots__ = ("t", "pkt", "frames")

    def __init__(self, t, pkt):
        self.t = t            # arrival time (monotonic)
        self.pkt = pkt        # bytearray, edited in place when muting
        self.frames = None    # audio frames that start in this packet


def payload_start(pkt):
    afc = (pkt[3] >> 4) & 3
    if not afc & 1:
        return None
    off = 4 + (1 + pkt[4] if afc & 2 else 0)
    return off if off < PKT else None


def parse_ac3(h):
    """(frame size, key) for an AC-3 frame header, or None. Needs 8 bytes."""
    if len(h) < 8 or h[0] != 0x0B or h[1] != 0x77:
        return None
    fscod, frmsizecod, bsid = h[4] >> 6, h[4] & 0x3F, h[5] >> 3
    if fscod != 0 or frmsizecod >= 38 or bsid > 10:  # 48 kHz plain AC-3 only
        return None
    acmod = h[6] >> 5
    bits, pos = (h[6] << 8) | h[7], 3
    if acmod & 1 and acmod != 1:
        pos += 2  # cmixlev
    if acmod & 4:
        pos += 2  # surmixlev
    if acmod == 2:
        pos += 2  # dsurmod
    lfeon = (bits >> (15 - pos)) & 1
    rate = frmsizecod >> 1
    return BITRATES[rate] * 4, (rate, acmod, lfeon)


class SilentFrames:
    """Silent AC-3 frames that match a stream's bitrate and channel layout, made with ffmpeg."""

    def __init__(self, ffmpeg):
        self.ffmpeg = ffmpeg
        self.frames = {}

    def get(self, key):
        frame = self.frames.get(key)
        if frame is None:
            self.frames[key] = False  # in progress (or failed)
            threading.Thread(target=self._make, args=(key,), daemon=True).start()
        return frame or None

    def _make(self, key):
        rate, acmod, lfeon = key
        kbps, layout = BITRATES[rate], LAYOUTS.get(acmod)
        if not layout:
            log.warning("can't mute AC-3 channel mode %d", acmod)
            return
        layout += "+LFE" if lfeon else ""
        size = kbps * 4
        cmd = [self.ffmpeg, "-v", "error", "-f", "lavfi", "-i", f"anullsrc=r=48000:cl={layout}",
               "-t", "0.5", "-c:a", "ac3", "-b:a", f"{kbps}k", "-f", "ac3", "-"]
        try:
            out = subprocess.run(cmd, capture_output=True, timeout=30, check=True).stdout
        except Exception as exc:
            log.warning("making a silent AC-3 frame failed: %s", exc)
            return
        frame = out[size:2 * size]  # second frame, past any encoder start-up
        if parse_ac3(frame) != (size, key):
            log.warning("ffmpeg's silent frame doesn't match %d kbps %s", kbps, layout)
            return
        self.frames[key] = frame
        log.info("silent AC-3 frame ready: %d kbps, %s", kbps, layout)


def silence(frame, silent):
    """Overwrite one audio frame (spread over several packets) with silence."""
    spans, key = frame
    data = silent.get(key)
    if not data:
        return False
    pos = 0
    for entry, off, n in spans:
        entry.pkt[off:off + n] = data[pos:pos + n]
        pos += n
    return True


class AudioTrack:
    """Splits one AC-3 track into frames and records which packet bytes hold each one."""

    def __init__(self, silent):
        self.silent = silent
        self.buf = bytearray()   # track bytes not yet assigned to a frame
        self.spans = deque()     # (entry, offset, length) for each piece of buf

    def feed(self, entry):
        pkt = entry.pkt
        off = payload_start(pkt)
        if off is None:
            return
        if pkt[1] & 0x40:  # a PES header starts here; skip past it
            if off + 9 > PKT or pkt[off:off + 3] != b"\x00\x00\x01":
                return
            off += 9 + pkt[off + 8]
            if off >= PKT:
                return
        self.buf += pkt[off:]
        self.spans.append((entry, off, PKT - off))
        self._split()

    def _split(self):
        buf = self.buf
        while True:
            if not buf.startswith(SYNC):
                i = buf.find(SYNC)
                if i < 0:
                    i = len(buf) - 1 if buf.endswith(b"\x0b") else len(buf)
                self._take(i)
                if not buf.startswith(SYNC):
                    return
            if len(buf) < 8:
                return
            hdr = parse_ac3(buf)
            if hdr is None:
                self._take(1)
                continue
            size, key = hdr
            if len(buf) < size:
                return
            spans = self._take(size)
            first = spans[0][0]
            if first.frames is None:
                first.frames = []
            first.frames.append((spans, key))
            self.silent.get(key)  # have a matching silent frame ready before it's needed

    def _take(self, n):
        out, left = [], n
        while left:
            entry, off, ln = self.spans[0]
            k = min(left, ln)
            out.append((entry, off, k))
            if k == ln:
                self.spans.popleft()
            else:
                self.spans[0] = (entry, off + k, ln - k)
            left -= k
        del self.buf[:n]
        return out


def _has_ac3_descriptor(desc):
    i = 0
    while i + 2 <= len(desc):
        tag, n = desc[i], desc[i + 1]
        if tag in (0x6A, 0x81) or (tag == 0x05 and desc[i + 2:i + 6] == b"AC-3"):
            return True
        i += 2 + n
    return False


class Demux:
    """Reads the PAT and PMT to find AC-3 tracks, then hands their packets to AudioTrack."""

    def __init__(self, silent):
        self.silent = silent
        self.pmt_pids = set()
        self.audio = {}

    def feed(self, entry):
        pkt = entry.pkt
        pid = ((pkt[1] & 0x1F) << 8) | pkt[2]
        track = self.audio.get(pid)
        if track:
            track.feed(entry)
        elif pkt[1] & 0x40 and (pid == 0 or pid in self.pmt_pids):
            self._psi(pid, pkt)

    def _psi(self, pid, pkt):
        off = payload_start(pkt)
        if off is None:
            return
        off += 1 + pkt[off]  # pointer field
        if off + 12 > PKT:
            return
        table = pkt[off]
        end = min(off + 3 + (((pkt[off + 1] & 0x0F) << 8) | pkt[off + 2]) - 4, PKT)
        if pid == 0 and table == 0:
            for i in range(off + 8, end - 3, 4):
                if (pkt[i] << 8) | pkt[i + 1]:
                    self.pmt_pids.add(((pkt[i + 2] & 0x1F) << 8) | pkt[i + 3])
        elif table == 2:
            i = off + 12 + (((pkt[off + 10] & 0x0F) << 8) | pkt[off + 11])
            while i + 5 <= end:
                stype = pkt[i]
                epid = ((pkt[i + 1] & 0x1F) << 8) | pkt[i + 2]
                n = ((pkt[i + 3] & 0x0F) << 8) | pkt[i + 4]
                desc = bytes(pkt[i + 5:i + 5 + n])
                if (stype == 0x81 or (stype == 0x06 and _has_ac3_descriptor(desc))) and epid not in self.audio:
                    self.audio[epid] = AudioTrack(self.silent)
                    log.info("found AC-3 audio track on PID %d", epid)
                i += 5 + n
