"""Ad-mute proxy for HDHomeRun live TV.

Jellyfin uses this as an M3U tuner. Nothing runs until someone opens a channel. Then it
pulls that channel from the HDHomeRun, holds it back by DELAY seconds, and starts an
ffmpeg that decodes it on the GPU for the ad detector. The detector sees everything
DELAY seconds before the viewer does, so a mute can start right where the ad starts.
When the last viewer leaves (after LINGER seconds), everything for that channel stops:
the tuner is released, ffmpeg exits and the GPU is free again.
"""

import asyncio
import logging
import os
import time
from pathlib import Path

import aiohttp
from aiohttp import web

from detect import AdDetector
from pipeline import FFMPEG, DetectorProc, Pipeline
from ts import SilentFrames

HDHR = os.environ["HDHR_HOST"]                      # HDHomeRun address, set in docker-compose.yaml
PORT = int(os.environ.get("PORT", "5004"))
PUBLIC_URL = os.environ.get("PUBLIC_URL", f"http://admute:{PORT}").rstrip("/")
CHANNELS = os.environ.get("CHANNELS", "favorites")  # comma list like 4.1,5.1, "favorites" or "all"
NAME_SUFFIX = os.environ.get("NAME_SUFFIX", " (ad-mute)")
DELAY = float(os.environ.get("DELAY", "10"))       # seconds the viewer is behind live
LINGER = float(os.environ.get("LINGER", "15"))     # keep a channel open this long after the last viewer
DETECT_AFTER = float(os.environ.get("DETECT_AFTER", "3"))  # ignore quick probes
DETECT_GPU = os.environ.get("DETECT_GPU", "1") == "1"
DATA = Path(os.environ.get("DATA", "/data"))

log = logging.getLogger("admute")


class TunerError(Exception):
    pass


_lineup = {"at": 0.0, "data": []}


async def lineup(http):
    if not _lineup["data"] or time.monotonic() - _lineup["at"] > 600:
        async with http.get(f"http://{HDHR}/lineup.json", timeout=aiohttp.ClientTimeout(total=10)) as r:
            _lineup["data"] = await r.json(content_type=None)
        _lineup["at"] = time.monotonic()
    data = _lineup["data"]
    if CHANNELS == "all":
        return data
    if CHANNELS == "favorites":
        return [c for c in data if c.get("Favorite")]
    wanted = {c.strip() for c in CHANNELS.split(",")}
    return [c for c in data if c["GuideNumber"] in wanted]


class Session:
    """One channel being watched: tuner stream in, delayed and muted stream out to every viewer."""

    def __init__(self, app, chan):
        self.app = app
        self.num, self.name, self.url = chan["GuideNumber"], chan["GuideName"], chan["URL"]
        self.clients = set()
        self.pipe = Pipeline(self.num, app["silent"])
        self.proc = None
        self.started = time.monotonic()
        self.tasks, self.linger, self.resp, self.closed = [], None, None, False

    async def open(self):
        self.resp = await self.app["http"].get(
            self.url, timeout=aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=20))
        if self.resp.status != 200:
            status = self.resp.status
            self.resp.release()
            raise TunerError(status)
        self.tasks = [asyncio.create_task(c) for c in (self._read(), self._emit(), self._run_detector())]
        log.info("%s %s: tuned in, viewers are %.0fs behind live", self.num, self.name, DELAY)

    async def _read(self):
        try:
            async for chunk in self.resp.content.iter_any():
                now = time.monotonic()
                data = self.pipe.ingest(chunk, now)
                if self.proc and data:
                    self.proc.feed(data, now)
            log.warning("%s: tuner stream ended", self.num)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("%s: tuner stream failed: %s", self.num, exc)
        asyncio.create_task(self.close())

    async def _emit(self):
        while True:
            await asyncio.sleep(0.05)
            out = self.pipe.take(time.monotonic() - DELAY)
            if out:
                data = b"".join(e.pkt for e in out)
                for q in list(self.clients):
                    if q.qsize() > 400:  # viewer can't keep up; cut them loose
                        log.warning("%s: dropping a viewer that fell behind", self.num)
                        self.clients.discard(q)
                        while not q.empty():
                            q.get_nowait()
                        q.put_nowait(None)
                    else:
                        q.put_nowait(data)

    async def _run_detector(self):
        await asyncio.sleep(DETECT_AFTER)
        self.pipe.detector = AdDetector(self.num, self.name, DATA)
        gpu = DETECT_GPU
        while True:
            self.proc = DetectorProc(self.pipe.detector, gpu)
            started = time.monotonic()
            try:
                await self.proc.start()
                await self.proc.wait()
            finally:
                self.proc.kill()
                self.proc = None
            if gpu and time.monotonic() - started < 15:
                log.warning("%s: GPU decode failed, switching to CPU", self.num)
                gpu = False
                continue
            log.warning("%s: detector stopped, restarting", self.num)
            await asyncio.sleep(2)

    def add(self, q):
        self.clients.add(q)
        if self.linger:
            self.linger.cancel()
            self.linger = None

    def remove(self, q):
        self.clients.discard(q)
        if not self.clients and not self.closed and not self.linger:
            self.linger = asyncio.create_task(self._linger())

    async def _linger(self):
        await asyncio.sleep(LINGER)
        self.linger = None
        if not self.clients:
            await self.close()

    async def close(self):
        if self.closed:
            return
        self.closed = True
        self.app["sessions"].pop(self.num, None)
        me = asyncio.current_task()
        tasks = [t for t in self.tasks if t is not me]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if self.pipe.detector:
            self.pipe.detector.close()
        if self.resp:
            self.resp.close()
        for q in self.clients:
            q.put_nowait(None)
        log.info("%s %s: closed, tuner released", self.num, self.name)

    def status(self):
        return {
            "channel": self.num, "name": self.name, "viewers": len(self.clients),
            "open_seconds": round(time.monotonic() - self.started),
            "buffered_seconds": round(time.monotonic() - self.pipe.buffer[0].t, 1) if self.pipe.buffer else 0,
            "muted_now": self.pipe.muting, "detector_running": self.proc is not None,
            "detector_gpu": self.proc.gpu if self.proc else None,
            "audio_tracks": len(self.pipe.demux.audio),
            **(self.pipe.detector.status() if self.pipe.detector else {}),
        }


# --- HTTP -------------------------------------------------------------------

async def m3u(request):
    lines = ["#EXTM3U"]
    for c in await lineup(request.app["http"]):
        num, name = c["GuideNumber"], c["GuideName"]
        lines.append(f'#EXTINF:-1 tvg-id="{num}" tvg-chno="{num}" tvg-name="{name}" '
                     f'group-title="Ad-mute",{name}{NAME_SUFFIX}')
        lines.append(f"{PUBLIC_URL}/stream/{num}")
    return web.Response(text="\n".join(lines) + "\n", content_type="audio/x-mpegurl")


async def stream(request):
    num = request.match_info["num"]
    if request.method == "HEAD":
        return web.Response(content_type="video/mp2t")
    app = request.app
    async with app["lock"]:
        sess = app["sessions"].get(num)
        if not sess:
            chans = {c["GuideNumber"]: c for c in await lineup(app["http"])}
            if num not in chans:
                raise web.HTTPNotFound(text=f"channel {num} isn't enabled\n")
            sess = Session(app, chans[num])
            try:
                await sess.open()
            except TunerError as exc:
                log.warning("%s: HDHomeRun refused the stream (HTTP %s), probably no free tuner", num, exc)
                raise web.HTTPServiceUnavailable(text="no free tuner\n")
            app["sessions"][num] = sess
    q = asyncio.Queue()
    sess.add(q)
    log.info("%s: viewer joined (%d watching)", num, len(sess.clients))
    resp = web.StreamResponse(headers={"Content-Type": "video/mp2t", "Cache-Control": "no-cache"})
    try:
        await resp.prepare(request)
        while (data := await q.get()) is not None:
            await resp.write(data)
    except (ConnectionResetError, ConnectionError):
        pass
    finally:
        sess.remove(q)
        log.info("%s: viewer left (%d watching)", num, len(sess.clients))
    return resp


async def status(request):
    return web.json_response({
        "delay_seconds": DELAY, "channels": CHANNELS,
        "sessions": [s.status() for s in request.app["sessions"].values()],
    })


async def mute(request):
    """Manual mute for testing: /mute/4.1?seconds=20 mutes what viewers hear next."""
    sess = request.app["sessions"].get(request.match_info["num"])
    if not sess:
        raise web.HTTPNotFound(text="that channel isn't open\n")
    secs = float(request.query.get("seconds", "20"))
    start = time.monotonic() - DELAY  # the moment viewers are hearing now
    sess.pipe.forced = [*sess.pipe.forced[-5:], (start, start + secs)]
    return web.Response(text=f"muting {sess.num} for {secs:.0f}s\n")


async def startup(app):
    app["http"] = aiohttp.ClientSession()


async def cleanup(app):
    for s in list(app["sessions"].values()):
        await s.close()
    await app["http"].close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logging.getLogger("aiohttp.access").setLevel(logging.WARNING)
    app = web.Application()
    app["sessions"], app["lock"] = {}, asyncio.Lock()
    app["silent"] = SilentFrames(FFMPEG)
    app.router.add_get("/lineup.m3u", m3u)
    app.router.add_get("/stream/{num}", stream)
    app.router.add_get("/status", status)
    app.router.add_get("/mute/{num}", mute)
    app.on_startup.append(startup)
    app.on_cleanup.append(cleanup)
    log.info("ad-mute listening on :%d, channels: %s, delay %.0fs", PORT, CHANNELS, DELAY)
    web.run_app(app, port=PORT, print=None)


if __name__ == "__main__":
    main()
