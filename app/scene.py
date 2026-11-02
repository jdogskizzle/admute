"""Scene check: does this picture look like the broadcast or like an ad?

An image model (CLIP ViT-B/32, run with onnxruntime) turns one frame a second into 512
numbers describing what's on screen. A small classifier learned from reviewed recordings
(replay.py train, or the studio) turns those into the chance it's an ad. Unlike the logo
check, this still knows it's the game during stat cards, replays and studio cut-ins with
every graphic gone, and it knows the commentators are back before the scorebug is.
"""

import logging
import os
import re
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger("admute")

MODEL = os.environ.get("SCENE_MODEL", "/opt/models/scene.onnx")
CLASSIFIER = Path(os.environ.get("DATA", "/data")) / "scene" / "classifier.npz"
THREADS = int(os.environ.get("SCENE_THREADS", "2"))

_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32).reshape(1, 3, 1, 1)
_session, _lock = None, threading.Lock()


def available():
    return os.path.exists(MODEL)


def _model():
    global _session
    with _lock:
        if _session is None:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = THREADS
            opts.inter_op_num_threads = 1
            _session = ort.InferenceSession(MODEL, opts, providers=["CPUExecutionProvider"])
        return _session


def _resize_index(n_in, n_out):
    """Linear interpolation positions for resizing an axis from n_in to n_out."""
    x = (np.arange(n_out) + 0.5) * n_in / n_out - 0.5
    x0 = np.clip(np.floor(x).astype(int), 0, n_in - 1)
    x1 = np.clip(x0 + 1, 0, n_in - 1)
    return x0, x1, (x - x0).clip(0, 1).astype(np.float32)


_idx = {}


def picture(y, uv):
    """224x224 RGB (float, 0..1) from the detector's brightness (H x W) and colour (H x W x 2)
    pictures. The whole frame is squeezed in, so nothing at the edges is cut off."""
    h, w = y.shape
    if (h, w) not in _idx:
        _idx[(h, w)] = (_resize_index(h, 224), _resize_index(w, 224))
    (r0, r1, rf), (c0, c1, cf) = _idx[(h, w)]
    yy = y.astype(np.float32) - 16
    u = uv[..., 0].astype(np.float32) - 128
    v = uv[..., 1].astype(np.float32) - 128
    rgb = np.stack([1.164 * yy + 1.793 * v, 1.164 * yy - 0.213 * u - 0.533 * v, 1.164 * yy + 2.112 * u], -1) / 255
    rgb = rgb[r0] * (1 - rf)[:, None, None] + rgb[r1] * rf[:, None, None]
    rgb = rgb[:, c0] * (1 - cf)[None, :, None] + rgb[:, c1] * cf[None, :, None]
    return rgb.clip(0, 1)


def features(rgb):
    """512 numbers describing the picture (unit length)."""
    x = (rgb.transpose(2, 0, 1)[None].astype(np.float32) - _MEAN) / _STD
    return _model().run(None, {"image": x})[0][0]


class Classifier:
    """Chance that a picture is an ad, from its features: a logistic regression."""

    def __init__(self, w, b, info=None):
        self.w, self.b, self.info = np.asarray(w, np.float32), float(b), info or {}

    def p(self, f):
        z = np.asarray(f, np.float32) @ self.w + self.b
        return 1 / (1 + np.exp(-np.clip(z, -30, 30)))

    @staticmethod
    def load(path=CLASSIFIER):
        try:
            with np.load(path, allow_pickle=False) as d:
                import json
                return Classifier(d["w"], float(d["b"]), json.loads(str(d["info"])) if "info" in d else {})
        except FileNotFoundError:
            return None
        except Exception as exc:
            log.warning("couldn't load the scene classifier: %s", exc)
            return None

    def save(self, path=CLASSIFIER):
        import json
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(tmp, w=self.w, b=self.b, info=json.dumps(self.info))
        os.replace(tmp, path)

    @staticmethod
    def train(X, y, l2=2.0, iters=30):
        """Class-balanced logistic regression by Newton's method (512 inputs: quick)."""
        X = np.asarray(X, np.float64)
        y = np.asarray(y, np.float64)
        n, k = X.shape
        Xb = np.hstack([X, np.ones((n, 1))])
        pos = max(1.0, y.sum())
        neg = max(1.0, n - y.sum())
        sw = np.where(y > 0, n / (2 * pos), n / (2 * neg))
        wb = np.zeros(k + 1)
        reg = np.full(k + 1, l2)
        reg[-1] = 0
        for _ in range(iters):
            z = np.clip(Xb @ wb, -30, 30)
            p = 1 / (1 + np.exp(-z))
            g = Xb.T @ (sw * (p - y)) + reg * wb
            H = (Xb * (sw * p * (1 - p))[:, None]).T @ Xb + np.diag(reg)
            step = np.linalg.solve(H, g)
            wb -= step
            if np.abs(step).max() < 1e-6:
                break
        return Classifier(wb[:-1], wb[-1])


def show_title(name):
    """The program's title from a recording name: "NFL Football 2026_11_01_... - A at B" or
    "NFL Football - A at B 2026-11-01" -> "NFL Football"."""
    return re.split(r"\s+(?:\d{4}[_-]\d|-\s)", name.strip(), maxsplit=1)[0].strip()


def covers(classifier, title):
    """Whether the classifier learned from this kind of program (by guide title)."""
    return bool(classifier and title and title in classifier.info.get("titles", []))


class Guide:
    """Whether a sports event is on a channel now, from an XMLTV guide (GUIDE_XMLTV). The scene
    check was learned from games, so live TV only uses it during programs the guide calls
    sports (and for a while after, since games run long)."""

    GRACE = 30 * 60  # seconds after a sports program's scheduled end

    def __init__(self, path):
        self.path, self.loaded, self.mtime, self.sports = path, 0.0, None, {}

    def _load(self):
        import time
        import xml.etree.ElementTree as ET
        from datetime import datetime
        if time.monotonic() - self.loaded < 600:
            return
        self.loaded = time.monotonic()
        try:
            mtime = os.path.getmtime(self.path)
            if mtime == self.mtime:
                return
            sports = {}
            for p in ET.parse(self.path).getroot().iter("programme"):
                if any((c.text or "").strip().lower() == "sports" for c in p.findall("category")):
                    a = datetime.strptime(p.get("start"), "%Y%m%d%H%M%S %z").timestamp()
                    b = datetime.strptime(p.get("stop"), "%Y%m%d%H%M%S %z").timestamp()
                    sports.setdefault(p.get("channel"), []).append((a, b, (p.findtext("title") or "").strip()))
            self.sports, self.mtime = sports, mtime
        except Exception as exc:
            log.warning("couldn't read the guide %s: %s", self.path, exc)

    def sports_now(self, channel, now):
        """Title of the sports program on now (or that ended less than GRACE ago), else None."""
        self._load()
        on = [(a, t) for a, b, t in self.sports.get(str(channel), []) if a <= now < b + self.GRACE]
        return max(on)[1] if on else None


_guide = None


def guide():
    global _guide
    path = os.environ.get("GUIDE_XMLTV", "").strip()
    if not path:
        return None
    if _guide is None or _guide.path != path:
        _guide = Guide(path)
    return _guide
