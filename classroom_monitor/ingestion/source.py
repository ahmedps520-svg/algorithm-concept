"""Frame sources.

Every source yields ``Frame`` objects at (roughly) the configured processing FPS. RTSP and
file sources need OpenCV (``pip install .[vision]``). ``SyntheticSource`` needs nothing and
drives a scripted scene so the whole pipeline can run without cameras or models.
"""
from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Iterator, Protocol

import numpy as np

from ..config import CameraConfig

log = logging.getLogger(__name__)


@dataclass
class Frame:
    room_id: str
    index: int
    ts: float                     # wall-clock seconds
    image: np.ndarray             # HxWx3 uint8 BGR
    # Ground-truth boxes for synthetic scenes (id, x1,y1,x2,y2 px). Unused with real cameras.
    synthetic_boxes: list[tuple[int, float, float, float, float]] = field(default_factory=list)
    synthetic_keypoints: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.image.shape[:2]
        return w, h


class FrameSource(Protocol):
    def frames(self) -> Iterator[Frame]: ...
    def close(self) -> None: ...


class _Pacer:
    """Sleeps so that frames are delivered no faster than ``fps``."""

    def __init__(self, fps: float) -> None:
        self.interval = 1.0 / fps if fps > 0 else 0.0
        self._next = time.monotonic()

    def wait(self) -> None:
        if self.interval <= 0:
            return
        now = time.monotonic()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = max(self._next + self.interval, now)


# --------------------------------------------------------------------------------------
# OpenCV-backed sources
# --------------------------------------------------------------------------------------
class _CvSource:
    def __init__(self, room_id: str, cfg: CameraConfig, fps: float, loop: bool = False) -> None:
        try:
            import cv2  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("OpenCV is required for camera/file sources: pip install .[vision]") from e
        self.room_id = room_id
        self.cfg = cfg
        self.fps = fps
        self.loop = loop
        self._cap = None
        self._closed = False

    @property
    def is_live(self) -> bool:
        return self.cfg.url.startswith("rtsp://")

    def _open(self):
        import cv2

        if self.is_live:
            # Force TCP so packets aren't dropped on lossy school Wi-Fi; UDP is opt-in.
            os.environ.setdefault(
                "OPENCV_FFMPEG_CAPTURE_OPTIONS", f"rtsp_transport;{self.cfg.transport}|stimeout;5000000"
            )
        cap = cv2.VideoCapture(self.cfg.url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            raise ConnectionError(f"[{self.room_id}] cannot open {self.cfg.url}")
        # Keep the buffer tiny so we process the newest frame, not a backlog.
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def frames(self) -> Iterator[Frame]:
        import cv2

        pacer = _Pacer(self.fps)
        index = 0
        while not self._closed:
            try:
                self._cap = self._open()
            except ConnectionError as e:
                log.warning("%s; retrying in %.1fs", e, self.cfg.reconnect_delay_s)
                time.sleep(self.cfg.reconnect_delay_s)
                continue
            while not self._closed:
                ok, img = self._cap.read()
                if not ok:
                    if self.loop and not self.is_live:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    log.warning("[%s] stream ended/dropped; reconnecting", self.room_id)
                    break
                if self.cfg.scale != 1.0:
                    img = cv2.resize(img, None, fx=self.cfg.scale, fy=self.cfg.scale, interpolation=cv2.INTER_AREA)
                yield Frame(room_id=self.room_id, index=index, ts=time.time(), image=img)
                index += 1
                pacer.wait()
            self._cap.release()
            if not self.loop and not self.is_live:
                return
            time.sleep(self.cfg.reconnect_delay_s)

    def close(self) -> None:
        self._closed = True
        if self._cap is not None:
            self._cap.release()


class RTSPSource(_CvSource):
    """Live camera. Reconnects forever on drop."""


class VideoFileSource(_CvSource):
    """Recorded file, useful for replaying incidents and tuning thresholds."""


# --------------------------------------------------------------------------------------
# Synthetic source: scripted actors so the pipeline runs end-to-end with no hardware
# --------------------------------------------------------------------------------------
@dataclass
class Actor:
    """Scripted person. ``x, y`` is the base position; per-frame ``jitter`` is added on top
    (not accumulated, so seated actors stay put). ``move_to`` glides toward a target at
    ``speed`` frame-widths per second so tracks stay continuous."""

    id: int
    x: float          # base centre, normalised
    y: float
    w: float = 0.10   # box size, normalised
    h: float = 0.28
    head_drop: float = 0.0     # 0 = upright, 1 = head fully below shoulders
    arms_up: bool = False
    jitter: float = 0.0        # per-frame positional noise amplitude (normalised, not accumulated)
    speed: float = 0.3         # glide speed toward target (frame widths / s)
    target: tuple[float, float] | None = None

    def move_to(self, x: float, y: float) -> None:
        self.target = (x, y)

    def step(self, dt: float) -> None:
        if self.target is None:
            return
        tx, ty = self.target
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        max_step = self.speed * dt
        if dist <= max_step:
            self.x, self.y, self.target = tx, ty, None
        else:
            self.x += dx / dist * max_step
            self.y += dy / dist * max_step

    def rendered(self, rng: np.random.Generator) -> tuple[float, float]:
        if self.jitter <= 0:
            return self.x, self.y
        return (float(np.clip(self.x + rng.normal(0, self.jitter), 0.02, 0.98)),
                float(np.clip(self.y + rng.normal(0, self.jitter), 0.02, 0.98)))

    def keypoints(self, rx: float, ry: float, W: int, H: int) -> np.ndarray:
        """COCO-17 keypoints (x, y, conf) in pixels for a crude stick figure at rendered pos."""
        cx, cy, w, h = rx * W, ry * H, self.w * W, self.h * H
        top = cy - h / 2
        shoulder_y = top + 0.25 * h
        nose_y = top + 0.08 * h + self.head_drop * 0.35 * h
        hip_y = top + 0.6 * h
        wrist_y = (top + 0.05 * h) if self.arms_up else (top + 0.55 * h)
        kp = np.zeros((17, 3), dtype=np.float32)
        kp[:, 2] = 0.9
        kp[0] = (cx, nose_y, 0.9)                              # nose
        kp[1] = (cx - 0.04 * w, nose_y - 0.02 * h, 0.8)        # eyes
        kp[2] = (cx + 0.04 * w, nose_y - 0.02 * h, 0.8)
        kp[3] = (cx - 0.15 * w, nose_y, 0.7)                   # ears
        kp[4] = (cx + 0.15 * w, nose_y, 0.7)
        kp[5] = (cx - 0.35 * w, shoulder_y, 0.9)               # shoulders
        kp[6] = (cx + 0.35 * w, shoulder_y, 0.9)
        kp[7] = (cx - 0.45 * w, shoulder_y + 0.15 * h, 0.8)    # elbows
        kp[8] = (cx + 0.45 * w, shoulder_y + 0.15 * h, 0.8)
        kp[9] = (cx - 0.5 * w, wrist_y, 0.8)                   # wrists
        kp[10] = (cx + 0.5 * w, wrist_y, 0.8)
        kp[11] = (cx - 0.25 * w, hip_y, 0.8)                   # hips
        kp[12] = (cx + 0.25 * w, hip_y, 0.8)
        kp[13] = (cx - 0.25 * w, hip_y + 0.2 * h, 0.6)         # knees
        kp[14] = (cx + 0.25 * w, hip_y + 0.2 * h, 0.6)
        kp[15] = (cx - 0.25 * w, top + h, 0.5)                 # ankles
        kp[16] = (cx + 0.25 * w, top + h, 0.5)
        return kp


class SyntheticSource:
    """Scripted classroom. ``script(t, actors)`` mutates actors over time; the source renders
    coloured boxes and publishes ground-truth boxes/keypoints for the stub perception backend."""

    def __init__(
        self,
        room_id: str,
        actors: list[Actor],
        script=None,
        fps: float = 10.0,
        size: tuple[int, int] = (640, 360),
        duration_s: float | None = None,
        realtime: bool = True,
        seed: int = 0,
    ) -> None:
        self.room_id = room_id
        self.actors = actors
        self.script = script or (lambda t, actors: None)
        self.fps = fps
        self.size = size
        self.duration_s = duration_s
        self.realtime = realtime
        self._rng = np.random.default_rng(seed)
        self._closed = False

    def frames(self) -> Iterator[Frame]:
        W, H = self.size
        pacer = _Pacer(self.fps if self.realtime else 0)
        index = 0
        t0 = time.time()
        while not self._closed:
            t = index / self.fps
            if self.duration_s is not None and t >= self.duration_s:
                return
            self.script(t, self.actors)
            img = np.full((H, W, 3), 40, dtype=np.uint8)
            boxes, kps = [], {}
            for a in self.actors:
                a.step(1.0 / self.fps)
                rx, ry = a.rendered(self._rng)
                x1, y1 = int((rx - a.w / 2) * W), int((ry - a.h / 2) * H)
                x2, y2 = int((rx + a.w / 2) * W), int((ry + a.h / 2) * H)
                colour = (60 + 40 * a.id % 200, 120, 200 - 30 * a.id % 200)
                img[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = colour
                boxes.append((a.id, x1, y1, x2, y2))
                kps[a.id] = a.keypoints(rx, ry, W, H)
            yield Frame(
                room_id=self.room_id, index=index, ts=t0 + t if self.realtime else t,
                image=img, synthetic_boxes=boxes, synthetic_keypoints=kps,
            )
            index += 1
            pacer.wait()

    def close(self) -> None:
        self._closed = True


def open_source(room_id: str, cfg: CameraConfig, fps: float) -> FrameSource:
    if cfg.url == "synthetic":
        from .scenarios import default_classroom
        return default_classroom(room_id, fps)
    if cfg.url.startswith("rtsp://"):
        return RTSPSource(room_id, cfg, fps)
    return VideoFileSource(room_id, cfg, fps, loop=True)
