"""Clip capture: pre-roll from the ring buffer + post-roll from live frames, written as an
image sequence directory (default) or an mp4 (needs OpenCV).

Clips are only ever created in response to an alert. There is no continuous recording."""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from pathlib import Path
from typing import Callable

import numpy as np

from ..alerts.models import Alert
from ..alerts.store import AlertStore
from ..config import ClipConfig
from ..ingestion.ring_buffer import FrameRingBuffer
from . import imgcodec

log = logging.getLogger(__name__)

ClipReadyCallback = Callable[[Alert, list[tuple[float, np.ndarray]]], None]


class ClipRecorder:
    """Accumulates frames for one clip until post-roll is satisfied, then finalises."""

    def __init__(self, cfg: ClipConfig, store: AlertStore, alert: Alert, pre: list[tuple[float, np.ndarray]],
                 clock: Callable[[], float] = time.time, on_ready: ClipReadyCallback | None = None) -> None:
        self.cfg = cfg
        self.store = store
        self.alert = alert
        self.clock = clock
        self.on_ready = on_ready
        self.clip_id = uuid.uuid4().hex[:12]
        self.frames: list[tuple[float, np.ndarray]] = list(pre)
        self.trigger_ts = alert.first_ts
        self.done = False
        self._lock = threading.Lock()

    def push(self, ts: float, image: np.ndarray) -> None:
        with self._lock:
            if self.done:
                return
            self.frames.append((ts, image))
            if ts - self.trigger_ts >= self.cfg.post_roll_s:
                self._finalise()

    def abort_and_finalise(self) -> None:
        with self._lock:
            if not self.done:
                self._finalise()

    def _finalise(self) -> None:
        self.done = True
        if not self.frames:
            return
        room_dir = self.cfg.directory / self.alert.room_id / self.clip_id
        room_dir.mkdir(parents=True, exist_ok=True)
        if self.cfg.format == "mp4":
            path = self._write_mp4(room_dir)
        else:
            path = self._write_image_dir(room_dir)
        start, end = self.frames[0][0], self.frames[-1][0]
        self.store.add_clip(self.clip_id, self.alert.id, self.alert.room_id, path, self.clock(), start, end, len(self.frames))
        log.info("clip %s saved for alert %s (%d frames, %.1fs)", self.clip_id, self.alert.id, len(self.frames), end - start)
        if self.on_ready is not None:
            try:
                self.on_ready(self.alert, list(self.frames))
            except Exception:  # noqa: BLE001
                log.exception("clip-ready callback failed for alert %s", self.alert.id)

    def _write_image_dir(self, d: Path) -> Path:
        frames_dir = d / "frames"
        frames_dir.mkdir(exist_ok=True)
        index = []
        for i, (ts, img) in enumerate(self.frames):
            data, ext = imgcodec.encode(img)
            name = f"{i:06d}.{ext}"
            (frames_dir / name).write_bytes(data)
            index.append({"file": f"frames/{name}", "ts": ts, "offset_s": round(ts - self.trigger_ts, 3)})
        (d / "index.json").write_text(json.dumps({
            "clip_id": self.clip_id, "alert_id": self.alert.id, "room_id": self.alert.room_id,
            "behavior": self.alert.behavior, "trigger_ts": self.trigger_ts, "fps": self.cfg.fps, "frames": index,
        }, indent=1))
        return d

    def _write_mp4(self, d: Path) -> Path:
        import cv2  # optional dependency

        path = d / "clip.mp4"
        h, w = self.frames[0][1].shape[:2]
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.cfg.fps, (w, h))
        for _, img in self.frames:
            vw.write(img)
        vw.release()
        return path


class ClipWriter:
    """Owns the per-room ring buffer and any in-flight recorders."""

    def __init__(self, cfg: ClipConfig, store: AlertStore, clock: Callable[[], float] = time.time,
                 on_ready: ClipReadyCallback | None = None) -> None:
        self.cfg = cfg
        self.store = store
        self.clock = clock
        self.on_ready = on_ready
        self.ring = FrameRingBuffer(cfg.pre_roll_s, cfg.fps)
        self._active: list[ClipRecorder] = []
        self._lock = threading.Lock()
        self._last_kept_ts = 0.0

    def observe(self, ts: float, image: np.ndarray) -> None:
        """Feed every processed frame. Down-samples to the clip FPS."""
        if ts - self._last_kept_ts < 1.0 / self.cfg.fps - 1e-6:
            return
        self._last_kept_ts = ts
        self.ring.push(ts, image)
        with self._lock:
            for r in self._active:
                r.push(ts, image)
            self._active = [r for r in self._active if not r.done]

    def start(self, alert: Alert) -> ClipRecorder:
        rec = ClipRecorder(self.cfg, self.store, alert, self.ring.snapshot(), self.clock, self.on_ready)
        with self._lock:
            self._active.append(rec)
        return rec

    def flush(self) -> None:
        with self._lock:
            for r in self._active:
                r.abort_and_finalise()
            self._active.clear()
