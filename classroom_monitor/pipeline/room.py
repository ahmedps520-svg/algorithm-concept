"""One classroom = one camera = one RoomPipeline thread.

frame -> perception (detect+track+pose) -> behavior engine -> alert engine
      -> clip writer (ring buffer, alert-triggered save)
      -> preview frame for the dashboard grid
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from ..alerts.engine import AlertEngine
from ..alerts.models import Alert
from ..alerts.store import AlertStore
from ..behavior.engine import BehaviorEngine
from ..clips.writer import ClipWriter
from ..clips import imgcodec
from ..config import RoomConfig, SystemConfig
from ..ingestion.source import FrameSource, open_source
from ..perception.backend import PerceptionBackend, build_backend
from ..perception.types import TrackedPerson

log = logging.getLogger(__name__)


@dataclass
class RoomStatus:
    room_id: str
    name: str
    online: bool = False
    fps: float = 0.0
    people: int = 0
    frames: int = 0
    last_frame_ts: float = 0.0
    open_alerts: int = 0
    error: str | None = None
    tracks: list[dict] = field(default_factory=list)


class RoomPipeline:
    def __init__(self, room: RoomConfig, system: SystemConfig, store: AlertStore, alerts: AlertEngine,
                 source: FrameSource | None = None, backend: PerceptionBackend | None = None,
                 verifier=None) -> None:
        self.room = room
        self.system = system
        self.store = store
        self.alerts = alerts
        self.source = source or open_source(room.id, room.camera, system.process_fps)
        self.backend = backend or build_backend(system.models)
        self.behavior = BehaviorEngine(
            room.id, room.resolved_thresholds(system.default_thresholds), room.seat_zones, system.process_fps,
        )
        self.verifier = verifier
        self.clips = ClipWriter(system.clips, store, on_ready=(verifier.enqueue if verifier else None))
        self.status = RoomStatus(room.id, room.name)
        self._preview: tuple[bytes, str] | None = None
        self._preview_lock = threading.Lock()
        self._last_preview = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fps_window: list[float] = []

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self._thread = threading.Thread(target=self.run, name=f"room-{self.room.id}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.source.close()
        self.clips.flush()
        if self._thread:
            self._thread.join(timeout=5)

    def run(self) -> None:
        try:
            for frame in self.source.frames():
                if self._stop.is_set():
                    break
                self.process_frame(frame)
        except Exception as e:  # noqa: BLE001
            log.exception("[%s] pipeline crashed", self.room.id)
            self.status.error = str(e)
        finally:
            self.status.online = False
            self.clips.flush()

    # ---------------------------------------------------------- per frame
    def process_frame(self, frame) -> list[Alert]:
        w, h = frame.size
        people: list[TrackedPerson] = self.backend.process(frame)
        events = self.behavior.update(people, frame.ts, w, h)
        self.clips.observe(frame.ts, frame.image)

        new_alerts: list[Alert] = []
        for ev in events:
            a = self.alerts.handle(ev)
            if a is not None:
                self.clips.start(a)
                new_alerts.append(a)

        self._update_status(frame, people)
        self._maybe_render_preview(frame, people)
        return new_alerts

    def _update_status(self, frame, people: list[TrackedPerson]) -> None:
        s = self.status
        s.online, s.people, s.frames, s.last_frame_ts, s.error = True, len(people), frame.index + 1, frame.ts, None
        self._fps_window.append(time.monotonic())
        self._fps_window = self._fps_window[-30:]
        if len(self._fps_window) > 1:
            span = self._fps_window[-1] - self._fps_window[0]
            s.fps = round((len(self._fps_window) - 1) / span, 1) if span > 0 else 0.0
        w, h = frame.size
        s.tracks = [
            {"id": p.track_id, "box": [round(p.box.x1 / w, 3), round(p.box.y1 / h, 3), round(p.box.x2 / w, 3), round(p.box.y2 / h, 3)]}
            for p in people
        ]

    def _maybe_render_preview(self, frame, people: list[TrackedPerson]) -> None:
        now = time.monotonic()
        if now - self._last_preview < 1.0 / max(self.system.dashboard.preview_fps, 0.1):
            return
        self._last_preview = now
        img = frame.image.copy()
        for p in people:
            _draw_box(img, p.box.x1, p.box.y1, p.box.x2, p.box.y2)
        data, ext = imgcodec.encode(img, quality=60)
        with self._preview_lock:
            self._preview = (data, "image/jpeg" if ext == "jpg" else "image/png")

    def preview(self) -> tuple[bytes, str] | None:
        with self._preview_lock:
            return self._preview


def _draw_box(img: np.ndarray, x1: float, y1: float, x2: float, y2: float, colour=(0, 255, 0), t: int = 2) -> None:
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(max(0, min(v, lim - 1))) for v, lim in ((x1, w), (y1, h), (x2, w), (y2, h)))
    img[y1:y1 + t, x1:x2] = colour
    img[max(y2 - t, 0):y2, x1:x2] = colour
    img[y1:y2, x1:x1 + t] = colour
    img[y1:y2, max(x2 - t, 0):x2] = colour
