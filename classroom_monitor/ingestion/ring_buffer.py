"""Pre-roll buffer. Keeps only the last N seconds of (downsampled) frames in memory so an alert
can be saved with context. This is the *only* place raw frames are retained before an alert;
there is no continuous recording."""
from __future__ import annotations

import threading
from collections import deque

import numpy as np


class FrameRingBuffer:
    def __init__(self, seconds: float, fps: float) -> None:
        self.capacity = max(1, int(seconds * fps))
        self._buf: deque[tuple[float, np.ndarray]] = deque(maxlen=self.capacity)
        self._lock = threading.Lock()

    def push(self, ts: float, image: np.ndarray) -> None:
        with self._lock:
            self._buf.append((ts, image))

    def snapshot(self) -> list[tuple[float, np.ndarray]]:
        with self._lock:
            return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)
