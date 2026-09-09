"""Minimal IoU tracker with a short memory. Good enough for the stub backend and as a fallback;
production runs use ByteTrack through ultralytics (see ``UltralyticsBackend``)."""
from __future__ import annotations

from dataclasses import dataclass

from .types import Detection, TrackedPerson


@dataclass
class _Track:
    id: int
    box: Detection
    age: int = 1
    missed: int = 0


class IoUTracker:
    def __init__(self, iou_threshold: float = 0.2, max_missed: int = 15) -> None:
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self._tracks: list[_Track] = []
        self._next_id = 1

    def update(self, detections: list[Detection]) -> list[TrackedPerson]:
        # Greedy matching by IoU, highest first.
        pairs = []
        for ti, t in enumerate(self._tracks):
            for di, d in enumerate(detections):
                iou = _iou(t.box, d)
                if iou >= self.iou_threshold:
                    pairs.append((iou, ti, di))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t = self._tracks[ti]
            t.box, t.age, t.missed = detections[di], t.age + 1, 0
        for ti, t in enumerate(self._tracks):
            if ti not in used_t:
                t.missed += 1
        for di, d in enumerate(detections):
            if di not in used_d:
                self._tracks.append(_Track(self._next_id, d))
                self._next_id += 1
        self._tracks = [t for t in self._tracks if t.missed <= self.max_missed]
        return [TrackedPerson(t.id, t.box, age=t.age) for t in self._tracks if t.missed == 0]


def _iou(a: Detection, b: Detection) -> float:
    ix1, iy1, ix2, iy2 = max(a.x1, b.x1), max(a.y1, b.y1), min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = a.w * a.h + b.w * b.h - inter
    return inter / ua if ua > 0 else 0.0
