"""Perception backends: frame -> list[TrackedPerson].

``UltralyticsBackend``  YOLOv8 detection + built-in ByteTrack + YOLOv8-pose (real deployments).
``StubBackend``         Uses the ground-truth boxes a SyntheticSource attaches to each frame,
                        passed through the same IoU tracker so track-continuity logic is exercised.
"""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

from ..config import ModelConfig
from ..ingestion.source import Frame
from .tracker import IoUTracker
from .types import Detection, Keypoints, TrackedPerson

log = logging.getLogger(__name__)


class PerceptionBackend(Protocol):
    def process(self, frame: Frame) -> list[TrackedPerson]: ...


class StubBackend:
    """No ML. Detections come from the synthetic frame; tracking is done by ``IoUTracker`` so
    id continuity is still a real (if easy) problem being solved."""

    def __init__(self) -> None:
        self.tracker = IoUTracker()

    def process(self, frame: Frame) -> list[TrackedPerson]:
        dets = [Detection(x1, y1, x2, y2, 0.95) for (_, x1, y1, x2, y2) in frame.synthetic_boxes]
        tracks = self.tracker.update(dets)
        # Attach keypoints by matching each tracked box to the synthetic actor it overlaps most.
        for tp in tracks:
            best, best_iou = None, 0.0
            for (aid, x1, y1, x2, y2) in frame.synthetic_boxes:
                iou = _iou((tp.box.x1, tp.box.y1, tp.box.x2, tp.box.y2), (x1, y1, x2, y2))
                if iou > best_iou:
                    best, best_iou = aid, iou
            if best is not None and best in frame.synthetic_keypoints:
                tp.keypoints = Keypoints(frame.synthetic_keypoints[best])
        return tracks


class UltralyticsBackend:
    """YOLOv8-pose already returns person boxes *and* keypoints in one pass, and ultralytics'
    ``track()`` wires in ByteTrack, so a single model call yields tracked, posed people."""

    def __init__(self, cfg: ModelConfig) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("pip install .[vision] for the ultralytics backend") from e
        self.cfg = cfg
        self.model = YOLO(cfg.pose_weights)
        self._ages: dict[int, int] = {}

    def process(self, frame: Frame) -> list[TrackedPerson]:
        res = self.model.track(
            frame.image, persist=True, conf=self.cfg.detection_conf, classes=[0],
            tracker=self.cfg.tracker, device=self.cfg.device, verbose=False,
        )[0]
        out: list[TrackedPerson] = []
        if res.boxes is None or res.boxes.id is None:
            return out
        ids = res.boxes.id.int().tolist()
        xyxy = res.boxes.xyxy.tolist()
        confs = res.boxes.conf.tolist()
        kps = res.keypoints.data.cpu().numpy() if res.keypoints is not None else None
        seen = set()
        for i, tid in enumerate(ids):
            seen.add(tid)
            self._ages[tid] = self._ages.get(tid, 0) + 1
            x1, y1, x2, y2 = xyxy[i]
            kp = Keypoints(np.asarray(kps[i], dtype=np.float32)) if kps is not None else None
            out.append(TrackedPerson(tid, Detection(x1, y1, x2, y2, confs[i]), kp, age=self._ages[tid]))
        for tid in [t for t in self._ages if t not in seen]:
            self._ages.pop(tid, None)
        return out


def build_backend(cfg: ModelConfig) -> PerceptionBackend:
    if cfg.backend == "ultralytics":
        return UltralyticsBackend(cfg)
    return StubBackend()


def _iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0
