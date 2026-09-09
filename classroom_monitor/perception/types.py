from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# COCO-17 keypoint indices
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE = 11, 12, 13, 14, 15, 16


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2

    @property
    def foot(self) -> tuple[float, float]:
        """Bottom-centre; the best single point for "where is this person standing/sitting"."""
        return self.cx, self.y2


@dataclass
class Keypoints:
    """(17, 3) array of x, y, confidence in pixel coordinates."""

    xyc: np.ndarray

    def pt(self, idx: int, min_conf: float = 0.3) -> tuple[float, float] | None:
        x, y, c = self.xyc[idx]
        return (float(x), float(y)) if c >= min_conf else None


@dataclass
class TrackedPerson:
    """One person in one frame, with a session-stable track id.

    The id is an arbitrary integer assigned by the tracker. It carries no identity; a student
    who leaves the frame and comes back gets a fresh id."""

    track_id: int
    box: Detection
    keypoints: Keypoints | None = None
    # Frames since the track was first seen (tracker-provided, helps ignore newborn tracks).
    age: int = 0
    extra: dict = field(default_factory=dict)
