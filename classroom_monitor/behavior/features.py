"""Per-frame feature extraction for one tracked person, plus the sliding-window state that
classifiers read from. All geometry is normalised so thresholds transfer across cameras."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from ..perception.types import (
    L_EAR, L_EYE, L_SHOULDER, L_WRIST, NOSE, R_EAR, R_EYE, R_SHOULDER, R_WRIST, L_HIP, R_HIP, TrackedPerson,
)


@dataclass
class Observation:
    ts: float
    cx: float            # box centre, normalised to frame
    cy: float
    w: float             # box size, normalised to frame
    h: float
    foot_x: float
    foot_y: float
    motion: float        # centre displacement since last obs, in box-widths per second
    vx: float            # velocity, normalised frame units / s
    vy: float
    head_drop: float | None      # (nose_y - shoulder_y) / torso_h; >0 means head below shoulders
    arms_up: bool | None         # any wrist above its shoulder
    head_yaw: float | None       # ear asymmetry proxy for looking sideways (-1..1)
    pose_available: bool
    head_height: float | None = None   # (shoulder_y - nose_y) / shoulder_w; camera-distance invariant
    head_tilt: float | None = None     # ear line vs horizontal, degrees
    nose: tuple[float, float] | None = None   # normalised nose position
    shoulder_w: float | None = None    # normalised shoulder width
    head_motion: float = 0.0           # nose displacement, shoulder widths per second
    eyes_visible: bool = False         # both eye keypoints confidently detected
    mouth: float | None = None         # lip opening 0-1 from a face-landmark stream, if any
    rel_drop: float | None = None      # 1 - head_height / this track's upright baseline


class TrackState:
    """Sliding window of observations for one track id."""

    def __init__(self, track_id: int, window_s: float) -> None:
        self.track_id = track_id
        self.window_s = window_s
        self.obs: deque[Observation] = deque()
        self.last_seen: float = 0.0
        self.first_seen: float | None = None
        self.head_base: float | None = None    # upright head height, learned over the first frames
        self.head_base_n: int = 0

    def push(self, o: Observation) -> None:
        if self.first_seen is None:
            self.first_seen = o.ts
        prev = self.latest
        if prev is not None and o.nose is not None and prev.nose is not None:
            dt = o.ts - prev.ts
            if dt > 1e-3:
                unit = max(o.shoulder_w or o.w * 0.6, 1e-3)
                o.head_motion = math.hypot(o.nose[0] - prev.nose[0], o.nose[1] - prev.nose[1]) / unit / dt
        # Upright baseline = the tallest head position seen lately: rises fast, decays very
        # slowly, so a baseline captured in an odd posture corrects itself within seconds.
        if o.head_height is not None:
            if self.head_base is None:
                self.head_base = o.head_height
            elif o.head_height > self.head_base:
                self.head_base += 0.3 * (o.head_height - self.head_base)
            else:
                self.head_base *= 0.9995
            self.head_base_n += 1
        if o.head_height is not None and self.head_base is not None and self.head_base_n >= 20:
            o.rel_drop = 1.0 - o.head_height / max(self.head_base, 1e-3)
        self.obs.append(o)
        self.last_seen = o.ts
        while self.obs and o.ts - self.obs[0].ts > self.window_s:
            self.obs.popleft()

    @property
    def latest(self) -> Observation | None:
        return self.obs[-1] if self.obs else None

    def recent(self, seconds: float) -> list[Observation]:
        if not self.obs:
            return []
        cutoff = self.obs[-1].ts - seconds
        return [o for o in self.obs if o.ts >= cutoff]


def observe(person: TrackedPerson, ts: float, frame_w: int, frame_h: int,
            prev: Observation | None) -> Observation:
    b = person.box
    cx, cy = b.cx / frame_w, b.cy / frame_h
    w, h = max(b.w, 1.0) / frame_w, max(b.h, 1.0) / frame_h
    fx, fy = b.foot[0] / frame_w, b.foot[1] / frame_h

    vx = vy = motion = 0.0
    if prev is not None:
        dt = ts - prev.ts
        if dt > 1e-3:
            vx, vy = (cx - prev.cx) / dt, (cy - prev.cy) / dt
            # Displacement expressed in box widths per second: camera-distance invariant.
            motion = math.hypot(cx - prev.cx, cy - prev.cy) / max(w, 1e-3) / dt

    head_drop = arms_up = head_yaw = None
    head_height = head_tilt = nose_n = shoulder_w = None
    pose_ok = eyes_visible = False
    kp = person.keypoints
    if kp is not None:
        eyes_visible = kp.pt(L_EYE, 0.4) is not None and kp.pt(R_EYE, 0.4) is not None
        nose = kp.pt(NOSE)
        ls, rs = kp.pt(L_SHOULDER), kp.pt(R_SHOULDER)
        lh, rh = kp.pt(L_HIP), kp.pt(R_HIP)
        if nose and ls and rs:
            shoulder_y = (ls[1] + rs[1]) / 2
            if lh and rh:
                torso = max(abs((lh[1] + rh[1]) / 2 - shoulder_y), 1.0)
            else:
                torso = max(b.h * 0.35, 1.0)
            head_drop = (nose[1] - shoulder_y) / torso
            sw = max(abs(rs[0] - ls[0]), 1.0)
            head_height = (shoulder_y - nose[1]) / sw
            shoulder_w = sw / frame_w
            nose_n = (nose[0] / frame_w, nose[1] / frame_h)
            pose_ok = True
        lw, rw = kp.pt(L_WRIST), kp.pt(R_WRIST)
        if ls and rs:
            ups = [w_[1] < s_[1] for w_, s_ in ((lw, ls), (rw, rs)) if w_ is not None]
            arms_up = any(ups) if ups else None
        le, re = kp.pt(L_EAR), kp.pt(R_EAR)
        if nose and le and re:
            span = max(abs(re[0] - le[0]), 1.0)
            head_yaw = ((nose[0] - le[0]) - (re[0] - nose[0])) / span
            head_tilt = math.degrees(math.atan2(re[1] - le[1], re[0] - le[0]))

    return Observation(ts, cx, cy, w, h, fx, fy, motion, vx, vy, head_drop, arms_up, head_yaw, pose_ok,
                       head_height=head_height, head_tilt=head_tilt, nose=nose_n, shoulder_w=shoulder_w,
                       eyes_visible=eyes_visible)
