"""Rule-based per-track behavior classifiers.

Each classifier is instantiated per (room, track) and called once per frame with the track's
own state plus the other tracks in the room. It returns a ``BehaviorEvent`` only while the
sustained gate is open; the alert engine decides whether that becomes a new alert.
"""
from __future__ import annotations

import math
from pathlib import Path
from collections import deque
from typing import Protocol

from ..config import (
    BehaviorThresholds, FightingThresholds, OutOfSeatThresholds, SeatZone, SleepingThresholds,
    TalkingThresholds,
)
from .events import Behavior, BehaviorEvent
from .features import Observation, TrackState
from .sustain import SustainedCondition


class Classifier(Protocol):
    behavior: Behavior

    def evaluate(self, me: TrackState, others: list[TrackState], ts: float) -> BehaviorEvent | None: ...


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------
def _gap_in_box_widths(a: Observation, b: Observation) -> float:
    """Horizontal/vertical gap between two boxes, in units of the smaller box width.
    0 means overlapping."""
    dx = max(0.0, abs(a.cx - b.cx) - (a.w + b.w) / 2)
    dy = max(0.0, abs(a.cy - b.cy) - (a.h + b.h) / 2)
    return math.hypot(dx, dy) / max(min(a.w, b.w), 1e-3)


def _nearest(me: Observation, others: list[TrackState]) -> tuple[TrackState | None, float]:
    best, best_gap = None, math.inf
    for o in others:
        lo = o.latest
        if lo is None:
            continue
        g = _gap_in_box_widths(me, lo)
        if g < best_gap:
            best, best_gap = o, g
    return best, best_gap


# --------------------------------------------------------------------------------------
# Fighting / physical aggression
# --------------------------------------------------------------------------------------
class FightingClassifier:
    """Per-frame aggression score from four cues, then a strict sustain gate.

    cues (each 0/1): motion spike on me, another person within reach, high relative motion
    between us, arm raised. Weighted so that motion alone or proximity alone never crosses
    the per-frame threshold: you need someone close *and* violent movement.
    """

    behavior = Behavior.FIGHTING
    # Per-frame threshold is 0.6: motion + proximity alone (0.5) does NOT cross it, so two
    # students walking side by side never count; you need relative motion between them.
    WEIGHTS = {"motion": 0.3, "proximity": 0.2, "relative": 0.3, "arms": 0.2}

    def __init__(self, t: FightingThresholds, fps: float) -> None:
        self.t = t
        self.gate = SustainedCondition(t.window_s, t.min_duration_s, t.min_active_fraction)
        self._arm_frames = 0
        self._partner: int | None = None
        self._scores: list[float] = []
        self._max_hist = max(1, int(t.window_s * fps))

    def evaluate(self, me: TrackState, others: list[TrackState], ts: float) -> BehaviorEvent | None:
        if not self.t.enabled or me.latest is None:
            return None
        cur = me.latest
        nearest, gap = _nearest(cur, others)
        # Among people within reach, engage with the one moving most: in a scuffle both parties
        # move, so a still bystander sitting next to the action is not picked as the partner.
        candidates = [o for o in others if o.latest is not None
                      and _gap_in_box_widths(cur, o.latest) <= self.t.proximity_box_widths]
        partner = max(candidates, key=lambda o: o.latest.motion, default=None)
        near = partner is not None

        motion = cur.motion >= self.t.motion_spike
        relative = False
        if near and partner.latest is not None:
            p = partner.latest
            # Velocity *difference*, so two people moving together (walking, carrying a table)
            # score ~0 while two people moving against each other score high. The partner must
            # also be moving: walking past a seated student is not an engagement.
            rel_speed = math.hypot(cur.vx - p.vx, cur.vy - p.vy) / max(cur.w, 1e-3)
            relative = rel_speed >= self.t.relative_motion and p.motion >= self.t.partner_motion

        self._arm_frames = self._arm_frames + 1 if cur.arms_up else 0
        arms = self._arm_frames >= self.t.arm_raise_min_frames

        score = (
            self.WEIGHTS["motion"] * motion
            + self.WEIGHTS["proximity"] * near
            + self.WEIGHTS["relative"] * relative
            + self.WEIGHTS["arms"] * arms
        )
        self._scores.append(score)
        if len(self._scores) > self._max_hist:
            self._scores.pop(0)

        active = score >= self.t.frame_score_threshold
        if active and partner is not None:
            self._partner = partner.track_id
        res = self.gate.update(ts, active)
        if not res.triggered:
            return None

        mean_score = sum(self._scores) / len(self._scores)
        # Confidence grows with how long it has lasted beyond the minimum and how dense the evidence is.
        duration_bonus = min(0.2, 0.05 * (res.sustained_s - self.t.min_duration_s))
        conf = max(self.t.confidence_floor, min(1.0, 0.5 * mean_score + 0.5 * res.active_fraction + duration_bonus))
        ids = [me.track_id] + ([self._partner] if self._partner is not None else [])
        return BehaviorEvent(
            room_id="", behavior=self.behavior, track_ids=ids, ts=ts, score=conf,
            sustained_s=res.sustained_s,
            evidence={
                "active_fraction": round(res.active_fraction, 2), "mean_frame_score": round(mean_score, 2),
                "motion": round(cur.motion, 2), "gap_box_widths": round(gap, 2) if nearest else None,
                "arm_raise_frames": self._arm_frames,
            },
        )


# --------------------------------------------------------------------------------------
# Sleeping / head down
# --------------------------------------------------------------------------------------
class SleepingClassifier:
    behavior = Behavior.SLEEPING

    def __init__(self, t: SleepingThresholds, fps: float) -> None:
        self.t = t
        self.gate = SustainedCondition(t.window_s, t.min_duration_s, t.min_active_fraction)

    def evaluate(self, me: TrackState, others: list[TrackState], ts: float) -> BehaviorEvent | None:
        if not self.t.enabled or me.latest is None:
            return None
        cur = me.latest
        if cur.head_drop is None:
            # No pose this frame: don't count for or against; keep the gate's history honest.
            self.gate.update(ts, False)
            return None
        recent = me.recent(1.0)
        mean_motion = sum(o.motion for o in recent) / max(len(recent), 1)
        mean_head = sum(o.head_motion for o in recent) / max(len(recent), 1)
        still = mean_motion <= self.t.max_motion and mean_head <= self.t.max_head_motion
        down_abs = cur.head_drop >= self.t.head_below_shoulders_ratio
        down_rel = cur.rel_drop is not None and cur.rel_drop >= self.t.relative_drop
        tilted = cur.head_tilt is not None and abs(cur.head_tilt) >= self.t.tilt_deg
        res = self.gate.update(ts, (down_abs or down_rel or tilted) and still)
        if not res.triggered:
            return None
        if down_abs:
            cue, depth = "nose below shoulders", min(1.0, cur.head_drop / max(self.t.head_below_shoulders_ratio * 3, 1e-3))
        elif down_rel:
            cue, depth = "head dropped vs own baseline", min(1.0, cur.rel_drop / 0.7)
        else:
            cue, depth = "head tilted", 0.6
        conf = min(1.0, 0.4 + 0.3 * res.active_fraction + 0.3 * depth)
        return BehaviorEvent(
            room_id="", behavior=self.behavior, track_ids=[me.track_id], ts=ts, score=conf,
            sustained_s=res.sustained_s,
            evidence={"cue": cue, "head_drop": round(cur.head_drop, 2),
                      "rel_drop": None if cur.rel_drop is None else round(cur.rel_drop, 2),
                      "tilt_deg": None if cur.head_tilt is None else round(cur.head_tilt),
                      "motion": round(mean_motion, 3), "active_fraction": round(res.active_fraction, 2)},
        )


# --------------------------------------------------------------------------------------
# Out of seat
# --------------------------------------------------------------------------------------
class OutOfSeatClassifier:
    behavior = Behavior.OUT_OF_SEAT

    def __init__(self, t: OutOfSeatThresholds, zones: list[SeatZone], fps: float) -> None:
        self.t = t
        self.zones = zones
        self.gate = SustainedCondition(t.window_s, t.min_duration_s, t.min_active_fraction)
        self.home_zone: str | None = None

    def _zone_at(self, x: float, y: float) -> SeatZone | None:
        for z in self.zones:
            if z.contains(x, y, self.t.zone_margin):
                return z
        return None

    def evaluate(self, me: TrackState, others: list[TrackState], ts: float) -> BehaviorEvent | None:
        if not self.t.enabled or not self.zones or me.latest is None:
            return None
        cur = me.latest
        z = self._zone_at(cur.foot_x, cur.foot_y)
        if z is not None and self.home_zone is None:
            self.home_zone = z.id           # first zone we settle in becomes "assigned"
        if self.home_zone is None and self.t.require_home_zone:
            return None                     # never seen seated: teacher / visitor / just arrived
        active = z is None
        res = self.gate.update(ts, active)
        if not res.triggered:
            return None
        conf = min(1.0, 0.5 + 0.5 * res.active_fraction)
        return BehaviorEvent(
            room_id="", behavior=self.behavior, track_ids=[me.track_id], ts=ts, score=conf,
            sustained_s=res.sustained_s,
            evidence={"home_zone": self.home_zone, "foot": (round(cur.foot_x, 2), round(cur.foot_y, 2)),
                      "active_fraction": round(res.active_fraction, 2)},
        )


# --------------------------------------------------------------------------------------
# Talking (LOW CONFIDENCE from video)
# --------------------------------------------------------------------------------------
class TalkingClassifier:
    """Video-only proxy: repeated head turns toward a neighbour. At classroom camera distance
    lip motion is not resolvable, so this is capped at LOW confidence and is off by default.
    Pair with a per-room audio level channel before trusting it."""

    behavior = Behavior.TALKING

    def __init__(self, t: TalkingThresholds, fps: float) -> None:
        self.t = t
        self.gate = SustainedCondition(t.window_s, t.min_duration_s, t.min_active_fraction)
        self._yaws: deque[float] = deque(maxlen=max(2, int(2.0 * fps)))

    def evaluate(self, me: TrackState, others: list[TrackState], ts: float) -> BehaviorEvent | None:
        if not self.t.enabled or me.latest is None:
            return None
        cur = me.latest
        partner, gap = _nearest(cur, others)
        near = partner is not None and gap <= self.t.proximity_box_widths
        if self.t.require_neighbour and not near:
            self.gate.update(ts, False)
            return None

        turned = False
        if cur.head_yaw is not None:
            if self._yaws:
                # Compare against the recent baseline as well as the previous frame, so a slow
                # turn away from "facing forward" counts, not only a fast one.
                baseline = sum(self._yaws) / len(self._yaws)
                turned = (abs(cur.head_yaw - self._yaws[-1]) >= self.t.head_turn_delta
                          or abs(cur.head_yaw - baseline) >= self.t.head_turn_delta)
            self._yaws.append(cur.head_yaw)

        res = self.gate.update(ts, turned)
        if not res.triggered:
            return None
        conf = min(self.t.max_confidence, 0.2 + 0.2 * res.active_fraction)   # hard cap
        return BehaviorEvent(
            room_id="", behavior=self.behavior, track_ids=[me.track_id], ts=ts, score=conf,
            sustained_s=res.sustained_s,
            evidence={"note": "video-only cue (head turning); low confidence by design, add audio to confirm",
                      "neighbour": partner.track_id if near else None,
                      "active_fraction": round(res.active_fraction, 2)},
        )


def build_classifiers(t: BehaviorThresholds, zones: list[SeatZone], fps: float) -> list[Classifier]:
    out: list[Classifier] = [
        FightingClassifier(t.fighting, fps),
        SleepingClassifier(t.sleeping, fps),
        OutOfSeatClassifier(t.out_of_seat, zones, fps),
        TalkingClassifier(t.talking, fps),
    ]
    if t.learned.enabled and Path(t.learned.weights_path).exists():
        from .learned import LearnedClassifier, load_model
        out.append(LearnedClassifier(t.learned, load_model(t.learned.weights_path), zones, fps))
    return out
