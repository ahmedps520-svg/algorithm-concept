"""Learned behavior classifier.

Window features (mean/std/min/max of 12 per-frame cues over the last 2 s) -> softmax regression.
The feature order and model JSON format are shared with the browser demo (docs/index.html), so
examples recorded there and exported can train this model with ``scripts/train_behavior.py``.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import LearnedThresholds, SeatZone
from .events import Behavior, BehaviorEvent
from .features import Observation, TrackState
from .sustain import SustainedCondition

LABELS = ["normal", "sleeping", "out_of_seat", "talking", "fighting"]
FEAT_NAMES = ["headDrop", "relDrop", "tilt", "yaw", "motion", "headMotion", "armsUp", "mouth", "inZone", "gap", "partnerMotion", "hasPose"]
WINDOW_S = 2.0
MIN_OBS = 8


def frame_features(o: Observation, in_zone: bool, gap: float, partner_motion: float, mouth: float = 0.0) -> list[float]:
    return [
        o.head_drop or 0.0, o.rel_drop or 0.0, (o.head_tilt or 0.0) / 45.0, o.head_yaw or 0.0,
        min(o.motion, 5.0), min(o.head_motion, 5.0), 1.0 if o.arms_up else 0.0, mouth,
        1.0 if in_zone else 0.0, min(gap, 3.0) / 3.0, min(partner_motion, 5.0), 1.0 if o.head_drop is not None else 0.0,
    ]


def window_features(frames: list[list[float]]) -> list[float] | None:
    if len(frames) < MIN_OBS:
        return None
    F = np.asarray(frames, dtype=float)
    return np.stack([F.mean(0), F.std(0), F.min(0), F.max(0)], axis=1).reshape(-1).tolist()


@dataclass
class SoftmaxModel:
    labels: list[str]
    mu: np.ndarray
    sd: np.ndarray
    W: np.ndarray            # (K, D+1), column 0 = bias

    @classmethod
    def from_json(cls, d: dict) -> "SoftmaxModel":
        return cls(list(d["labels"]), np.asarray(d["mu"], float), np.asarray(d["sd"], float), np.asarray(d["W"], float))

    def to_json(self) -> dict:
        return {"labels": self.labels, "mu": self.mu.tolist(), "sd": self.sd.tolist(), "W": self.W.tolist()}

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        # Clip standardised inputs: a value far outside the training range should count as
        # "very high", not overflow the logits.
        z = np.clip((x - self.mu) / self.sd, -10.0, 10.0)
        logits = self.W[:, 0] + z @ self.W[:, 1:].T
        logits -= logits.max(axis=-1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=-1, keepdims=True)

    def predict(self, x: list[float]) -> tuple[str, float]:
        p = self.predict_proba(np.asarray(x, float))
        k = int(p.argmax())
        return self.labels[k], float(p[k])


def train(X: np.ndarray, y: list[str], labels: list[str] = LABELS, epochs: int = 400, lr: float = 0.1, l2: float = 1e-3) -> SoftmaxModel:
    """Class-weighted multinomial logistic regression by full-batch gradient descent."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-6] = 1.0          # effectively constant feature: leave unscaled rather than blow up
    Z = (X - mu) / sd
    Y = np.asarray([labels.index(v) for v in y])
    K, D, N = len(labels), X.shape[1], X.shape[0]
    counts = np.bincount(Y, minlength=K)
    cw = np.where(counts > 0, N / (K * np.maximum(counts, 1)), 0.0)
    W = np.zeros((K, D + 1))
    Zb = np.hstack([np.ones((N, 1)), Z])
    onehot = np.eye(K)[Y]
    for _ in range(epochs):
        logits = Zb @ W.T
        logits -= logits.max(1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(1, keepdims=True)
        G = ((P - onehot) * cw[Y][:, None]).T @ Zb / N
        G[:, 1:] += l2 * W[:, 1:]
        W -= lr * G
    return SoftmaxModel(list(labels), mu, sd, W)


def load_model(path: Path | str) -> SoftmaxModel:
    d = json.loads(Path(path).read_text())
    return SoftmaxModel.from_json(d["model"] if "model" in d and d["model"] else d)


class LearnedClassifier:
    """Runs a trained SoftmaxModel per track behind one sustained gate per behavior."""

    behavior = Behavior.SLEEPING   # nominal; emits any behavior in the label set

    def __init__(self, t: LearnedThresholds, model: SoftmaxModel, zones: list[SeatZone], fps: float) -> None:
        self.t = t
        self.model = model
        self.zones = zones
        self.gates = {b: SustainedCondition(t.window_s, t.min_duration_s.get(b, 5.0), t.min_active_fraction) for b in model.labels if b != "normal"}
        self._frames: list[tuple[float, list[float]]] = []

    def _in_zone(self, o: Observation) -> bool:
        return any(z.contains(o.foot_x, o.foot_y, 0.03) for z in self.zones)

    def evaluate(self, me: TrackState, others: list[TrackState], ts: float) -> BehaviorEvent | None:
        cur = me.latest
        if cur is None:
            return None
        gap, partner_motion, partner = 3.0, 0.0, None
        for o in others:
            p = o.latest
            if p is None:
                continue
            dx = max(0.0, abs(cur.cx - p.cx) - (cur.w + p.w) / 2)
            dy = max(0.0, abs(cur.cy - p.cy) - (cur.h + p.h) / 2)
            g = math.hypot(dx, dy) / max(min(cur.w, p.w), 1e-3)
            if g < gap:
                gap, partner_motion, partner = g, p.motion, o
        self._frames.append((ts, frame_features(cur, self._in_zone(cur), gap, partner_motion)))
        self._frames = [f for f in self._frames if ts - f[0] <= WINDOW_S]
        x = window_features([f for _, f in self._frames])
        if x is None:
            return None
        label, prob = self.model.predict(x)
        out = None
        for b, gate in self.gates.items():
            res = gate.update(ts, label == b and prob >= self.t.min_prob)
            if res.triggered and out is None:
                ids = [me.track_id] + ([partner.track_id] if b == "fighting" and partner is not None else [])
                out = BehaviorEvent(room_id="", behavior=Behavior(b), track_ids=ids, ts=ts, score=min(1.0, prob),
                                    sustained_s=res.sustained_s,
                                    evidence={"cue": "learned model", "prob": round(prob, 2), "active_fraction": round(res.active_fraction, 2)})
        return out
