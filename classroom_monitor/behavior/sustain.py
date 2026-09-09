"""Sustained-condition gate.

Every classifier feeds a per-frame boolean into one of these. It only reports ``triggered``
when the condition has held for ``min_duration_s`` *and* at least ``min_active_fraction`` of
frames since the run began (bounded to the trailing ``window_s``) were active. A single active
frame can never trigger, and a run that goes quiet is abandoned.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass
class SustainResult:
    triggered: bool
    sustained_s: float
    active_fraction: float


class SustainedCondition:
    def __init__(self, window_s: float, min_duration_s: float, min_active_fraction: float) -> None:
        self.window_s = window_s
        self.min_duration_s = min_duration_s
        self.min_active_fraction = min_active_fraction
        self._hist: deque[tuple[float, bool]] = deque()
        self._run_start: float | None = None

    def update(self, ts: float, active: bool) -> SustainResult:
        self._hist.append((ts, active))
        while self._hist and ts - self._hist[0][0] > self.window_s:
            self._hist.popleft()

        if active and self._run_start is None:
            self._run_start = ts
        if self._run_start is None:
            return SustainResult(False, 0.0, 0.0)

        # Fraction is measured over the current run only (never over frames before it began),
        # so a long quiet history doesn't delay a genuine run, and bounded by the window so a
        # long run is judged on its recent density.
        in_run = [a for t, a in self._hist if t >= self._run_start]
        fraction = sum(in_run) / len(in_run)
        if fraction < self.min_active_fraction:
            self._run_start = ts if active else None
            return SustainResult(False, 0.0, fraction)

        sustained = ts - self._run_start
        triggered = sustained >= self.min_duration_s and len(in_run) >= 2
        return SustainResult(triggered, sustained, fraction)

    def reset(self) -> None:
        self._hist.clear()
        self._run_start = None
