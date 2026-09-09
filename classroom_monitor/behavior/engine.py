"""Per-room behavior engine: maintains a TrackState + classifier set per track id and turns
tracked people into BehaviorEvents."""
from __future__ import annotations

from ..config import BehaviorThresholds, SeatZone
from ..perception.types import TrackedPerson
from .classifiers import Classifier, build_classifiers
from .events import BehaviorEvent
from .features import TrackState, observe


class BehaviorEngine:
    def __init__(self, room_id: str, thresholds: BehaviorThresholds, seat_zones: list[SeatZone],
                 fps: float, track_ttl_s: float = 5.0, min_track_age: int = 3) -> None:
        self.room_id = room_id
        self.thresholds = thresholds
        self.seat_zones = seat_zones
        self.fps = fps
        self.track_ttl_s = track_ttl_s
        self.min_track_age = min_track_age
        self.window_s = max(
            thresholds.fighting.window_s, thresholds.sleeping.window_s,
            thresholds.out_of_seat.window_s, thresholds.talking.window_s,
        )
        self._states: dict[int, TrackState] = {}
        self._classifiers: dict[int, list[Classifier]] = {}

    def update(self, people: list[TrackedPerson], ts: float, frame_w: int, frame_h: int) -> list[BehaviorEvent]:
        for p in people:
            st = self._states.get(p.track_id)
            if st is None:
                st = TrackState(p.track_id, self.window_s)
                self._states[p.track_id] = st
                self._classifiers[p.track_id] = build_classifiers(self.thresholds, self.seat_zones, self.fps)
            st.push(observe(p, ts, frame_w, frame_h, st.latest))

        # Evict tracks not seen recently so their windows don't go stale.
        for tid in [t for t, s in self._states.items() if ts - s.last_seen > self.track_ttl_s]:
            self._states.pop(tid, None)
            self._classifiers.pop(tid, None)

        events: list[BehaviorEvent] = []
        present = {p.track_id: p for p in people}
        for tid, p in present.items():
            if p.age < self.min_track_age:
                continue                     # newborn tracks are often detector flicker
            me = self._states[tid]
            others = [s for t, s in self._states.items() if t != tid and t in present]
            for clf in self._classifiers[tid]:
                ev = clf.evaluate(me, others, ts)
                if ev is not None:
                    ev.room_id = self.room_id
                    events.append(ev)
        return events

    @property
    def active_track_ids(self) -> list[int]:
        return list(self._states)
