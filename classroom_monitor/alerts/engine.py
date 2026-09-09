"""Alert engine: turns a stream of BehaviorEvents into de-duplicated alerts and enforces the
human review workflow.

Debounce rules (all configurable in ``DebounceConfig``):
  1. Same room+behavior+track set with an alert still open  -> merge (bump occurrences/last_ts).
  2. Same room+behavior+track set alerted within per_track_cooldown_s -> suppress.
  3. Same room+behavior (any track) alerted within per_room_cooldown_s -> suppress.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from ..behavior.events import BehaviorEvent
from ..config import DebounceConfig
from .models import Alert, AlertStatus, ClipFlag, ReviewOutcome
from .store import AlertStore

log = logging.getLogger(__name__)

AlertCallback = Callable[[Alert], None]


class ReviewError(Exception):
    """Raised when the human-review workflow is violated (e.g. confirm before acknowledge)."""


class AlertEngine:
    def __init__(self, store: AlertStore, debounce: DebounceConfig, clock: Callable[[], float] = time.time) -> None:
        self.store = store
        self.debounce = debounce
        self.clock = clock
        self._lock = threading.RLock()
        self._last_by_key: dict[tuple, float] = {}          # (room, behavior, tracks) -> last alert ts
        self._last_by_room_behavior: dict[tuple, float] = {}
        self._open_by_key: dict[tuple, str] = {}            # -> alert id while open
        self._on_new: list[AlertCallback] = []
        self._on_update: list[AlertCallback] = []

    # ---------------------------------------------------------------- wiring
    def on_new_alert(self, cb: AlertCallback) -> None:
        self._on_new.append(cb)

    def on_alert_updated(self, cb: AlertCallback) -> None:
        self._on_update.append(cb)

    def notify_updated(self, alert: Alert) -> None:
        """Let subscribers (dashboard) know an alert changed outside the review flow."""
        self._emit(self._on_update, alert)

    def _emit(self, cbs: list[AlertCallback], alert: Alert) -> None:
        for cb in cbs:
            try:
                cb(alert)
            except Exception:  # noqa: BLE001
                log.exception("alert callback failed")

    # ------------------------------------------------------------- ingestion
    def handle(self, ev: BehaviorEvent) -> Alert | None:
        """Returns the newly created Alert, or None if merged/suppressed."""
        key = ev.key
        rb_key = (ev.room_id, ev.behavior.value)
        with self._lock:
            open_id = self._open_by_key.get(key)
            if open_id and self.debounce.merge_into_open_alert:
                a = self.store.get(open_id)
                if a and a.status.is_open:
                    a.occurrences += 1
                    a.last_ts = ev.ts
                    a.sustained_s = max(a.sustained_s, ev.sustained_s)
                    a.score = max(a.score, ev.score)
                    a.confidence = ev.confidence.value if ev.score >= a.score else a.confidence
                    self.store.upsert(a)
                    self._emit(self._on_update, a)
                    return None
                self._open_by_key.pop(key, None)

            last = self._last_by_key.get(key)
            if last is not None and ev.ts - last < self.debounce.per_track_cooldown_s:
                return None
            last_rb = self._last_by_room_behavior.get(rb_key)
            if last_rb is not None and ev.ts - last_rb < self.debounce.per_room_cooldown_s:
                return None

            a = Alert(
                room_id=ev.room_id, behavior=ev.behavior.value, track_ids=sorted(ev.track_ids),
                score=ev.score, confidence=ev.confidence.value, sustained_s=ev.sustained_s,
                first_ts=ev.ts, last_ts=ev.ts, evidence=dict(ev.evidence),
            )
            self.store.upsert(a)
            self._last_by_key[key] = ev.ts
            self._last_by_room_behavior[rb_key] = ev.ts
            self._open_by_key[key] = a.id
        log.info("ALERT %s %s %s tracks=%s conf=%s", a.id, a.room_id, a.behavior, a.track_ids, a.confidence)
        self._emit(self._on_new, a)
        return a

    # ---------------------------------------------------------- human review
    def acknowledge(self, alert_id: str, staff_id: str) -> Alert:
        if not staff_id or not staff_id.strip():
            raise ReviewError("a staff id is required to acknowledge an alert")
        with self._lock:
            a = self._require(alert_id)
            if a.status is not AlertStatus.NEW:
                raise ReviewError(f"alert {alert_id} is {a.status.value}; only NEW alerts can be acknowledged")
            a.status = AlertStatus.ACKNOWLEDGED
            a.acknowledged_by = staff_id.strip()
            a.acknowledged_at = self.clock()
            self.store.upsert(a)
        self._emit(self._on_update, a)
        return a

    def review(self, alert_id: str, staff_id: str, outcome: ReviewOutcome, notes: str = "") -> Alert:
        """The *only* way a clip becomes ACTIONABLE. Requires a prior acknowledgment."""
        if not staff_id or not staff_id.strip():
            raise ReviewError("a staff id is required to review an alert")
        with self._lock:
            a = self._require(alert_id)
            if a.status is AlertStatus.NEW:
                raise ReviewError(f"alert {alert_id} must be acknowledged before it can be reviewed")
            if not a.status.is_open:
                raise ReviewError(f"alert {alert_id} is already {a.status.value}")
            a.reviewed_by = staff_id.strip()
            a.reviewed_at = self.clock()
            a.review_notes = notes or ""
            if outcome is ReviewOutcome.CONFIRM:
                a.status = AlertStatus.CONFIRMED
                a.clip_flag = ClipFlag.ACTIONABLE
            else:
                a.status = AlertStatus.DISMISSED
                a.clip_flag = ClipFlag.DISMISSED
            self.store.upsert(a)
            self._open_by_key.pop((a.room_id, a.behavior, tuple(a.track_ids)), None)
        self._emit(self._on_update, a)
        return a

    def _require(self, alert_id: str) -> Alert:
        a = self.store.get(alert_id)
        if a is None:
            raise ReviewError(f"unknown alert {alert_id}")
        return a

    # ------------------------------------------------------------- queries
    def queue(self) -> list[Alert]:
        """Alerts awaiting a human (NEW first, then ACKNOWLEDGED), oldest first."""
        new = self.store.list(status=AlertStatus.NEW, limit=500)
        ack = self.store.list(status=AlertStatus.ACKNOWLEDGED, limit=500)
        return sorted(new, key=lambda a: a.first_ts) + sorted(ack, key=lambda a: a.first_ts)
