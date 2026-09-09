"""Alert lifecycle.

    NEW ──acknowledge(staff)──► ACKNOWLEDGED ──review(staff, outcome)──► CONFIRMED | DISMISSED

* ``NEW``           raised by the pipeline; nobody has looked at it.
* ``ACKNOWLEDGED``  a named staff member has opened it and is responsible for it.
* ``CONFIRMED``     staff judged it a real concern -> the clip becomes ``ClipFlag.ACTIONABLE``.
* ``DISMISSED``     staff judged it a false positive / not a concern -> clip is purged sooner.

There is no path from NEW straight to CONFIRMED, and no code path that sets ACTIONABLE
without a human reviewer id. The system itself never acts on an alert.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum


class AlertStatus(str, Enum):
    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    CONFIRMED = "confirmed"
    DISMISSED = "dismissed"

    @property
    def is_open(self) -> bool:
        return self in (AlertStatus.NEW, AlertStatus.ACKNOWLEDGED)


class ReviewOutcome(str, Enum):
    CONFIRM = "confirm"
    DISMISS = "dismiss"


class ClipFlag(str, Enum):
    UNREVIEWED = "unreviewed"
    ACTIONABLE = "actionable"      # only reachable through a human CONFIRM
    DISMISSED = "dismissed"


@dataclass
class Alert:
    room_id: str
    behavior: str
    track_ids: list[int]
    score: float
    confidence: str
    sustained_s: float
    first_ts: float
    last_ts: float
    evidence: dict = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    status: AlertStatus = AlertStatus.NEW
    occurrences: int = 1
    acknowledged_by: str | None = None
    acknowledged_at: float | None = None
    reviewed_by: str | None = None
    reviewed_at: float | None = None
    review_notes: str = ""
    clip_id: str | None = None
    clip_flag: ClipFlag = ClipFlag.UNREVIEWED
    # AI second opinion from the clip verifier (advisory only; a human still decides).
    ai_verdict: str | None = None        # likely_real | likely_false_alarm | unclear | error
    ai_confidence: float | None = None
    ai_summary: str = ""                 # what the model says is happening
    ai_reason: str = ""
    ai_model: str | None = None
    ai_at: float | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["clip_flag"] = self.clip_flag.value
        return d
