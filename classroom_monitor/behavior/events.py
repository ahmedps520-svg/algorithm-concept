from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Behavior(str, Enum):
    FIGHTING = "fighting"
    SLEEPING = "sleeping"
    OUT_OF_SEAT = "out_of_seat"
    TALKING = "talking"


class Confidence(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

    @classmethod
    def from_score(cls, score: float) -> "Confidence":
        if score >= 0.75:
            return cls.HIGH
        if score >= 0.5:
            return cls.MEDIUM
        return cls.LOW


@dataclass
class BehaviorEvent:
    room_id: str
    behavior: Behavior
    track_ids: list[int]           # one person, or two for fighting
    ts: float
    score: float                   # 0-1
    sustained_s: float             # how long the condition has held
    evidence: dict = field(default_factory=dict)

    @property
    def confidence(self) -> Confidence:
        return Confidence.from_score(self.score)

    @property
    def key(self) -> tuple[str, str, tuple[int, ...]]:
        return self.room_id, self.behavior.value, tuple(sorted(self.track_ids))
