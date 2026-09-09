"""Scripted synthetic scenes used by the demo and tests."""
from __future__ import annotations

from .source import Actor, SyntheticSource

SEATS = [(0.15, 0.48), (0.40, 0.48), (0.65, 0.48), (0.15, 0.80), (0.40, 0.80), (0.65, 0.80)]


def seated_actors(jitter: float = 0.002) -> list[Actor]:
    return [Actor(id=i + 1, x=x, y=y, jitter=jitter) for i, (x, y) in enumerate(SEATS)]


def default_classroom(room_id: str, fps: float = 10.0, realtime: bool = True,
                      duration_s: float | None = None) -> SyntheticSource:
    """Six seated students. Over ~2 minutes:
        15-45s   student 3 wanders out of seat and comes back      -> OUT_OF_SEAT alert
        30s+     student 4 puts head down and stays still          -> SLEEPING alert
        60-61.5s brief horseplay between 1 and 2 (short spike)     -> must NOT alert
        80-88s   sustained scuffle between 5 and 6                 -> FIGHTING alert
    """
    actors = seated_actors()
    fired: set[str] = set()

    def once(tag: str, t: float, when: float) -> bool:
        if tag not in fired and t >= when:
            fired.add(tag)
            return True
        return False

    def script(t: float, actors: list[Actor]) -> None:
        a1, a2, a3, a4, a5, a6 = actors
        if once("3-leave", t, 15):
            a3.move_to(0.88, 0.25)
        if once("3-back", t, 45):
            a3.move_to(*SEATS[2])
        if t >= 30:
            a4.head_drop = min(1.0, (t - 30) / 3.0)
            a4.jitter = 0.0005
        if once("horse-start", t, 60):
            a1.jitter = a2.jitter = 0.03
            a1.arms_up = a2.arms_up = True
            a1.move_to(0.25, 0.48); a2.move_to(0.31, 0.48)
        if once("horse-stop", t, 61.5):
            a1.jitter = a2.jitter = 0.002
            a1.arms_up = a2.arms_up = False
            a1.move_to(*SEATS[0]); a2.move_to(*SEATS[1])
        if once("fight-start", t, 80):
            a5.move_to(0.50, 0.80); a6.move_to(0.56, 0.80)
            a5.jitter = a6.jitter = 0.025
            a5.arms_up = a6.arms_up = True
        if once("fight-stop", t, 88):
            a5.move_to(*SEATS[4]); a6.move_to(*SEATS[5])
            a5.jitter = a6.jitter = 0.002
            a5.arms_up = a6.arms_up = False

    return SyntheticSource(room_id, actors, script, fps=fps, realtime=realtime, duration_s=duration_s)
