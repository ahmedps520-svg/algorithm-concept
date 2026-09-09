"""Drive the BehaviorEngine with the synthetic source (stub perception) and assert which
behaviors fire, when, and for whom."""
from classroom_monitor.behavior import Behavior, BehaviorEngine
from classroom_monitor.config import BehaviorThresholds, SeatZone, TalkingThresholds
from classroom_monitor.ingestion.scenarios import SEATS, seated_actors
from classroom_monitor.ingestion.source import Actor, SyntheticSource
from classroom_monitor.perception.backend import StubBackend

ZONES = [SeatZone(id=f"S{i}", x1=x - 0.12, y1=y - 0.16, x2=x + 0.12, y2=y + 0.16) for i, (x, y) in enumerate(SEATS)]


def run(actors, script, duration, thresholds=None, zones=ZONES, fps=10):
    src = SyntheticSource("r", actors, script, fps=fps, realtime=False, duration_s=duration)
    eng = BehaviorEngine("r", thresholds or BehaviorThresholds(), zones, fps)
    backend = StubBackend()
    events = []
    for f in src.frames():
        w, h = f.size
        for ev in eng.update(backend.process(f), f.ts, w, h):
            events.append((f.ts, ev))
    return events


def first(events, behavior):
    for ts, ev in events:
        if ev.behavior is behavior:
            return ts, ev
    return None


def test_quiet_classroom_produces_no_events():
    assert run(seated_actors(), lambda t, a: None, 60) == []


def test_brief_horseplay_does_not_fire_fighting():
    def script(t, a):
        if 5 <= t < 6.5:
            a[0].jitter = a[1].jitter = 0.03; a[0].arms_up = a[1].arms_up = True
            a[0].move_to(0.25, 0.48); a[1].move_to(0.31, 0.48)
        elif t >= 6.5:
            a[0].jitter = a[1].jitter = 0.002; a[0].arms_up = a[1].arms_up = False
    events = run(seated_actors(), script, 30)
    assert first(events, Behavior.FIGHTING) is None


def test_sustained_scuffle_fires_fighting_after_min_duration_with_both_tracks():
    def script(t, a):
        if t >= 5:
            a[4].x, a[5].x = 0.50, 0.56
            a[4].jitter = a[5].jitter = 0.025
            a[4].arms_up = a[5].arms_up = True
    th = BehaviorThresholds()
    events = run(seated_actors(), script, 20, th)
    hit = first(events, Behavior.FIGHTING)
    assert hit is not None
    ts, ev = hit
    assert 5 + th.fighting.min_duration_s <= ts <= 5 + th.fighting.min_duration_s + 1.5
    assert len(ev.track_ids) == 2
    assert ev.score >= th.fighting.confidence_floor


def test_solo_flailing_far_from_others_does_not_fire_fighting():
    actors = [Actor(id=1, x=0.15, y=0.48, jitter=0.002), Actor(id=2, x=0.85, y=0.85, jitter=0.002)]

    def script(t, a):
        if t >= 3:
            a[0].jitter, a[0].arms_up = 0.03, True
    assert first(run(actors, script, 20), Behavior.FIGHTING) is None


def test_two_people_walking_together_do_not_fire_fighting():
    actors = [Actor(id=1, x=0.10, y=0.5, jitter=0.001, speed=0.15),
              Actor(id=2, x=0.17, y=0.5, jitter=0.001, speed=0.15)]

    def script(t, a):
        if t < 0.2:
            a[0].move_to(0.80, 0.5); a[1].move_to(0.87, 0.5)
    assert first(run(actors, script, 12, zones=[]), Behavior.FIGHTING) is None


def test_head_down_and_still_fires_sleeping():
    def script(t, a):
        if t >= 2:
            a[3].head_drop, a[3].jitter = 1.0, 0.0005
    th = BehaviorThresholds()
    hit = first(run(seated_actors(), script, 40, th), Behavior.SLEEPING)
    assert hit is not None
    ts, ev = hit
    assert abs(ts - (2 + th.sleeping.min_duration_s)) < 1.0
    assert ev.track_ids == [4]


def test_head_down_but_fidgeting_does_not_fire_sleeping():
    def script(t, a):
        if t >= 2:
            a[3].head_drop, a[3].jitter = 1.0, 0.01     # ~1 box width/s of motion
    assert first(run(seated_actors(), script, 40), Behavior.SLEEPING) is None


def test_leaving_seat_zone_fires_out_of_seat_and_returning_stops_it():
    def script(t, a):
        if 3 <= t < 3.2:
            a[2].move_to(0.9, 0.2)
        if 25 <= t < 25.2:
            a[2].move_to(*SEATS[2])
    th = BehaviorThresholds()
    events = run(seated_actors(), script, 45, th)
    oos = [(ts, ev) for ts, ev in events if ev.behavior is Behavior.OUT_OF_SEAT]
    assert oos and oos[0][1].track_ids == [3] and oos[0][1].evidence["home_zone"] == "S2"
    assert 3 + th.out_of_seat.min_duration_s <= oos[0][0] <= 3 + th.out_of_seat.min_duration_s + 2
    assert max(ts for ts, _ in oos) < 30   # stops once back in seat


def test_person_never_seated_is_not_out_of_seat():
    actors = seated_actors() + [Actor(id=9, x=0.9, y=0.2, jitter=0.002)]    # teacher at the front
    assert first(run(actors, lambda t, a: None, 40), Behavior.OUT_OF_SEAT) is None


def test_talking_is_off_by_default_and_capped_low_when_enabled():
    def script(t, a):
        a[0].x = 0.30 + (0.002 if int(t * 10) % 2 else -0.002)   # head turning proxy via yaw jitter
    assert first(run(seated_actors(), script, 30), Behavior.TALKING) is None
    th = BehaviorThresholds(talking=TalkingThresholds(enabled=True, head_turn_delta=0.0, min_duration_s=2))
    events = run(seated_actors(), script, 30, th)
    talking = [ev for _, ev in events if ev.behavior is Behavior.TALKING]
    assert talking
    assert all(ev.score <= th.talking.max_confidence for ev in talking)
    assert all(ev.confidence.value == "low" for ev in talking)
