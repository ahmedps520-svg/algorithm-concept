"""Drive the BehaviorEngine with the synthetic source (stub perception) and assert which
behaviors fire, when, and for whom."""
from classroom_monitor.behavior import Behavior, BehaviorEngine
from classroom_monitor.config import BehaviorThresholds, RoomConfig, SeatZone, TalkingThresholds
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


def test_walking_past_seated_students_does_not_fire_fighting():
    actors = seated_actors()

    def script(t, a):
        if t < 0.2:
            a[2].move_to(0.12, 0.85)       # from the top-right seat, brushing past 2, 1, 4 and 5
        if 12 <= t < 12.2:
            a[2].move_to(*SEATS[2])
    assert first(run(actors, script, 30), Behavior.FIGHTING) is None


def test_looking_down_does_not_fire_sleeping():
    """The reported false positive: glancing down at a desk or page tips the face but does not
    bring the head down, and the eyes stay visible. That must never read as sleeping."""
    def script(t, a):
        if t >= 2:
            a[3].head_drop, a[3].jitter = 0.55, 0.0005     # nose dips toward, but not past, the shoulders
    assert first(run(seated_actors(), script, 45), Behavior.SLEEPING) is None


def test_head_fully_down_fires_sleeping():
    def script(t, a):
        if t >= 2:
            a[3].head_drop, a[3].jitter = 1.0, 0.0005     # head resting down on the desk
    th = BehaviorThresholds()
    hit = first(run(seated_actors(), script, 45, th), Behavior.SLEEPING)
    assert hit is not None
    ts, ev = hit
    assert ev.evidence["cue"] == "head down below the shoulder line"
    assert ev.evidence["head_drop"] >= th.sleeping.fully_down_ratio
    assert abs(ts - (2 + th.sleeping.min_duration_s)) < 1.5


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


def _talk_events(mouth_at, *, neighbour, enabled=True, seconds=20, fps=10):
    """Drive TalkingClassifier directly: the synthetic source has no face-landmark stream, and
    lip opening is the only cue talking accepts."""
    from classroom_monitor.behavior.classifiers import TalkingClassifier
    from classroom_monitor.behavior.features import Observation, TrackState
    from classroom_monitor.config import TalkingThresholds

    t = TalkingThresholds(enabled=enabled, min_duration_s=2.0)
    clf = TalkingClassifier(t, fps)
    me, other = TrackState(1, 10.0), TrackState(2, 10.0)
    out = []
    for i in range(seconds * fps):
        ts = i / fps
        def obs(cx, mouth):
            return Observation(ts, cx, 0.5, 0.1, 0.3, cx, 0.65, motion=0.0, vx=0, vy=0, head_drop=-0.3,
                               arms_up=False, head_yaw=0.0, pose_available=True, mouth=mouth)
        me.push(obs(0.40, mouth_at(ts)))
        other.push(obs(0.46 if neighbour else 0.95, None))
        ev = clf.evaluate(me, [other], ts)
        if ev is not None:
            out.append(ev)
    return out


def test_talking_is_off_by_default():
    from classroom_monitor.config import TalkingThresholds
    assert TalkingThresholds().enabled is False
    assert BehaviorThresholds().talking.enabled is False
    assert not _talk_events(lambda ts: 0.3 if int(ts * 4) % 2 else 0.0, neighbour=True, enabled=False)


def test_talking_ignores_head_turning_without_lip_movement():
    """Head turning is not a cue any more: it fired on anyone who looked around."""
    assert not _talk_events(lambda ts: 0.05, neighbour=True)          # mouth barely moves
    assert not _talk_events(lambda ts: None, neighbour=True)          # no face-landmark stream at all


def test_talking_needs_lip_movement_and_a_neighbour_and_stays_low():
    speaking = lambda ts: 0.30 if int(ts * 4) % 2 else 0.02
    events = _talk_events(speaking, neighbour=True)
    assert events, "lip movement next to a neighbour should produce talking events"
    th = events[0]
    assert th.evidence["cue"] == "lip movement with a neighbour in reach"
    from classroom_monitor.config import TalkingThresholds
    cap = TalkingThresholds().max_confidence
    assert all(e.score <= cap for e in events)
    assert all(e.confidence.value == "low" for e in events)
    assert not _talk_events(speaking, neighbour=False), "nobody to talk to means no talking event"


def test_talking_can_be_enabled_per_room():
    from classroom_monitor.config import TalkingThresholds
    room = RoomConfig(id="x", name="x", camera={"url": "synthetic"}, thresholds={"talking": {"enabled": True}})
    assert room.resolved_thresholds(BehaviorThresholds()).talking.enabled is True
    assert BehaviorThresholds().talking.enabled is False