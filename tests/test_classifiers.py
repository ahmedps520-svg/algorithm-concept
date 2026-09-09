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


def test_walking_past_seated_students_does_not_fire_fighting():
    actors = seated_actors()

    def script(t, a):
        if t < 0.2:
            a[2].move_to(0.12, 0.85)       # from the top-right seat, brushing past 2, 1, 4 and 5
        if 12 <= t < 12.2:
            a[2].move_to(*SEATS[2])
    assert first(run(actors, script, 30), Behavior.FIGHTING) is None


def test_head_down_and_still_fires_sleeping():
    def script(t, a):
        if t >= 2:
            a[3].head_drop, a[3].jitter = 1.0, 0.0005
    th = BehaviorThresholds()
    hit = first(run(seated_actors(), script, 40, th), Behavior.SLEEPING)
    assert hit is not None
    ts, ev = hit
    assert abs(ts - (2 + th.sleeping.min_duration_s)) < 2.0   # the baseline cue can fire a little before the absolute one
    assert ev.track_ids == [4]


def test_head_slumping_toward_shoulders_fires_sleeping_via_baseline():
    """Eye-level webcam: the nose never goes below the shoulder line, but the head height
    collapses to ~45 % of this person's own upright baseline."""
    def script(t, a):
        if t >= 6:
            a[3].head_drop, a[3].jitter = 0.5, 0.0005      # nose ends level with the shoulders, not below them
    th = BehaviorThresholds()
    hit = first(run(seated_actors(), script, 45, th), Behavior.SLEEPING)
    assert hit is not None
    ts, ev = hit
    assert ev.evidence["cue"] == "head dropped vs own baseline"
    assert ev.evidence["head_drop"] < th.sleeping.head_below_shoulders_ratio
    assert abs(ts - (6 + th.sleeping.min_duration_s)) < 1.5


def test_upright_person_shifting_posture_never_fires_sleeping():
    """The user-reported false positive: sitting still, facing the camera, head above the
    shoulders the whole time, but leaning in and out so head height varies a lot. The relative
    cue must not fire while the nose is still well above shoulder level."""
    def script(t, a):
        a[3].head_drop = 0.30 if int(t) % 6 < 3 else 0.0     # leans in and back, never head-down
    events = run(seated_actors(), script, 60)
    assert first(events, Behavior.SLEEPING) is None


def test_eyes_visible_at_normal_height_vetoes_sleeping():
    from classroom_monitor.behavior.features import Observation, TrackState
    from classroom_monitor.behavior.classifiers import SleepingClassifier
    from classroom_monitor.config import SleepingThresholds

    t = SleepingThresholds(min_duration_s=1.0, window_s=5.0)
    clf = SleepingClassifier(t, fps=10)
    st = TrackState(1, 5.0)
    fired = False
    for i in range(60):
        o = Observation(i / 10, 0.5, 0.5, 0.1, 0.3, 0.5, 0.65, motion=0.0, vx=0, vy=0,
                        head_drop=0.4, arms_up=False, head_yaw=0.0, pose_available=True,
                        head_height=0.1, head_tilt=50.0, nose=(0.5, 0.3), shoulder_w=0.08,
                        eyes_visible=True)
        o.rel_drop = 0.05                       # head at its normal height
        st.push(o)
        fired |= clf.evaluate(st, [], o.ts) is not None
    assert not fired, "eyes visible at normal head height must veto sleeping"


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


def test_talking_requires_a_neighbour_and_is_capped_low():
    """Talking is on but deliberately weak: it needs someone to talk to, and can never exceed
    LOW confidence from video alone."""
    def turning(t, a):
        a[0].head_yaw_drive = t          # unused by the source; yaw comes from geometry below
        a[0].x = 0.15 + (0.01 if int(t * 10) % 4 < 2 else -0.01)

    th = BehaviorThresholds(talking=TalkingThresholds(head_turn_delta=0.0, min_duration_s=2))
    events = run(seated_actors(), turning, 30, th)
    talking = [ev for _, ev in events if ev.behavior is Behavior.TALKING]
    assert talking, "a head-turning student next to neighbours should produce talking events"
    assert all(ev.score <= th.talking.max_confidence for ev in talking)
    assert all(ev.confidence.value == "low" for ev in talking)
    assert all(ev.evidence["neighbour"] is not None for ev in talking)

    # Alone in frame: no neighbour, so no talking event however much they turn.
    alone = [Actor(id=1, x=0.15, y=0.48, jitter=0.002)]
    th2 = BehaviorThresholds(talking=TalkingThresholds(head_turn_delta=0.0, min_duration_s=2))
    assert first(run(alone, turning, 30, th2, zones=[]), Behavior.TALKING) is None


def test_talking_can_be_disabled_per_room():
    th = BehaviorThresholds(talking=TalkingThresholds(enabled=False, head_turn_delta=0.0, min_duration_s=2))
    assert first(run(seated_actors(), lambda t, a: None, 30, th), Behavior.TALKING) is None
