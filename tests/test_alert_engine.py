import pytest

from classroom_monitor.alerts import AlertStatus, ClipFlag, ReviewError, ReviewOutcome
from classroom_monitor.behavior import Behavior, BehaviorEvent


def ev(ts, behavior=Behavior.FIGHTING, tracks=(1, 2), room="r1", score=0.8):
    return BehaviorEvent(room, behavior, list(tracks), ts, score, 3.0, {})


def test_new_alert_created_and_persisted(engine, store):
    a = engine.handle(ev(100))
    assert a is not None and a.status is AlertStatus.NEW
    assert store.get(a.id).track_ids == [1, 2]


def test_repeat_events_merge_into_open_alert(engine, store):
    a = engine.handle(ev(100))
    for t in range(101, 400):
        assert engine.handle(ev(t, score=0.9)) is None
    a = store.get(a.id)
    assert a.occurrences == 300 and a.last_ts == 399 and a.score == 0.9
    assert len(store.list()) == 1


def test_per_track_cooldown_after_review(engine, clock):
    a = engine.handle(ev(100))
    engine.acknowledge(a.id, "staff"); engine.review(a.id, "staff", ReviewOutcome.DISMISS)
    assert engine.handle(ev(150)) is None            # within 120s per-track cooldown
    assert engine.handle(ev(230)) is not None        # cooldown elapsed


def test_per_room_cooldown_suppresses_other_tracks_same_behavior(engine):
    assert engine.handle(ev(100, tracks=(1, 2))) is not None
    assert engine.handle(ev(110, tracks=(3, 4))) is None
    assert engine.handle(ev(140, tracks=(3, 4))) is not None


def test_different_behaviors_and_rooms_are_independent(engine):
    assert engine.handle(ev(100)) is not None
    assert engine.handle(ev(100, behavior=Behavior.SLEEPING, tracks=(1,))) is not None
    assert engine.handle(ev(100, room="r2")) is not None


def test_review_requires_prior_acknowledgment(engine):
    a = engine.handle(ev(100))
    with pytest.raises(ReviewError):
        engine.review(a.id, "staff", ReviewOutcome.CONFIRM)
    assert engine.store.get(a.id).clip_flag is ClipFlag.UNREVIEWED


def test_ack_and_review_require_staff_id(engine):
    a = engine.handle(ev(100))
    with pytest.raises(ReviewError):
        engine.acknowledge(a.id, "  ")
    engine.acknowledge(a.id, "j.smith")
    with pytest.raises(ReviewError):
        engine.review(a.id, "", ReviewOutcome.CONFIRM)


def test_confirm_marks_clip_actionable_with_reviewer(engine, clock):
    a = engine.handle(ev(100))
    engine.acknowledge(a.id, "j.smith")
    clock.advance(30)
    a = engine.review(a.id, "m.jones", ReviewOutcome.CONFIRM, notes="two students, broke it up")
    assert a.status is AlertStatus.CONFIRMED and a.clip_flag is ClipFlag.ACTIONABLE
    assert a.acknowledged_by == "j.smith" and a.reviewed_by == "m.jones"
    assert a.reviewed_at == clock.t and a.review_notes.startswith("two")


def test_dismiss_marks_clip_dismissed(engine):
    a = engine.handle(ev(100))
    engine.acknowledge(a.id, "s")
    a = engine.review(a.id, "s", ReviewOutcome.DISMISS)
    assert a.status is AlertStatus.DISMISSED and a.clip_flag is ClipFlag.DISMISSED


def test_cannot_review_twice_or_ack_twice(engine):
    a = engine.handle(ev(100))
    engine.acknowledge(a.id, "s")
    with pytest.raises(ReviewError):
        engine.acknowledge(a.id, "s")
    engine.review(a.id, "s", ReviewOutcome.CONFIRM)
    with pytest.raises(ReviewError):
        engine.review(a.id, "s", ReviewOutcome.DISMISS)


def test_queue_orders_new_before_acknowledged_oldest_first(engine):
    a1 = engine.handle(ev(100))
    a2 = engine.handle(ev(200, room="r2"))
    a3 = engine.handle(ev(300, room="r3"))
    engine.acknowledge(a1.id, "s")
    assert [a.id for a in engine.queue()] == [a2.id, a3.id, a1.id]


def test_callbacks_fire(engine):
    seen = []
    engine.on_new_alert(lambda a: seen.append(("new", a.id)))
    engine.on_alert_updated(lambda a: seen.append(("upd", a.id)))
    a = engine.handle(ev(100))
    engine.acknowledge(a.id, "s")
    assert seen == [("new", a.id), ("upd", a.id)]
