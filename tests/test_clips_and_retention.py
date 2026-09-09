import json

import numpy as np

from classroom_monitor.alerts import Alert, AlertStatus, ReviewOutcome
from classroom_monitor.clips import ClipWriter, RetentionEnforcer
from classroom_monitor.config import ClipConfig, RetentionConfig


def frame(v=0):
    return np.full((36, 64, 3), v, dtype=np.uint8)


def make_alert(ts, room="r1"):
    return Alert(room, "fighting", [1, 2], 0.8, "high", 3.0, ts, ts)


def test_clip_has_pre_and_post_roll(tmp_path, store):
    cfg = ClipConfig(pre_roll_s=2, post_roll_s=2, fps=10, directory=tmp_path)
    w = ClipWriter(cfg, store)
    for i in range(50):                      # 5s of history; only the last 2s are retained
        w.observe(i / 10, frame(i))
    a = make_alert(4.9)
    store.upsert(a)
    w.start(a)
    for i in range(50, 80):
        w.observe(i / 10, frame(i))
    clip = store.get_clip(store.get(a.id).clip_id)
    idx = json.loads((tmp_path / "r1" / clip["id"] / "index.json").read_text())
    offsets = [f["offset_s"] for f in idx["frames"]]
    assert min(offsets) >= -2.0 and max(offsets) >= 2.0
    assert clip["frames"] == len(idx["frames"]) and 38 <= clip["frames"] <= 42
    assert (tmp_path / "r1" / clip["id"] / "frames" / "000000.png").exists()


def test_no_clip_without_alert(tmp_path, store):
    w = ClipWriter(ClipConfig(directory=tmp_path), store)
    for i in range(100):
        w.observe(i / 10, frame())
    assert not list(tmp_path.rglob("*"))


def _seed(store, tmp_path, engine, clock):
    """Three clips: unreviewed, dismissed, confirmed."""
    cfg = ClipConfig(pre_roll_s=0.5, post_roll_s=0.5, fps=10, directory=tmp_path)
    ids = {}
    for name in ("unreviewed", "dismissed", "confirmed"):
        w = ClipWriter(cfg, store, clock)
        a = make_alert(clock())
        store.upsert(a)
        w.start(a)
        for i in range(1, 12):
            w.observe(clock() + i / 10, frame())
        ids[name] = a.id
    engine.acknowledge(ids["dismissed"], "s"); engine.review(ids["dismissed"], "s", ReviewOutcome.DISMISS)
    engine.acknowledge(ids["confirmed"], "s"); engine.review(ids["confirmed"], "s", ReviewOutcome.CONFIRM)
    return ids


def test_retention_policy_per_review_outcome(tmp_path, store, engine, clock):
    ids = _seed(store, tmp_path, engine, clock)
    ret = RetentionEnforcer(RetentionConfig(unreviewed_hours=48, dismissed_hours=24, actionable_days=90), store, clock)
    paths = {k: tmp_path / "r1" / store.get(v).clip_id for k, v in ids.items()}
    assert all(p.exists() for p in paths.values())

    assert ret.sweep_once() == []
    clock.advance(25 * 3600)
    assert ret.sweep_once() == [store.get(ids["dismissed"]).clip_id]
    assert not paths["dismissed"].exists() and paths["unreviewed"].exists()
    clock.advance(24 * 3600)
    assert ret.sweep_once() == [store.get(ids["unreviewed"]).clip_id]
    assert paths["confirmed"].exists()
    clock.advance(89 * 86400)
    assert ret.sweep_once() == [store.get(ids["confirmed"]).clip_id]
    assert not any(p.exists() for p in paths.values())
    assert store.get_clip(store.get(ids["confirmed"]).clip_id)["deleted_at"] == clock()
    assert ret.sweep_once() == []


def test_unreviewed_clip_purged_even_if_acknowledged(tmp_path, store, engine, clock):
    ids = _seed(store, tmp_path, engine, clock)
    engine.acknowledge(ids["unreviewed"], "s")      # acknowledged but never reviewed
    ret = RetentionEnforcer(RetentionConfig(unreviewed_hours=1), store, clock)
    clock.advance(2 * 3600)
    assert store.get(ids["unreviewed"]).clip_id in ret.sweep_once()
    assert store.get(ids["unreviewed"]).status is AlertStatus.ACKNOWLEDGED
