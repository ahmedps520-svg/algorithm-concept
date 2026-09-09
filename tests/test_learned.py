import json

import numpy as np

from classroom_monitor.behavior.learned import FEAT_NAMES, LABELS, SoftmaxModel, frame_features, train, window_features
from classroom_monitor.behavior import Behavior, BehaviorEngine
from classroom_monitor.config import BehaviorThresholds, LearnedThresholds
from classroom_monitor.ingestion.scenarios import seated_actors
from classroom_monitor.ingestion.source import SyntheticSource
from classroom_monitor.perception.backend import StubBackend
from tests.test_classifiers import ZONES


def _windows(rng, label, n):
    """Synthetic window examples with the structure the demo records."""
    out = []
    for _ in range(n):
        frames = []
        for _ in range(20):
            if label == "sleeping":
                f = [0.3 + rng.normal(0, .05), 0.5 + rng.normal(0, .05), 0, 0, 0.05, 0.1, 0, 0, 1, 1, 0, 1]
            elif label == "out_of_seat":
                f = [-0.2, 0, 0, 0, 1.2 + rng.normal(0, .2), 0.8, 0, 0, 0, 1, 0, 1]
            elif label == "talking":
                f = [-0.2, 0, 0, rng.normal(0, .3), 0.1, 0.4, 0, 0.2 + 0.15 * rng.normal(), 1, 0.3, 0.1, 1]
            else:
                f = [-0.2 + rng.normal(0, .03), 0.02, 0, 0, 0.1 + abs(rng.normal(0, .05)), 0.15, 0, 0.02, 1, 1, 0, 1]
            frames.append(f)
        out.append((window_features(frames), label))
    return out


def test_feature_dims_match_demo():
    frames = [[0.0] * len(FEAT_NAMES)] * 8
    assert len(window_features(frames)) == 4 * len(FEAT_NAMES) == 48
    assert window_features(frames[:5]) is None


def test_train_predict_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    data = sum((_windows(rng, l, 40) for l in ["normal", "sleeping", "out_of_seat", "talking"]), [])
    X = np.asarray([x for x, _ in data]); y = [l for _, l in data]
    m = train(X[::2], y[::2])
    acc = np.mean([m.predict(x)[0] == l for x, l in zip(X[1::2], y[1::2])])
    assert acc > 0.95
    m2 = SoftmaxModel.from_json(json.loads(json.dumps(m.to_json())))
    assert m2.predict(X[3])[0] == m.predict(X[3])[0]


def test_learned_classifier_runs_in_engine_and_merges_with_rules(tmp_path):
    """A model that says 'sleeping' whenever the head is down fires through the engine."""
    rng = np.random.default_rng(1)
    data = sum((_windows(rng, l, 60) for l in ["normal", "sleeping"]), [])
    m = train(np.asarray([x for x, _ in data]), [l for _, l in data])
    path = tmp_path / "m.json"; path.write_text(json.dumps({"model": m.to_json()}))
    th = BehaviorThresholds(learned=LearnedThresholds(enabled=True, weights_path=path, min_duration_s={"sleeping": 5.0}))

    def script(t, a):
        if t >= 3:
            a[3].head_drop, a[3].jitter = 1.0, 0.0005
    src = SyntheticSource("r", seated_actors(), script, fps=10, realtime=False, duration_s=40)
    eng = BehaviorEngine("r", th, ZONES, 10); backend = StubBackend()
    learned = []
    for f in src.frames():
        w, h = f.size
        for ev in eng.update(backend.process(f), f.ts, w, h):
            if ev.evidence.get("cue") == "learned model":
                learned.append((f.ts, ev))
    assert learned and learned[0][1].behavior is Behavior.SLEEPING and learned[0][1].track_ids == [4]
    assert learned[0][0] < 3 + 20   # learned model fires before the 20 s rule
