from classroom_monitor.behavior.sustain import SustainedCondition


def feed(gate, start, seconds, active, fps=10):
    res = None
    for i in range(int(seconds * fps)):
        res = gate.update(start + i / fps, active)
    return res


def test_single_frame_never_triggers():
    g = SustainedCondition(window_s=4, min_duration_s=0.0, min_active_fraction=0.5)
    assert not g.update(0.0, True).triggered


def test_short_burst_does_not_trigger():
    g = SustainedCondition(window_s=4, min_duration_s=3, min_active_fraction=0.6)
    assert not feed(g, 0, 1.5, True).triggered
    assert not feed(g, 1.5, 3.0, False).triggered


def test_sustained_triggers_at_min_duration():
    g = SustainedCondition(window_s=4, min_duration_s=3, min_active_fraction=0.6)
    fired_at = None
    for i in range(60):
        r = g.update(i / 10, True)
        if r.triggered and fired_at is None:
            fired_at = i / 10
    assert fired_at == 3.0


def test_long_quiet_history_does_not_delay_a_run():
    g = SustainedCondition(window_s=25, min_duration_s=20, min_active_fraction=0.85)
    feed(g, 0, 60, False)
    fired_at = None
    for i in range(400):
        t = 60 + i / 10
        if g.update(t, True).triggered and fired_at is None:
            fired_at = t
    assert fired_at == 80.0


def test_tolerates_sparse_gaps_but_not_dropouts():
    g = SustainedCondition(window_s=4, min_duration_s=3, min_active_fraction=0.6)
    fired = False
    for i in range(50):
        fired |= g.update(i / 10, i % 5 != 0).triggered      # 80% active
    assert fired
    g.reset()
    fired = False
    for i in range(50):
        fired |= g.update(i / 10, i % 3 == 0).triggered      # 33% active
    assert not fired
