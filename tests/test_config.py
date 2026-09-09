from pathlib import Path

import pytest

from classroom_monitor.config import (
    BehaviorThresholds, RoomConfig, SeatZone, load_room_configs, load_system_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_configs_load():
    s = load_system_config(ROOT / "config/system.yaml")
    rooms = load_room_configs(ROOT / "config/rooms")
    assert {r.id for r in rooms} == {"room_101", "room_102"}
    assert s.default_thresholds.talking.enabled is True
    assert s.default_thresholds.talking.max_confidence <= 0.5   # never more than LOW


def test_room_overrides_merge_with_defaults():
    defaults = BehaviorThresholds()
    room = RoomConfig(id="x", name="x", camera={"url": "synthetic"},
                      thresholds={"fighting": {"min_duration_s": 6.0}})
    t = room.resolved_thresholds(defaults)
    assert t.fighting.min_duration_s == 6.0
    assert t.fighting.window_s == defaults.fighting.window_s
    assert t.sleeping == defaults.sleeping


def test_unknown_behavior_override_rejected():
    room = RoomConfig(id="x", name="x", camera={"url": "synthetic"}, thresholds={"dancing": {}})
    with pytest.raises(ValueError):
        room.resolved_thresholds(BehaviorThresholds())


def test_seat_zone_validation_and_margin():
    z = SeatZone(id="a", x1=0.1, y1=0.1, x2=0.3, y2=0.3)
    assert z.contains(0.2, 0.2) and not z.contains(0.32, 0.2) and z.contains(0.32, 0.2, margin=0.03)
    with pytest.raises(ValueError):
        SeatZone(id="b", x1=0.5, y1=0.1, x2=0.3, y2=0.3)
