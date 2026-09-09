"""Full pipeline on the shipped synthetic scenario: source -> stub perception -> behavior ->
alerts -> clips, then the dashboard API on top of it."""
import pytest
from fastapi.testclient import TestClient

from classroom_monitor.alerts import AlertEngine, AlertStore
from classroom_monitor.config import load_room_configs, load_system_config
from classroom_monitor.dashboard.app import create_app
from classroom_monitor.ingestion.scenarios import default_classroom
from classroom_monitor.pipeline.room import RoomPipeline
from classroom_monitor.pipeline.runner import MonitorSystem
from tests.test_config import ROOT


@pytest.fixture(scope="module")
def ran(tmp_path_factory):
    d = tmp_path_factory.mktemp("e2e")
    system = load_system_config(ROOT / "config/system.yaml")
    system.storage.sqlite_path = d / "m.db"
    system.clips.directory = d / "clips"
    rooms = load_room_configs(ROOT / "config/rooms")
    ms = MonitorSystem(system, rooms)
    room = rooms[0]
    src = default_classroom(room.id, fps=10, realtime=False, duration_s=100)
    p = RoomPipeline(room, system, ms.store, ms.alerts, source=src)
    ms.rooms[room.id] = p
    new = []
    for f in src.frames():
        new += [(f.ts, a) for a in p.process_frame(f)]
    p.clips.flush()
    return ms, new


def test_scenario_produces_exactly_the_expected_alerts(ran):
    ms, new = ran
    by_behavior = {a.behavior: ts for ts, a in new}
    assert set(by_behavior) == {"out_of_seat", "sleeping", "fighting"}
    assert 24 <= by_behavior["out_of_seat"] <= 28
    assert 50 <= by_behavior["sleeping"] <= 54
    assert 83 <= by_behavior["fighting"] <= 85          # not during the 60-61.5s horseplay
    assert all(a.clip_id for a in ms.store.list())


def test_dashboard_review_flow(ran):
    ms, _ = ran
    c = TestClient(create_app(ms))
    rooms = c.get("/api/rooms").json()
    assert {r["room_id"] for r in rooms} == {"room_101", "room_102"}
    assert c.get("/api/rooms/room_101/preview").status_code == 200

    q = c.get("/api/alerts/queue").json()
    assert len(q) == 3 and all(a["status"] == "new" for a in q)
    fight = next(a for a in q if a["behavior"] == "fighting")

    # Review before acknowledgment is refused; missing staff id is refused.
    assert c.post(f"/api/alerts/{fight['id']}/review", json={"outcome": "confirm"}, headers={"X-Staff-Id": "s"}).status_code == 409
    assert c.post(f"/api/alerts/{fight['id']}/ack").status_code == 409

    assert c.post(f"/api/alerts/{fight['id']}/ack", headers={"X-Staff-Id": "j.smith"}).json()["status"] == "acknowledged"
    r = c.post(f"/api/alerts/{fight['id']}/review", json={"outcome": "confirm", "notes": "real"}, headers={"X-Staff-Id": "j.smith"}).json()
    assert r["status"] == "confirmed" and r["clip_flag"] == "actionable"

    clip = c.get(f"/api/clips/{r['clip_id']}").json()
    assert clip["index"]["behavior"] == "fighting" and clip["index"]["frames"]
    first = clip["index"]["frames"][0]["file"]
    assert c.get(f"/api/clips/{r['clip_id']}/{first}").status_code == 200
    assert c.get(f"/api/clips/{r['clip_id']}/frames/../index.json").status_code in (404, 400)

    log = c.get("/api/alerts", params={"status": "confirmed", "room_id": "room_101"}).json()
    assert [a["id"] for a in log] == [fight["id"]]
    assert len(c.get("/api/alerts/queue").json()) == 2
