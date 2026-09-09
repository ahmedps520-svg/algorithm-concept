import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from classroom_monitor.alerts import Alert, AlertStore
from classroom_monitor.config import VerifierConfig, load_room_configs, load_system_config
from classroom_monitor.verify import OllamaVerifier, build_contact_sheet, parse_verdict
from tests.test_config import ROOT


def frames(n=200, fps=10, t0=100.0):
    return [(t0 + i / fps, np.full((36, 64, 3), i % 255, dtype=np.uint8)) for i in range(n)]


def test_contact_sheet_picks_nearest_frames_and_tiles():
    sheet, used = build_contact_sheet(frames(), trigger_ts=110.0, offsets=[-6, -3, -1, 0, 2, 5], tile_width=64, columns=3)
    assert used == [-6.0, -3.0, -1.0, 0.0, 2.0, 5.0]
    assert sheet.shape == (2 * 36, 3 * 64, 3)
    # frame at +5 s is index 150 -> value 150 in the bottom-right tile (past the separator)
    assert sheet[36 + 10, 2 * 64 + 10, 0] == 150


@pytest.mark.parametrize("text,verdict,conf", [
    ('{"verdict":"likely_real","what_is_happening":"two students grappling","confidence":0.82,"reason":"sustained contact"}', "likely_real", 0.82),
    ('Sure! ```json\n{"verdict": "likely false alarm", "what_is_happening": "high five", "confidence": 90, "reason": "smiling"}\n```', "likely_false_alarm", 0.9),
    ('I think {"verdict":"maybe","confidence":"x","what_is_happening":"","reason":""} ok', "unclear", 0.0),
])
def test_parse_verdict_is_tolerant(text, verdict, conf):
    v = parse_verdict(text)
    assert v["verdict"] == verdict and abs(v["confidence"] - conf) < 1e-9


def test_parse_verdict_rejects_garbage():
    with pytest.raises(ValueError):
        parse_verdict("no json here")


class FakeOllama:
    """Records requests; answers /api/tags and /api/chat."""

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def __call__(self, url, body, timeout):
        self.calls.append((url, body))
        if url.endswith("/api/tags"):
            return {"models": [{"name": "qwen3-vl:8b"}]}
        assert body["model"] == "qwen3-vl:8b" and body["stream"] is False
        assert body["messages"][0]["images"][0].startswith("iVBOR") or len(body["messages"][0]["images"][0]) > 100
        assert "flagged:" in body["messages"][0]["content"] and "Reply with JSON only" in body["messages"][0]["content"]
        return {"message": {"content": self.reply}}


def test_verify_now_attaches_verdict_without_touching_status(store, clock):
    a = Alert("r1", "fighting", [1, 2], 0.8, "high", 3.0, 110.0, 110.0)
    store.upsert(a)
    fake = FakeOllama('{"verdict":"likely_false_alarm","what_is_happening":"two students play-fighting and laughing","confidence":0.7,"reason":"open hands, no force"}')
    seen = []
    v = OllamaVerifier(VerifierConfig(enabled=True, tile_width=64), store, notify=seen.append, transport=fake, clock=clock)
    assert v.health()["model_pulled"] is True
    out = v.verify_now(a, frames())
    assert out.ai_verdict == "likely_false_alarm" and out.ai_confidence == 0.7 and out.ai_model == "qwen3-vl:8b"
    assert store.get(a.id).ai_summary.startswith("two students") and store.get(a.id).status.value == "new"
    assert seen == [out] and len(fake.calls) == 2


def test_thinking_model_with_empty_content_is_still_parsed(store, clock):
    """qwen3-vl reasons first. If a build ignores think=false the answer arrives in
    `thinking` with `content` empty, which used to surface as 'no JSON verdict in reply:'."""
    a = Alert("r1", "fighting", [1, 2], 0.8, "high", 3.0, 110.0, 110.0)
    store.upsert(a)
    sent = []

    def thinking(url, body, timeout):
        sent.append(body)
        return {"message": {"content": "", "thinking": 'Let me look. {"verdict":"likely_real","what_is_happening":"two students grappling","confidence":0.75,"reason":"sustained contact"}'}}

    v = OllamaVerifier(VerifierConfig(enabled=True, tile_width=64), store, transport=thinking, clock=clock)
    out = v.verify_now(a, frames())
    assert out.ai_verdict == "likely_real" and out.ai_confidence == 0.75
    assert sent[0]["think"] is False and sent[0]["options"]["num_predict"] >= 400


def test_empty_reply_is_reported_as_error_with_reason(store, clock):
    a = Alert("r1", "sleeping", [4], 0.9, "high", 20.0, 110.0, 110.0)
    store.upsert(a)
    v = OllamaVerifier(VerifierConfig(enabled=True, tile_width=64), store,
                       transport=lambda u, b, t: {"message": {"content": "  "}, "done_reason": "length"}, clock=clock)
    out = v.verify_now(a, frames())
    assert out.ai_verdict == "error" and "no text" in out.ai_reason and "length" in out.ai_reason


def test_verifier_error_is_recorded_not_raised(store, clock):
    a = Alert("r1", "sleeping", [4], 0.9, "high", 20.0, 110.0, 110.0)
    store.upsert(a)

    def down(url, body, timeout):
        raise OSError("connection refused")
    v = OllamaVerifier(VerifierConfig(enabled=True, tile_width=64), store, transport=down, clock=clock)
    assert v.health()["reachable"] is False
    out = v.verify_now(a, frames())
    assert out.ai_verdict == "error" and "connection refused" in out.ai_reason


def test_pipeline_runs_verifier_when_clip_is_ready_and_api_exposes_it(tmp_path):
    """Scenario -> alert -> clip saved -> verifier called with in-memory frames -> verdict on the
    alert via the API; on-demand re-verify decodes the saved clip from disk."""
    from classroom_monitor.dashboard.app import create_app
    from classroom_monitor.ingestion.scenarios import default_classroom
    from classroom_monitor.pipeline.room import RoomPipeline
    from classroom_monitor.pipeline.runner import MonitorSystem

    system = load_system_config(ROOT / "config/system.yaml")
    system.storage.sqlite_path = tmp_path / "m.db"; system.clips.directory = tmp_path / "clips"
    system.verifier = VerifierConfig(enabled=True, tile_width=64)
    rooms = load_room_configs(ROOT / "config/rooms")
    ms = MonitorSystem(system, rooms)
    fake = FakeOllama('{"verdict":"likely_real","what_is_happening":"a student has their head down on the desk","confidence":0.85,"reason":"still for the whole clip"}')
    ms.verifier.transport = fake
    room = rooms[0]
    src = default_classroom(room.id, fps=10, realtime=False, duration_s=45)
    p = RoomPipeline(room, system, ms.store, ms.alerts, source=src, verifier=ms.verifier)
    ms.rooms[room.id] = p
    for f in src.frames():
        p.process_frame(f)
    p.clips.flush()
    # the worker thread is not started in this test: drain the queue synchronously
    assert ms.verifier._q.qsize() == 1
    alert_id, fr, trig = ms.verifier._q.get()
    ms.verifier.verify_now(ms.store.get(alert_id), fr, trig)
    a = ms.store.get(alert_id)
    assert a.behavior == "out_of_seat" and a.ai_verdict == "likely_real" and a.ai_confidence == 0.85
    assert len(fr) >= 150   # pre-roll + post-roll frames were handed over in memory

    c = TestClient(create_app(ms))
    assert c.get("/api/verifier").json()["model_pulled"] is True
    q = c.get("/api/alerts/queue").json()
    assert q[0]["ai_verdict"] == "likely_real" and q[0]["ai_summary"].startswith("a student")
    fake.reply = '{"verdict":"unclear","what_is_happening":"hard to see","confidence":0.3,"reason":"low light"}'
    r = c.post(f"/api/alerts/{alert_id}/verify").json()          # re-run from the clip on disk
    assert r["ai_verdict"] == "unclear" and r["ai_confidence"] == 0.3
