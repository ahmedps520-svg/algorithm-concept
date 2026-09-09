"""Staff dashboard API + static UI.

AUTH NOTE: this scaffold identifies staff by an ``X-Staff-Id`` header the UI sets from a text
box. Before any real deployment put this behind the school's SSO (reverse-proxy auth header or
OIDC) and derive the staff id from the verified identity, never from client input.
"""
from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, Response
from pydantic import BaseModel

from ..alerts.models import AlertStatus, ReviewOutcome
from ..alerts.engine import ReviewError
from ..pipeline.runner import MonitorSystem

STATIC = Path(__file__).parent / "static"


class ReviewBody(BaseModel):
    outcome: ReviewOutcome
    notes: str = ""


class _Broadcaster:
    """Fan-out of alert events to websocket clients from pipeline threads."""

    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def publish(self, kind: str, payload: dict) -> None:
        if self.loop is None or not self.clients:
            return
        msg = json.dumps({"kind": kind, "payload": payload, "ts": time.time()})
        for ws in list(self.clients):
            asyncio.run_coroutine_threadsafe(self._send(ws, msg), self.loop)

    async def _send(self, ws: WebSocket, msg: str) -> None:
        try:
            await ws.send_text(msg)
        except Exception:  # noqa: BLE001
            self.clients.discard(ws)


def create_app(ms: MonitorSystem) -> FastAPI:
    bus = _Broadcaster()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        bus.loop = asyncio.get_running_loop()
        yield
        bus.loop = None

    app = FastAPI(title="Classroom Monitor", version="0.1", lifespan=lifespan)
    ms.alerts.on_new_alert(lambda a: bus.publish("alert.new", a.to_dict()))
    ms.alerts.on_alert_updated(lambda a: bus.publish("alert.updated", a.to_dict()))

    # ----------------------------------------------------------------- UI
    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC / "index.html").read_text()

    # -------------------------------------------------------------- rooms
    @app.get("/api/rooms")
    def rooms() -> list[dict]:
        open_counts: dict[str, int] = {}
        for a in ms.alerts.queue():
            open_counts[a.room_id] = open_counts.get(a.room_id, 0) + 1
        out = []
        for p in ms.rooms.values():
            s = p.status
            s.open_alerts = open_counts.get(p.room.id, 0)
            d = s.__dict__.copy()
            d["seat_zones"] = [z.model_dump() for z in p.room.seat_zones]
            out.append(d)
        return out

    @app.get("/api/rooms/{room_id}/preview")
    def preview(room_id: str) -> Response:
        p = ms.rooms.get(room_id)
        if p is None:
            raise HTTPException(404, "unknown room")
        pv = p.preview()
        if pv is None:
            raise HTTPException(503, "no frame yet")
        data, mime = pv
        return Response(content=data, media_type=mime, headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------- alerts
    @app.get("/api/alerts/queue")
    def alert_queue() -> list[dict]:
        return [a.to_dict() for a in ms.alerts.queue()]

    @app.get("/api/alerts")
    def alert_log(status: AlertStatus | None = None, room_id: str | None = None, behavior: str | None = None,
                  since: float | None = None, until: float | None = None,
                  limit: int = Query(200, le=1000)) -> list[dict]:
        return [a.to_dict() for a in ms.store.list(status, room_id, behavior, since, until, limit)]

    @app.get("/api/alerts/{alert_id}")
    def alert_get(alert_id: str) -> dict:
        a = ms.store.get(alert_id)
        if a is None:
            raise HTTPException(404, "unknown alert")
        return a.to_dict()

    @app.post("/api/alerts/{alert_id}/ack")
    def alert_ack(alert_id: str, x_staff_id: str = Header(default="")) -> dict:
        try:
            return ms.alerts.acknowledge(alert_id, x_staff_id).to_dict()
        except ReviewError as e:
            raise HTTPException(409, str(e)) from e

    @app.post("/api/alerts/{alert_id}/review")
    def alert_review(alert_id: str, body: ReviewBody, x_staff_id: str = Header(default="")) -> dict:
        try:
            return ms.alerts.review(alert_id, x_staff_id, body.outcome, body.notes).to_dict()
        except ReviewError as e:
            raise HTTPException(409, str(e)) from e

    # -------------------------------------------------------------- clips
    @app.get("/api/clips/{clip_id}")
    def clip_get(clip_id: str) -> dict:
        c = ms.store.get_clip(clip_id)
        if c is None or c["deleted_at"] is not None:
            raise HTTPException(404, "clip not found or expired by retention policy")
        idx = Path(c["path"]) / "index.json"
        if idx.exists():
            return {**c, "index": json.loads(idx.read_text())}
        return {**c, "index": None, "video": f"/api/clips/{clip_id}/video"}

    @app.get("/api/clips/{clip_id}/frames/{name}")
    def clip_frame(clip_id: str, name: str) -> FileResponse:
        c = _live_clip(ms, clip_id)
        f = (Path(c["path"]) / "frames" / name).resolve()
        if not f.is_file() or Path(c["path"]).resolve() not in f.parents:
            raise HTTPException(404, "frame not found")
        return FileResponse(f)

    @app.get("/api/clips/{clip_id}/video")
    def clip_video(clip_id: str) -> FileResponse:
        c = _live_clip(ms, clip_id)
        f = Path(c["path"])
        if not f.is_file():
            raise HTTPException(404, "no video file")
        return FileResponse(f, media_type="video/mp4")

    # ----------------------------------------------------------- verifier
    @app.get("/api/verifier")
    def verifier_status() -> dict:
        if ms.verifier is None:
            return {"enabled": False}
        return {"enabled": True, **ms.verifier.health()}

    @app.post("/api/alerts/{alert_id}/verify")
    def alert_verify(alert_id: str) -> dict:
        """Run (or re-run) the AI second opinion on a saved clip, synchronously."""
        if ms.verifier is None:
            raise HTTPException(409, "verifier is disabled in config")
        a = ms.store.get(alert_id)
        if a is None:
            raise HTTPException(404, "unknown alert")
        if not a.clip_id:
            raise HTTPException(409, "clip not saved yet")
        c = _live_clip(ms, a.clip_id)
        frames = _load_clip_frames(Path(c["path"]))
        if not frames:
            raise HTTPException(409, "clip has no readable frames")
        return ms.verifier.verify_now(a, frames).to_dict()

    # -------------------------------------------------------------- misc
    @app.get("/api/config")
    def config() -> dict:
        return {"retention": ms.system.retention.model_dump(), "debounce": ms.system.debounce.model_dump(),
                "clips": {k: v for k, v in ms.system.clips.model_dump(mode="json").items() if k != "directory"}}

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        bus.clients.add(websocket)
        try:
            while True:
                await websocket.receive_text()   # keepalive pings from the client
        except WebSocketDisconnect:
            pass
        finally:
            bus.clients.discard(websocket)

    return app


def _load_clip_frames(clip_dir: Path) -> list:
    """Frames of an image-sequence clip as (ts, image) pairs, for on-demand verification."""
    from ..clips import imgcodec

    idx = clip_dir / "index.json"
    if not idx.exists():
        return []
    meta = json.loads(idx.read_text())
    out = []
    for f in meta.get("frames", []):
        p = clip_dir / f["file"]
        if p.is_file():
            try:
                out.append((f["ts"], imgcodec.decode(p.read_bytes())))
            except ValueError:
                continue
    return out


def _live_clip(ms: MonitorSystem, clip_id: str) -> dict:
    c = ms.store.get_clip(clip_id)
    if c is None or c["deleted_at"] is not None:
        raise HTTPException(404, "clip not found or expired by retention policy")
    return c
