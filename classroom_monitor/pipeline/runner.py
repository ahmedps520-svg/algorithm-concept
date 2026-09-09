"""Multi-room orchestration + CLI entry point."""
from __future__ import annotations

import argparse
import logging
import signal
import threading
from pathlib import Path

from ..alerts.engine import AlertEngine
from ..alerts.store import AlertStore
from ..clips.retention import RetentionEnforcer
from ..config import RoomConfig, SystemConfig, load_room_configs, load_system_config
from ..verify import OllamaVerifier
from .room import RoomPipeline

log = logging.getLogger(__name__)


class MonitorSystem:
    def __init__(self, system: SystemConfig, rooms: list[RoomConfig]) -> None:
        self.system = system
        self.store = AlertStore(system.storage.sqlite_path)
        self.alerts = AlertEngine(self.store, system.debounce)
        self.retention = RetentionEnforcer(system.retention, self.store)
        self.verifier: OllamaVerifier | None = (
            OllamaVerifier(system.verifier, self.store, notify=self.alerts.notify_updated) if system.verifier.enabled else None
        )
        self.rooms: dict[str, RoomPipeline] = {
            r.id: RoomPipeline(r, system, self.store, self.alerts, verifier=self.verifier) for r in rooms
        }

    def start(self) -> None:
        self.retention.start()
        if self.verifier:
            self.verifier.start()
            log.info("clip verifier: %s at %s -> %s", self.system.verifier.model, self.system.verifier.base_url, self.verifier.health())
        for p in self.rooms.values():
            p.start()
            log.info("started room %s (%s) <- %s", p.room.id, p.room.name, p.room.camera.url)

    def stop(self) -> None:
        for p in self.rooms.values():
            p.stop()
        if self.verifier:
            self.verifier.stop()
        self.retention.stop()
        self.store.close()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Classroom monitoring backend + staff dashboard")
    ap.add_argument("--system", default="config/system.yaml")
    ap.add_argument("--rooms", default="config/rooms")
    ap.add_argument("--no-dashboard", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    system = load_system_config(args.system)
    rooms = load_room_configs(args.rooms)
    if not rooms:
        raise SystemExit(f"no room configs found in {Path(args.rooms).resolve()}")
    ms = MonitorSystem(system, rooms)
    ms.start()

    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    try:
        if args.no_dashboard:
            stop.wait()
        else:
            import uvicorn
            from ..dashboard.app import create_app

            uvicorn.run(create_app(ms), host=system.dashboard.host, port=system.dashboard.port, log_level="info")
    finally:
        ms.stop()


if __name__ == "__main__":
    main()
