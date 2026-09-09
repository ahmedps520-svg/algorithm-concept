#!/usr/bin/env python3
"""Replay a recorded video (or the synthetic scene) through one room's config and print every
behavior event and alert with timestamps. Use it to tune thresholds against known footage.

    python scripts/replay.py config/rooms/room_101.yaml --video incident.mp4 --fps 10
    python scripts/replay.py config/rooms/room_101.yaml --synthetic --duration 120
"""
from __future__ import annotations

import argparse
import logging
import tempfile
from pathlib import Path

from classroom_monitor.alerts import AlertEngine, AlertStore
from classroom_monitor.config import RoomConfig, load_system_config
from classroom_monitor.ingestion.scenarios import default_classroom
from classroom_monitor.ingestion.source import VideoFileSource
from classroom_monitor.pipeline.room import RoomPipeline

import yaml


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("room_yaml")
    ap.add_argument("--system", default="config/system.yaml")
    ap.add_argument("--video", help="path to a recording; requires the [vision] extras")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--duration", type=float, default=120)
    ap.add_argument("--fps", type=float, default=None)
    ap.add_argument("--keep-clips", action="store_true", help="write clips to the configured directory")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    system = load_system_config(args.system)
    if args.fps:
        system.process_fps = args.fps
    room = RoomConfig.model_validate(yaml.safe_load(Path(args.room_yaml).read_text()))
    scratch = Path(tempfile.mkdtemp(prefix="replay-"))
    system.storage.sqlite_path = scratch / "replay.db"
    if not args.keep_clips:
        system.clips.directory = scratch / "clips"

    if args.synthetic or not args.video:
        source = default_classroom(room.id, fps=system.process_fps, realtime=False, duration_s=args.duration)
        backend = None
    else:
        room.camera.url = args.video
        source = VideoFileSource(room.id, room.camera, system.process_fps, loop=False)
        backend = None                       # uses system.models.backend (set to ultralytics for real video)

    store = AlertStore(system.storage.sqlite_path)
    alerts = AlertEngine(store, system.debounce)
    p = RoomPipeline(room, system, store, alerts, source=source, backend=backend)
    t0 = None
    for f in source.frames():
        t0 = f.ts if t0 is None else t0
        for a in p.process_frame(f):
            print(f"[{f.ts - t0:7.1f}s] ALERT {a.behavior:<12} conf={a.confidence:<6} tracks={a.track_ids} "
                  f"sustained={a.sustained_s:.1f}s evidence={a.evidence}")
    p.clips.flush()
    print(f"\n{len(store.list())} alert(s). db: {system.storage.sqlite_path}"
          + (f"  clips: {system.clips.directory}" if args.keep_clips else ""))


if __name__ == "__main__":
    main()
