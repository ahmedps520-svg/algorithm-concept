"""Configuration models.

Two layers:
  * ``SystemConfig``  – global settings (storage, retention, dashboard, model choice).
  * ``RoomConfig``    – one per classroom: camera source, seat zones, per-behavior thresholds.

Per-behavior thresholds live in ``BehaviorThresholds``; every room may override any of them,
so sensitivity is tunable room by room without touching code.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------------------------------------------
# Behavior thresholds
# --------------------------------------------------------------------------------------
class FightingThresholds(BaseModel):
    """Aggression detector. Deliberately conservative: horseplay and real fights look alike
    in short windows, so every signal must be *sustained*."""

    enabled: bool = True
    # Window over which evidence is accumulated (seconds).
    window_s: float = 4.0
    # Fraction of frames in the window that must individually score as "aggressive".
    min_active_fraction: float = 0.6
    # Minimum continuous duration (seconds) of the active condition before an event fires.
    min_duration_s: float = 3.0
    # Subject motion (box widths per second) above this counts as a spike. Slow walking is ~1.
    motion_spike: float = 0.6
    # Two people are "nearby" if the gap between their boxes is under this many box-widths.
    proximity_box_widths: float = 0.6
    # Relative speed between the two nearby tracks (box widths/s) above which they are "engaging".
    # Two people walking together share a velocity and score ~0 here; a scuffle scores high.
    relative_motion: float = 0.5
    # The partner must also be moving at least this fast (box widths/s) for the relative-motion
    # cue to count. Walking past a seated student produces high relative speed but a still partner.
    partner_motion: float = 0.3
    # Wrist above shoulder counts as an arm raise; how many arm-raise frames are required.
    arm_raise_min_frames: int = 6
    # Per-frame score threshold (0-1) for a frame to count as active.
    frame_score_threshold: float = 0.6
    confidence_floor: float = 0.5


class SleepingThresholds(BaseModel):
    enabled: bool = True
    # Head (nose) must sit below the shoulder line by this fraction of torso height.
    head_below_shoulders_ratio: float = 0.15
    # Mean motion over the last second (box widths/s) below this counts as still.
    # Detector box jitter alone is ~0.1-0.2, so keep this comfortably above that.
    max_motion: float = 0.3
    # Must hold for this long.
    min_duration_s: float = 20.0
    window_s: float = 25.0
    min_active_fraction: float = 0.85


class OutOfSeatThresholds(BaseModel):
    enabled: bool = True
    # Fraction of the track's foot-point history outside every seat zone.
    min_active_fraction: float = 0.8
    min_duration_s: float = 8.0
    window_s: float = 10.0
    # Grace region around each seat zone (fraction of frame size) so leaning over isn't flagged.
    zone_margin: float = 0.03
    # Only people who were first seen *in* a seat zone can be "out of seat". This keeps the
    # teacher, visitors and students entering the room from triggering.
    require_home_zone: bool = True


class TalkingThresholds(BaseModel):
    """Video-only talking detection is LOW CONFIDENCE at classroom distances.

    At 540p from a ceiling camera a student's mouth is a handful of pixels, so lip motion is
    not resolvable; the only usable cue is repeated head turning toward a neighbour. Events are
    therefore hard-capped at ``max_confidence`` and can never reach HIGH, never flag a clip on
    their own, and are meant as a nudge for a staff member to look, nothing more. Pair with a
    per-room audio level channel before treating them as meaningful."""

    enabled: bool = True
    # Head-yaw change (normalised nose-to-ear asymmetry) above this suggests turning to a neighbour.
    head_turn_delta: float = 0.12
    proximity_box_widths: float = 0.8
    # Require a neighbour within reach. True for classrooms (talking needs someone to talk to);
    # set False only for single-person testing, where it degrades to "repeated head turning".
    require_neighbour: bool = True
    min_duration_s: float = 5.0
    window_s: float = 8.0
    min_active_fraction: float = 0.5
    # Hard cap: talking events can never exceed this confidence from video alone.
    max_confidence: float = 0.35


class BehaviorThresholds(BaseModel):
    fighting: FightingThresholds = FightingThresholds()
    sleeping: SleepingThresholds = SleepingThresholds()
    out_of_seat: OutOfSeatThresholds = OutOfSeatThresholds()
    talking: TalkingThresholds = TalkingThresholds()


# --------------------------------------------------------------------------------------
# Alerts / clips / retention
# --------------------------------------------------------------------------------------
class DebounceConfig(BaseModel):
    # Same behavior for the same track: suppress re-alerts for this long after the last alert.
    per_track_cooldown_s: float = 120.0
    # Same behavior anywhere in the same room: suppress duplicates within this window.
    per_room_cooldown_s: float = 30.0
    # If an alert for the same track+behavior is still awaiting review, merge rather than re-alert.
    merge_into_open_alert: bool = True


class ClipConfig(BaseModel):
    pre_roll_s: float = 10.0
    post_roll_s: float = 10.0
    fps: float = 10.0            # frames kept in the ring buffer / written to clip
    directory: Path = Path("data/clips")
    # "mjpeg_dir" writes JPEG frames + index.json (no video codec needed); "mp4" needs OpenCV.
    format: Literal["mjpeg_dir", "mp4"] = "mjpeg_dir"


class RetentionConfig(BaseModel):
    """Enforced by ``clips.retention.RetentionEnforcer``.

    * Clips whose alert was never confirmed by a human are deleted after ``unreviewed_hours``.
    * Clips a human dismissed are deleted after ``dismissed_hours``.
    * Clips a human marked actionable are kept for ``actionable_days``.
    """

    unreviewed_hours: float = 48.0
    dismissed_hours: float = 24.0
    actionable_days: float = 90.0
    sweep_interval_s: float = 300.0


class StorageConfig(BaseModel):
    sqlite_path: Path = Path("data/monitor.db")


class ModelConfig(BaseModel):
    # "ultralytics" runs YOLOv8 + ByteTrack + YOLOv8-pose; "stub" runs no ML (tests / synthetic demo).
    backend: Literal["ultralytics", "stub"] = "stub"
    detector_weights: str = "yolov8n.pt"
    pose_weights: str = "yolov8n-pose.pt"
    device: str = "cpu"
    detection_conf: float = 0.4
    tracker: str = "bytetrack.yaml"


class DashboardConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    # Live grid frames are pushed at this FPS regardless of pipeline FPS.
    preview_fps: float = 4.0


class SystemConfig(BaseModel):
    models: ModelConfig = ModelConfig()
    clips: ClipConfig = ClipConfig()
    retention: RetentionConfig = RetentionConfig()
    debounce: DebounceConfig = DebounceConfig()
    storage: StorageConfig = StorageConfig()
    dashboard: DashboardConfig = DashboardConfig()
    # Defaults every room inherits unless it overrides.
    default_thresholds: BehaviorThresholds = BehaviorThresholds()
    # Processing FPS target per room; frames above this are dropped.
    process_fps: float = 10.0


# --------------------------------------------------------------------------------------
# Rooms
# --------------------------------------------------------------------------------------
class SeatZone(BaseModel):
    """Axis-aligned seat zone in normalised frame coordinates (0-1)."""

    id: str
    x1: float
    y1: float
    x2: float
    y2: float

    @model_validator(mode="after")
    def _ordered(self) -> "SeatZone":
        if not (0 <= self.x1 < self.x2 <= 1 and 0 <= self.y1 < self.y2 <= 1):
            raise ValueError(f"seat zone {self.id}: coordinates must be normalised and ordered")
        return self

    def contains(self, x: float, y: float, margin: float = 0.0) -> bool:
        return (self.x1 - margin) <= x <= (self.x2 + margin) and (self.y1 - margin) <= y <= (self.y2 + margin)


class CameraConfig(BaseModel):
    # rtsp://user:pass@host:554/stream ; a file path ; or "synthetic" for the demo source.
    url: str
    transport: Literal["tcp", "udp"] = "tcp"
    reconnect_delay_s: float = 3.0
    # Downscale factor before inference (1.0 = native). 1080p cameras -> 0.5 is a good start.
    scale: float = 1.0


class RoomConfig(BaseModel):
    id: str
    name: str
    camera: CameraConfig
    seat_zones: list[SeatZone] = Field(default_factory=list)
    # Partial override: any field omitted falls back to SystemConfig.default_thresholds.
    thresholds: dict = Field(default_factory=dict)

    def resolved_thresholds(self, defaults: BehaviorThresholds) -> BehaviorThresholds:
        merged = defaults.model_dump()
        for behavior, overrides in self.thresholds.items():
            if behavior not in merged:
                raise ValueError(f"room {self.id}: unknown behavior '{behavior}' in thresholds")
            merged[behavior].update(overrides)
        return BehaviorThresholds.model_validate(merged)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
def load_system_config(path: Path | str) -> SystemConfig:
    with open(path) as f:
        return SystemConfig.model_validate(yaml.safe_load(f) or {})


def load_room_configs(directory: Path | str) -> list[RoomConfig]:
    rooms: list[RoomConfig] = []
    for p in sorted(Path(directory).glob("*.yaml")):
        with open(p) as f:
            rooms.append(RoomConfig.model_validate(yaml.safe_load(f)))
    ids = [r.id for r in rooms]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate room ids in {directory}: {ids}")
    return rooms
