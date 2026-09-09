# Classroom Monitor

Multi-classroom video monitoring backend with **human-reviewed** behavior alerts for school staff.

**Live demo of the staff dashboard using your own webcam:**
https://ahmedps520-svg.github.io/algorithm-concept/ — source in `docs/index.html`, deployed by
`.github/workflows/pages.yml`. Click "Add camera", allow camera access, draw a seat zone around
yourself. Person detection, tracking and pose run in the browser (MoveNet MultiPose via
TensorFlow.js); the classifiers, alert debounce, review workflow and clip logic are the same as
the Python backend. Video never leaves the device. A "Simulated room" button adds a scripted
room so every alert type can be seen without acting it out.

The demo deploys from `claude/trusting-lamport-a07jjo`, the repository's default branch. The
`github-pages` environment only accepts deployments from the default branch, so a push of
`docs/` to any other branch is rejected before the workflow runs a step.

One fixed camera per room. A per-room pipeline detects and tracks people, estimates pose, runs
rule-based behavior classifiers over sliding windows, and raises alerts to a staff dashboard.
The system never takes action on its own: every alert must be acknowledged and reviewed by a
named staff member before its clip is flagged actionable.

```
camera (RTSP) ─► ingestion ─► detect+track+pose ─► behavior engine ─► alert engine ─► dashboard
                     │                                                      │
                     └── pre-roll ring buffer ──────── clip writer ◄────────┘  (alert-triggered only)
                                                          │
                                                   retention sweeper
```

## What is scaffolded

| Piece | Module | Status |
|---|---|---|
| Ingestion: RTSP / file / synthetic sources, FPS pacing, reconnect | `classroom_monitor/ingestion/` | done |
| Perception: YOLOv8-pose + ByteTrack backend, stub backend, IoU tracker fallback | `classroom_monitor/perception/` | done (ultralytics path untested without weights) |
| Behavior: per-track sliding windows, sustained-condition gate, 4 classifiers | `classroom_monitor/behavior/` | done, tuned on synthetic scenes only |
| Alerts: debounce/cooldown/merge, NEW → ACKNOWLEDGED → CONFIRMED/DISMISSED, SQLite | `classroom_monitor/alerts/` | done |
| Clips: 10s pre/post roll, image-sequence or mp4, retention enforcement | `classroom_monitor/clips/` | done |
| Dashboard: live grid, ack queue, incident log with clip playback, websocket push | `classroom_monitor/dashboard/` | done, **no auth yet** |
| Multi-room runner + CLI | `classroom_monitor/pipeline/` | done |

## Quick start (no cameras, no GPU)

```bash
pip install -e .[dev]
pytest                       # 37 tests: gate timing, each classifier, review workflow, retention, API
classroom-monitor            # two synthetic rooms + dashboard on http://localhost:8080
```

The synthetic scene in each room plays a 2-minute script: a student leaves their seat at 15s,
one puts their head down at 30s, two have 1.5s of horseplay at 60s (must not alert), and two
have an 8s scuffle at 80s. Watch the alert queue fill, enter a staff id, acknowledge, view the
clip, confirm or dismiss.

Replay footage or the synthetic scene against a room config to tune thresholds:

```bash
python scripts/replay.py config/rooms/room_101.yaml --synthetic
python scripts/replay.py config/rooms/room_101.yaml --video incident.mp4   # needs [vision]
```

## Real cameras

```bash
pip install -e .[vision]     # opencv-python-headless + ultralytics (downloads yolov8n-pose.pt on first run)
```

Set `models.backend: ultralytics` in `config/system.yaml`, `device: cuda:0` if you have a GPU,
and put the RTSP URL in each room file:

```yaml
camera:
  url: rtsp://user:pass@10.0.1.101:554/Streaming/Channels/101
  transport: tcp
  scale: 0.5          # 1080p -> 540p before inference; plenty for whole-body pose
```

One process handles many rooms (a thread per room). Rough CPU budget with `yolov8n-pose` at
540p, 10 fps: ~1 core per room on a modern x86 CPU, or ~25 rooms per consumer GPU. Past that,
run one process per server and point all of them at one shared SQLite → Postgres swap in
`alerts/store.py`, or move inference to an edge box per room and forward only
`TrackedPerson` lists (the behavior/alert layers are hardware-independent).

## Configuration

* `config/system.yaml` – models, clip settings, **retention**, **debounce**, dashboard, default thresholds.
* `config/rooms/<room>.yaml` – camera, **seat zones** (normalised 0-1 boxes), per-behavior threshold overrides.

Any field of any behavior can be overridden per room; unspecified fields inherit the defaults.
See `classroom_monitor/config.py` for every knob and its documentation.

## Behavior classifiers and their guard rails

All classifiers feed a per-frame boolean into a `SustainedCondition` gate
(`behavior/sustain.py`). The gate opens only when the condition has held for
`min_duration_s` **and** at least `min_active_fraction` of frames in the run were active.
A single frame can never open it.

| Behavior | Per-frame cue | Default gate | Notes |
|---|---|---|---|
| **fighting** | weighted sum of: motion spike (0.3), another person within reach (0.2), *relative* velocity between the two (0.3), sustained arm raise (0.2); frame active at ≥ 0.6 | 3 s, 60 % of a 4 s window | Motion+proximity alone (0.5) never crosses the threshold, so two students walking together score nothing; relative velocity is a *difference*, so shared motion cancels. Event carries both track ids. |
| **sleeping** | nose below shoulder line by ≥ 15 % of torso height, and mean motion over the last second below 0.3 box-widths/s | 20 s, 85 % of 25 s | Frames without a pose neither count for nor against. |
| **out_of_seat** | foot point outside every seat zone (+ margin) | 8 s, 80 % of 10 s | Only fires for people first seen *inside* a zone (`require_home_zone`), so the teacher and visitors never trigger. |
| **talking** | repeated head turning (yaw change vs. the previous frame and a 2 s baseline) with a neighbour within reach | 5 s, 50 % of 8 s | **On, but deliberately weak.** Hard-capped at LOW confidence (`max_confidence: 0.35`), so it can never reach HIGH and never carries a clip on its own. `require_neighbour: false` degrades it to plain head turning; only useful for single-person testing. |

### On talking detection

Talking detection is enabled, and you should read its alerts as "a staff member might want to
glance at this", never as a finding. At 540p from a ceiling-corner camera a student's mouth is
a handful of pixels: lip motion is not resolvable, and head turns are ambiguous between talking
to a neighbour, looking at the board, and dropping a pencil. The classifier therefore uses the
one cue that survives at that resolution, repeated head turning toward a nearby person, and the
result is clamped so it can never outrank a fighting or sleeping alert in the queue.

To make it actually meaningful, add a per-room audio channel; most IP cameras carry an audio
track on the same RTSP stream. The right next step is a room-level sound-pressure or
voice-activity signal (no speech recognition, no transcription) fused with the head-turn cue,
i.e. "there is voice activity in the room AND this track is turned toward a neighbour". That
keeps the privacy footprint at "was there voice activity", never "what was said". Until then,
`enabled: false` per room is a legitimate choice for rooms where the noise is not worth it.

## Alert lifecycle and the mandatory human step

```
NEW ──acknowledge(staff_id)──► ACKNOWLEDGED ──review(staff_id, confirm|dismiss)──► CONFIRMED | DISMISSED
```

* `review()` refuses an alert that is still NEW (`ReviewError`, HTTP 409).
* Both steps require a non-empty staff id; the reviewer id and timestamp are stored on the alert.
* `ClipFlag.ACTIONABLE` is set in exactly one place: a human `CONFIRM`. Nothing else can set it.
* The system has no automated actions of any kind. It writes alerts to a queue; that's it.

Debounce (`config/system.yaml → debounce`): repeat events for the same room+behavior+tracks
merge into the open alert (bumping `occurrences`); after review the same key is suppressed for
`per_track_cooldown_s`; the same behavior anywhere in the room is suppressed for
`per_room_cooldown_s`, which also absorbs tracker id churn during a fight.

## Clips and retention

* Frames are kept only in a per-room ring buffer of `pre_roll_s` seconds. **There is no
  continuous recording.** A clip is written only when an alert is created, covering
  `pre_roll_s` before and `post_roll_s` after the trigger.
* Retention is decided by the human outcome, enforced by a background sweep:

  | Alert state | Clip deleted after |
  |---|---|
  | never reviewed (NEW or ACKNOWLEDGED) | `unreviewed_hours` (48 h) |
  | DISMISSED | `dismissed_hours` (24 h) |
  | CONFIRMED (actionable) | `actionable_days` (90 d) |

  Deleted clips are removed from disk and marked in the database; the alert record itself is kept.

## Dashboard API

| Method | Path | |
|---|---|---|
| GET | `/` | staff UI |
| GET | `/api/rooms` | status per room (online, fps, people, open alerts, seat zones) |
| GET | `/api/rooms/{id}/preview` | latest annotated frame |
| GET | `/api/alerts/queue` | alerts awaiting a human, NEW first, oldest first |
| GET | `/api/alerts?status=&room_id=&behavior=&since=&until=` | searchable incident log |
| POST | `/api/alerts/{id}/ack` | header `X-Staff-Id` required |
| POST | `/api/alerts/{id}/review` | body `{outcome: confirm|dismiss, notes}`, header `X-Staff-Id` |
| GET | `/api/clips/{id}` | clip index (frames + offsets) or mp4 link |
| WS | `/ws` | `alert.new` / `alert.updated` push |

**Before any real deployment** put the dashboard behind the school's SSO and derive
`X-Staff-Id` from the verified identity at the reverse proxy; the current text box exists only
so the workflow can be exercised.

## Not done yet / open decisions

* Authentication and role-based access on the dashboard (see above).
* Real-footage tuning. All thresholds were set against the synthetic scene and first
  principles; expect to retune `fighting.motion_spike`, `relative_motion` and
  `sleeping.head_below_shoulders_ratio` per camera height once you have a few labelled clips.
  `scripts/replay.py` is the tool for that.
* A seat-zone calibration UI (draw boxes on a preview). Zones are hand-edited YAML for now.
* Audio channel for talking (see above).
* Postgres store and multi-process deployment once the room count outgrows one machine.
