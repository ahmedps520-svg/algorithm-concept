# Classroom Monitor

Multi-classroom video monitoring backend with **human-reviewed** behavior alerts for school staff.

**Live demo of the staff dashboard using your own webcam:**
https://ahmedps520-svg.github.io/algorithm-concept/ — source in `docs/index.html`, deployed by
`.github/workflows/pages.yml`. Click "Add camera" and allow camera access; a seat zone is
created around you once you sit still. Person detection, tracking and pose run in the browser
(MoveNet MultiPose via TensorFlow.js), lip movement comes from MediaPipe Face Landmarker, and the
classifiers, alert debounce, review workflow and clip logic are the same as the Python backend.
Video never leaves the device. A "Simulated room" button adds a scripted room so every alert
type can be seen without acting it out.

**Teach it on real people.** The demo has a training panel: record 6-second labelled examples
of each behavior (and "normal") on your camera, press Train, and a classifier is fitted in the
browser to those examples (held-out accuracy shown) and runs alongside the rules. Only pose
feature vectors are stored, never video. Export the set and train the backend model with the
same features:

```bash
python scripts/train_behavior.py classroom-training-2026-09-09.json -o data/behavior_model.json
# then in config/system.yaml: default_thresholds.learned.enabled: true
```

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

Two layers run side by side and feed the same alert keys (so they merge, never duplicate):

* **Rules** (below): hand-set geometric cues behind sustained gates. Work with zero training.
* **Learned** (`behavior/learned.py`): 2-second windows of 12 per-frame cues (head drop,
  head height vs. this person's own upright baseline, tilt, yaw, torso motion, head motion,
  arm raise, lip movement, in-zone, nearest-neighbour gap, partner motion, pose present),
  summarised as mean/std/min/max into 48 features, then class-weighted softmax regression or a
  one-hidden-layer tanh network. Which shape is used is decided by stratified k-fold
  cross-validation over a grid of shapes, strengths and augmentation levels, not by hand. The
  model's prediction also sits behind a sustained gate, so a single confident frame never alerts.

Training happens where the data is. The demo records labelled 2-second windows from your own
camera (pose numbers only, never video), searches after every recording, and reports a
cross-validated accuracy. Export the examples and
`python scripts/train_behavior.py export.json --search -o data/behavior_model.json` fits the
same grid for the backend.

Clips are held in page memory, so the demo enforces its own budget: full frames are kept for
alerts still awaiting review plus the ten most recent, and older clips are reduced to a single
thumbnail. Without that, an hour of alerts grows past a gigabyte and the browser kills the tab.
The header shows heap use and how many clips are held.

**All training runs in a Web Worker**, never on the page thread: one model fit takes seconds, and
doing that on the page froze the tab, which the browser reports as unresponsive and eventually
kills. Overnight training uses the same worker for a much wider search: model shapes and strengths are sampled at random and each is scored
by *repeated* stratified cross-validation, which is what extra hours actually buy — a more
thorough search and a less noisy estimate of which model is best. The best five are refitted on
everything and kept as an ensemble whose members vote. Pick 1, 4, 8 or 12 hours, or run until
stopped; the best model so far is saved every couple of minutes, so stopping early or closing
the tab keeps the work. A screen wake lock is requested where the browser supports it.

More epochs on a fixed set stop helping quickly. Search and ensembling help a bit more. What
really moves accuracy is more recordings, from more people, at more camera angles and
distances. Neither layer is, or can be, "perfect": that is why every alert goes to a human.

All classifiers feed a per-frame boolean into a `SustainedCondition` gate
(`behavior/sustain.py`). The gate opens only when the condition has held for
`min_duration_s` **and** at least `min_active_fraction` of frames in the run were active.
A single frame can never open it.

| Behavior | Per-frame cue | Default gate | Notes |
|---|---|---|---|
| **fighting** | weighted sum of: motion spike (0.3), another person within reach (0.2), *relative* velocity between the two (0.3), sustained arm raise (0.2); frame active at ≥ 0.6 | 3 s, 60 % of a 4 s window | Motion+proximity alone (0.5) never crosses the threshold, so two students walking together score nothing; relative velocity is a *difference*, so shared motion cancels. Event carries both track ids. |
| **sleeping** | head actually DOWN: nose well below the shoulder line, or the face gone from view with the head at shoulder level; plus low torso and head motion | 20 s, 85 % of 25 s | Looking down at a desk or page is not sleeping, so head tilt and a drop against the person's own baseline are not triggers on their own. Both eyes plainly visible at normal head height vetoes it outright. |
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

## AI second opinion (local vision model)

A vision-language model is too slow to watch every frame, but it is much better than rules at
judging a clip: fight or horseplay, sleeping or reading. So it runs as a second stage on each
saved clip (`classroom_monitor/verify/ollama.py`): six frames around the trigger are tiled into
one contact sheet with their time offsets, sent to Ollama with the behavior's definition of
real vs. false alarm, and the model returns `likely_real | likely_false_alarm | unclear`, a
confidence, and a one-sentence reason. That lands on the alert (`ai_verdict`, `ai_confidence`,
`ai_summary`, `ai_reason`) and is shown to the reviewer as advisory. It never changes status.

```bash
ollama pull qwen3-vl:8b          # ~6 GB at Q4, ~1 s per clip on a 4060 Ti 16 GB
# config/system.yaml -> verifier.enabled: true
```

The demo page can call the same model from the browser: enable it in the "AI second opinion"
panel and start Ollama with `OLLAMA_ORIGINS=https://ahmedps520-svg.github.io` so it accepts
requests from that page. `POST /api/alerts/{id}/verify` re-runs it on the backend from the
saved clip. Qwen3-VL 32B or Gemma 3 27B also work (set `verifier.model`); they spill into
system RAM on a 16 GB card and take several seconds per clip, which is fine at alert rates.

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
| GET | `/api/verifier` | AI verifier status: reachable, model pulled |
| POST | `/api/alerts/{id}/verify` | run the AI second opinion on the saved clip now |
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
