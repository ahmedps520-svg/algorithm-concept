"""Clip verifier: a local vision-language model (Ollama, default qwen3-vl:8b) looks at a
contact sheet of frames around the alert trigger and gives an advisory verdict.

The fast pipeline is tuned for recall; this stage adds precision on exactly the cases rules
are bad at (fight vs horseplay, sleeping vs reading). The verdict is attached to the alert for
the human reviewer. It never changes the alert's status."""
from __future__ import annotations

import base64
import json
import logging
import queue
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Callable

import numpy as np

from ..alerts.models import Alert
from ..alerts.store import AlertStore
from ..clips import imgcodec
from ..config import VerifierConfig

log = logging.getLogger(__name__)

Frames = list[tuple[float, np.ndarray]]
Transport = Callable[[str, dict | None, float], dict]   # (url, json body or None for GET, timeout) -> json

BEHAVIOR_GUIDE = {
    "fighting": ("physical aggression: hitting, kicking, shoving with force, grappling, one person clearly "
                 "trying to hurt or dominate another",
                 "play-fighting, horseplay, high-fives, hugging, dancing, helping someone up, reaching past someone"),
    "sleeping": ("head down on the desk or arms, or eyes closed and slumped, staying that way",
                 "reading, writing, looking at a phone or paper on the desk, resting chin on hand while watching"),
    "out_of_seat": ("a student standing or walking away from their desk without an obvious task",
                    "leaning or reaching from the seat, briefly standing to hand something over, the teacher moving around"),
    "talking": ("a student turned to a neighbour and talking with them",
                "answering the teacher, reading aloud, group work; without audio this is genuinely hard to tell"),
}

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["likely_real", "likely_false_alarm", "unclear"]},
        "what_is_happening": {"type": "string"},
        "confidence": {"type": "number"},
        "reason": {"type": "string"},
        "people_involved": {"type": "integer"},
    },
    "required": ["verdict", "what_is_happening", "confidence", "reason"],
}


def build_contact_sheet(frames: Frames, trigger_ts: float, offsets: list[float], tile_width: int, columns: int
                        ) -> tuple[np.ndarray, list[float]]:
    """Pick the frame nearest each offset and tile them left-to-right, top-to-bottom.
    Returns the sheet and the actual offsets used."""
    if not frames:
        raise ValueError("no frames")
    picks, used = [], []
    for off in offsets:
        target = trigger_ts + off
        ts, img = min(frames, key=lambda f: abs(f[0] - target))
        picks.append(imgcodec.resize_nearest(img, tile_width))
        used.append(round(ts - trigger_ts, 1))
    th = max(p.shape[0] for p in picks)
    rows = (len(picks) + columns - 1) // columns
    sheet = np.zeros((rows * th, columns * tile_width, 3), dtype=np.uint8)
    for i, p in enumerate(picks):
        r, c = divmod(i, columns)
        sheet[r * th:r * th + p.shape[0], c * tile_width:c * tile_width + p.shape[1]] = p
        # thin white separator so the model sees distinct frames
        sheet[r * th:(r + 1) * th, c * tile_width:c * tile_width + 2] = 255
        sheet[r * th:r * th + 2, c * tile_width:(c + 1) * tile_width] = 255
    return sheet, used


def build_prompt(alert: Alert, offsets: list[float], columns: int) -> str:
    real, false = BEHAVIOR_GUIDE.get(alert.behavior, ("the alerted behavior", "anything else"))
    n = len(offsets)
    return (
        "You are helping a school staff member review an automated alert from a fixed classroom camera. "
        f"The image is a contact sheet of {n} frames from one clip, in reading order (left to right, then next row, "
        f"{columns} per row). Their times relative to the moment the alert fired are: "
        + ", ".join(f"{o:+.1f}s" for o in offsets) + ".\n\n"
        f"The automated system flagged: {alert.behavior.replace('_', ' ').upper()} involving {len(alert.track_ids)} tracked "
        f"person(s), sustained for {alert.sustained_s:.0f} seconds.\n"
        f"Count it as REAL only if you see: {real}.\n"
        f"Count it as a FALSE ALARM if it is more like: {false}.\n"
        "Be conservative: if the frames are ambiguous, answer unclear with low confidence rather than guessing. "
        "Do not identify anyone. Describe only what is visible.\n\n"
        "Reply with JSON only: {\"verdict\": \"likely_real\" | \"likely_false_alarm\" | \"unclear\", "
        "\"what_is_happening\": one sentence, \"confidence\": 0 to 1, \"reason\": one sentence, \"people_involved\": integer}."
    )


def parse_verdict(text: str) -> dict:
    """Tolerant parse: JSON, fenced JSON, or JSON embedded in prose."""
    text = text.strip()
    candidates = [text]
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        candidates.insert(0, m.group(1))
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        candidates.append(m.group(0))
    for c in candidates:
        try:
            d = json.loads(c)
            if isinstance(d, dict) and "verdict" in d:
                break
        except json.JSONDecodeError:
            continue
    else:
        raise ValueError(f"no JSON verdict in model reply: {text[:200]!r}")
    verdict = str(d.get("verdict", "unclear")).strip().lower().replace(" ", "_")
    if verdict not in ("likely_real", "likely_false_alarm", "unclear"):
        verdict = "unclear"
    try:
        conf = float(d.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    if conf > 1.0:
        conf /= 100.0
    return {
        "verdict": verdict,
        "confidence": max(0.0, min(1.0, conf)),
        "summary": str(d.get("what_is_happening", ""))[:400],
        "reason": str(d.get("reason", ""))[:400],
        "people_involved": d.get("people_involved"),
    }


def _http_json(url: str, body: dict | None, timeout: float) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


class OllamaVerifier:
    def __init__(self, cfg: VerifierConfig, store: AlertStore, notify: Callable[[Alert], None] | None = None,
                 transport: Transport = _http_json, clock: Callable[[], float] = time.time) -> None:
        self.cfg = cfg
        self.store = store
        self.notify = notify
        self.transport = transport
        self.clock = clock
        self._q: queue.Queue[tuple[str, Frames, float]] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        self._thread = threading.Thread(target=self._worker, name="verifier", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._q.put(("", [], 0.0))

    def _worker(self) -> None:
        while not self._stop.is_set():
            alert_id, frames, trigger = self._q.get()
            if not alert_id:
                continue
            alert = self.store.get(alert_id)
            if alert is None:
                continue
            try:
                self.verify_now(alert, frames, trigger)
            except Exception:  # noqa: BLE001
                log.exception("verifier failed for alert %s", alert_id)

    def health(self) -> dict:
        try:
            tags = self.transport(f"{self.cfg.base_url}/api/tags", None, 5.0)
            names = [m.get("name", "") for m in tags.get("models", [])]
            have = any(n == self.cfg.model or n.split(":")[0] == self.cfg.model.split(":")[0] for n in names)
            return {"reachable": True, "model": self.cfg.model, "model_pulled": have, "models": names}
        except Exception as e:  # noqa: BLE001
            return {"reachable": False, "model": self.cfg.model, "error": str(e)}

    # ------------------------------------------------------------- work
    def enqueue(self, alert: Alert, frames: Frames) -> None:
        """Called when a clip is ready (frames still in memory). Non-blocking."""
        self._q.put((alert.id, frames, alert.first_ts))

    def verify_now(self, alert: Alert, frames: Frames, trigger_ts: float | None = None) -> Alert:
        trigger_ts = alert.first_ts if trigger_ts is None else trigger_ts
        sheet, offsets = build_contact_sheet(frames, trigger_ts, self.cfg.frame_offsets_s, self.cfg.tile_width, self.cfg.columns)
        data, _ext = imgcodec.encode(sheet, quality=85)
        body = {
            "model": self.cfg.model,
            "stream": False,
            "format": SCHEMA,
            # qwen3-vl reasons before answering. With thinking on, the reply budget is spent on
            # that and "content" comes back empty, so turn it off and read "thinking" as a
            # fallback for models or builds that ignore the flag.
            "think": False,
            "keep_alive": "10m",
            "options": {"temperature": self.cfg.temperature, "num_predict": self.cfg.max_tokens},
            "messages": [{"role": "user", "content": build_prompt(alert, offsets, self.cfg.columns),
                          "images": [base64.b64encode(data).decode()]}],
        }
        t0 = self.clock()
        try:
            try:
                resp = self.transport(f"{self.cfg.base_url}/api/chat", body, self.cfg.timeout_s)
            except urllib.error.HTTPError as e:          # older builds reject unknown fields
                if e.code != 400:
                    raise
                resp = self.transport(f"{self.cfg.base_url}/api/chat", {k: v for k, v in body.items() if k != "think"}, self.cfg.timeout_s)
            msg = resp.get("message", {}) if isinstance(resp, dict) else {}
            text = (msg.get("content") or "").strip() or (msg.get("thinking") or "").strip()
            if not text:
                raise ValueError(f"model returned no text (done_reason: {resp.get('done_reason', 'unknown')})")
            v = parse_verdict(text)
            alert.ai_verdict, alert.ai_confidence = v["verdict"], v["confidence"]
            alert.ai_summary, alert.ai_reason = v["summary"], v["reason"]
        except (urllib.error.URLError, TimeoutError, ValueError, OSError) as e:
            log.warning("verifier error for alert %s: %s", alert.id, e)
            alert.ai_verdict, alert.ai_confidence = "error", None
            alert.ai_summary, alert.ai_reason = "", f"verifier unavailable: {e}"[:400]
        alert.ai_model = self.cfg.model
        alert.ai_at = self.clock()
        # Re-read status fields in case the reviewer acted while the model was thinking.
        fresh = self.store.get(alert.id)
        if fresh is not None:
            for f in ("status", "acknowledged_by", "acknowledged_at", "reviewed_by", "reviewed_at", "review_notes", "clip_flag", "clip_id", "occurrences", "last_ts"):
                setattr(alert, f, getattr(fresh, f))
        self.store.upsert(alert)
        log.info("verifier: alert %s -> %s (%.2f) in %.1fs", alert.id, alert.ai_verdict, alert.ai_confidence or 0.0, self.clock() - t0)
        if self.notify:
            self.notify(alert)
        return alert
