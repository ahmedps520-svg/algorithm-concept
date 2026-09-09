"""Retention policy enforcement.

Policy (from ``RetentionConfig``) is decided by what a *human* did with the alert:
  * never reviewed (NEW / ACKNOWLEDGED)  -> delete ``unreviewed_hours`` after capture
  * DISMISSED                            -> delete ``dismissed_hours`` after review
  * CONFIRMED (clip ACTIONABLE)          -> delete ``actionable_days`` after review

The sweep runs on a background thread; ``sweep_once`` is exposed for tests and manual runs.
"""
from __future__ import annotations

import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Callable

from ..alerts.models import Alert, AlertStatus
from ..alerts.store import AlertStore
from ..config import RetentionConfig

log = logging.getLogger(__name__)


class RetentionEnforcer:
    def __init__(self, cfg: RetentionConfig, store: AlertStore, clock: Callable[[], float] = time.time) -> None:
        self.cfg = cfg
        self.store = store
        self.clock = clock
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def expiry_for(self, clip: dict, alert: Alert) -> float:
        if alert.status is AlertStatus.CONFIRMED:
            return (alert.reviewed_at or clip["created_at"]) + self.cfg.actionable_days * 86400
        if alert.status is AlertStatus.DISMISSED:
            return (alert.reviewed_at or clip["created_at"]) + self.cfg.dismissed_hours * 3600
        return clip["created_at"] + self.cfg.unreviewed_hours * 3600

    def sweep_once(self) -> list[str]:
        now = self.clock()
        deleted: list[str] = []
        for clip, alert in self.store.live_clips_with_alerts():
            if now < self.expiry_for(clip, alert):
                continue
            path = Path(clip["path"])
            try:
                if path.is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                elif path.exists():
                    path.unlink()
            except OSError:
                log.exception("failed to delete clip %s at %s", clip["id"], path)
                continue
            self.store.mark_clip_deleted(clip["id"], now)
            deleted.append(clip["id"])
            log.info("retention: deleted clip %s (alert %s, status %s)", clip["id"], alert.id, alert.status.value)
        return deleted

    def start(self) -> None:
        def loop() -> None:
            while not self._stop.wait(self.cfg.sweep_interval_s):
                try:
                    self.sweep_once()
                except Exception:  # noqa: BLE001
                    log.exception("retention sweep failed")

        self._thread = threading.Thread(target=loop, name="retention", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
