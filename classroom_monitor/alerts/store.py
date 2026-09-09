"""SQLite persistence for alerts and clips. Single-file, thread-safe, no ORM."""
from __future__ import annotations

import json
import sqlite3
import threading
from pathlib import Path

from .models import Alert, AlertStatus, ClipFlag

_SCHEMA = """
CREATE TABLE IF NOT EXISTS alerts (
    id TEXT PRIMARY KEY,
    room_id TEXT NOT NULL,
    behavior TEXT NOT NULL,
    track_ids TEXT NOT NULL,
    score REAL NOT NULL,
    confidence TEXT NOT NULL,
    sustained_s REAL NOT NULL,
    first_ts REAL NOT NULL,
    last_ts REAL NOT NULL,
    evidence TEXT NOT NULL,
    status TEXT NOT NULL,
    occurrences INTEGER NOT NULL,
    acknowledged_by TEXT,
    acknowledged_at REAL,
    reviewed_by TEXT,
    reviewed_at REAL,
    review_notes TEXT NOT NULL DEFAULT '',
    clip_id TEXT,
    clip_flag TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS alerts_status ON alerts(status);
CREATE INDEX IF NOT EXISTS alerts_room_ts ON alerts(room_id, first_ts);
CREATE TABLE IF NOT EXISTS clips (
    id TEXT PRIMARY KEY,
    alert_id TEXT NOT NULL,
    room_id TEXT NOT NULL,
    path TEXT NOT NULL,
    created_at REAL NOT NULL,
    start_ts REAL NOT NULL,
    end_ts REAL NOT NULL,
    frames INTEGER NOT NULL,
    deleted_at REAL
);
"""


class AlertStore:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._db.executescript(_SCHEMA)

    # ---------------------------------------------------------------- alerts
    def upsert(self, a: Alert) -> None:
        with self._lock:
            self._db.execute(
                """INSERT OR REPLACE INTO alerts VALUES
                   (:id,:room_id,:behavior,:track_ids,:score,:confidence,:sustained_s,:first_ts,:last_ts,
                    :evidence,:status,:occurrences,:acknowledged_by,:acknowledged_at,:reviewed_by,:reviewed_at,
                    :review_notes,:clip_id,:clip_flag)""",
                {
                    **a.to_dict(),
                    "track_ids": json.dumps(a.track_ids),
                    "evidence": json.dumps(a.evidence, default=str),
                },
            )
            self._db.commit()

    def get(self, alert_id: str) -> Alert | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM alerts WHERE id=?", (alert_id,)).fetchone()
        return self._row_to_alert(row) if row else None

    def list(self, status: AlertStatus | None = None, room_id: str | None = None, behavior: str | None = None,
             since: float | None = None, until: float | None = None, limit: int = 200) -> list[Alert]:
        q, args = "SELECT * FROM alerts WHERE 1=1", []
        if status:
            q += " AND status=?"; args.append(status.value)
        if room_id:
            q += " AND room_id=?"; args.append(room_id)
        if behavior:
            q += " AND behavior=?"; args.append(behavior)
        if since is not None:
            q += " AND first_ts>=?"; args.append(since)
        if until is not None:
            q += " AND first_ts<=?"; args.append(until)
        q += " ORDER BY first_ts DESC LIMIT ?"; args.append(limit)
        with self._lock:
            rows = self._db.execute(q, args).fetchall()
        return [self._row_to_alert(r) for r in rows]

    @staticmethod
    def _row_to_alert(r: sqlite3.Row) -> Alert:
        d = dict(r)
        d["track_ids"] = json.loads(d["track_ids"])
        d["evidence"] = json.loads(d["evidence"])
        d["status"] = AlertStatus(d["status"])
        d["clip_flag"] = ClipFlag(d["clip_flag"])
        return Alert(**d)

    # ----------------------------------------------------------------- clips
    def add_clip(self, clip_id: str, alert_id: str, room_id: str, path: Path, created_at: float,
                 start_ts: float, end_ts: float, frames: int) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO clips VALUES (?,?,?,?,?,?,?,?,NULL)",
                (clip_id, alert_id, room_id, str(path), created_at, start_ts, end_ts, frames),
            )
            self._db.execute("UPDATE alerts SET clip_id=? WHERE id=?", (clip_id, alert_id))
            self._db.commit()

    def get_clip(self, clip_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()
        return dict(row) if row else None

    def live_clips_with_alerts(self) -> list[tuple[dict, Alert]]:
        """Every undeleted clip joined with its alert; used by the retention sweep."""
        with self._lock:
            rows = self._db.execute(
                "SELECT c.*, a.id AS a_id FROM clips c JOIN alerts a ON a.id=c.alert_id WHERE c.deleted_at IS NULL"
            ).fetchall()
        out = []
        for r in rows:
            clip = {k: r[k] for k in r.keys() if k != "a_id"}
            alert = self.get(r["a_id"])
            if alert:
                out.append((clip, alert))
        return out

    def mark_clip_deleted(self, clip_id: str, ts: float) -> None:
        with self._lock:
            self._db.execute("UPDATE clips SET deleted_at=? WHERE id=?", (ts, clip_id))
            self._db.commit()

    def close(self) -> None:
        with self._lock:
            self._db.close()
