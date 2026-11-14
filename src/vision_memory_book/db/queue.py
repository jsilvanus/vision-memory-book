"""Persistent, resumable SQLite job queue with atomic claiming and stale lock recovery."""

import json
import sqlite3
from typing import Any


class JobQueue:
    """SQLite-backed job queue for download and vision indexing workers."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        data = dict(row)
        raw_payload = data.get("payload_json") or "{}"
        try:
            data["payload"] = json.loads(raw_payload)
        except json.JSONDecodeError:
            data["payload"] = {}
        return data

    def enqueue(
        self,
        job_type: str,
        photo_id: int,
        payload: dict[str, Any] | None = None,
        priority: int = 0,
        max_attempts: int = 3,
    ) -> int:
        """Enqueue a new pending job and return its ID."""
        payload_json = json.dumps(payload or {})
        cur = self.conn.execute(
            """
            INSERT INTO jobs (job_type, photo_id, payload_json, priority, max_attempts)
            VALUES (?, ?, ?, ?, ?)
            """,
            (job_type, photo_id, payload_json, priority, max_attempts),
        )
        self.conn.commit()
        return int(cur.lastrowid)

    def reclaim_stale(self, stale_after_seconds: int = 300) -> int:
        """Recover running jobs whose heartbeat is older than stale_after_seconds."""
        threshold_days = float(stale_after_seconds) / 86400.0
        # Jobs that already exhausted max_attempts become failed
        cur_fail = self.conn.execute(
            """
            UPDATE jobs
            SET status = 'failed',
                locked_by = NULL,
                locked_at = NULL,
                heartbeat_at = NULL,
                last_error = COALESCE(last_error, 'stale worker timeout (max attempts reached)'),
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE status = 'running'
              AND attempts >= max_attempts
              AND (julianday('now') - julianday(COALESCE(heartbeat_at, locked_at))) > ?
            """,
            (threshold_days,),
        )
        # Jobs with remaining attempts return to pending
        cur_requeue = self.conn.execute(
            """
            UPDATE jobs
            SET status = 'pending',
                locked_by = NULL,
                locked_at = NULL,
                heartbeat_at = NULL,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE status = 'running'
              AND attempts < max_attempts
              AND (julianday('now') - julianday(COALESCE(heartbeat_at, locked_at))) > ?
            """,
            (threshold_days,),
        )
        self.conn.commit()
        return cur_fail.rowcount + cur_requeue.rowcount

    def claim_next(
        self,
        worker_id: str,
        job_types: list[str] | None = None,
        stale_after_seconds: int = 300,
    ) -> dict[str, Any] | None:
        """Atomically claim the highest-priority pending job, reclaiming stale locks first."""
        self.reclaim_stale(stale_after_seconds=stale_after_seconds)

        params: list[Any] = [worker_id]
        type_filter = ""
        if job_types:
            placeholders = ", ".join("?" for _ in job_types)
            type_filter = f"AND job_type IN ({placeholders})"
            params.extend(job_types)

        row = self.conn.execute(
            f"""
            UPDATE jobs
            SET status = 'running',
                locked_by = ?,
                locked_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                heartbeat_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                attempts = attempts + 1,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = (
                SELECT id FROM jobs
                WHERE status = 'pending'
                  AND attempts < max_attempts
                  {type_filter}
                ORDER BY priority DESC, id ASC
                LIMIT 1
            )
            RETURNING *
            """,
            params,
        ).fetchone()
        self.conn.commit()

        if row is None:
            return None
        return self._row_to_dict(row)

    def heartbeat(self, job_id: int, worker_id: str | None = None) -> bool:
        """Update heartbeat timestamp for a running job."""
        if worker_id is not None:
            cur = self.conn.execute(
                """
                UPDATE jobs
                SET heartbeat_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = ? AND status = 'running' AND locked_by = ?
                """,
                (job_id, worker_id),
            )
        else:
            cur = self.conn.execute(
                """
                UPDATE jobs
                SET heartbeat_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE id = ? AND status = 'running'
                """,
                (job_id,),
            )
        self.conn.commit()
        return cur.rowcount > 0

    def complete(self, job_id: int) -> bool:
        """Mark a job as done and clear lock metadata."""
        cur = self.conn.execute(
            """
            UPDATE jobs
            SET status = 'done',
                locked_by = NULL,
                locked_at = NULL,
                heartbeat_at = NULL,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?
            """,
            (job_id,),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def fail(self, job_id: int, error: str) -> str:
        """Record a job failure; requeue as pending if attempts < max_attempts, else mark failed."""
        row = self.conn.execute(
            "SELECT attempts, max_attempts FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"Job {job_id} not found")

        new_status = "pending" if row["attempts"] < row["max_attempts"] else "failed"
        self.conn.execute(
            """
            UPDATE jobs
            SET status = ?,
                last_error = ?,
                locked_by = NULL,
                locked_at = NULL,
                heartbeat_at = NULL,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = ?
            """,
            (new_status, error, job_id),
        )
        self.conn.commit()
        return new_status

    def pause_all(self, job_type: str | None = None) -> int:
        """Pause all pending jobs (optionally filtered by job_type)."""
        if job_type:
            cur = self.conn.execute(
                """
                UPDATE jobs
                SET status = 'paused',
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE status = 'pending' AND job_type = ?
                """,
                (job_type,),
            )
        else:
            cur = self.conn.execute(
                """
                UPDATE jobs
                SET status = 'paused',
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE status = 'pending'
                """
            )
        self.conn.commit()
        return cur.rowcount

    def resume_all(self, job_type: str | None = None) -> int:
        """Resume all paused jobs back to pending (optionally filtered by job_type)."""
        if job_type:
            cur = self.conn.execute(
                """
                UPDATE jobs
                SET status = 'pending',
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE status = 'paused' AND job_type = ?
                """,
                (job_type,),
            )
        else:
            cur = self.conn.execute(
                """
                UPDATE jobs
                SET status = 'pending',
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE status = 'paused'
                """
            )
        self.conn.commit()
        return cur.rowcount

    def retry_failed(self, job_type: str | None = None) -> int:
        """Reset failed jobs to pending with attempts=0."""
        if job_type:
            cur = self.conn.execute(
                """
                UPDATE jobs
                SET status = 'pending',
                    attempts = 0,
                    last_error = NULL,
                    locked_by = NULL,
                    locked_at = NULL,
                    heartbeat_at = NULL,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE status = 'failed' AND job_type = ?
                """,
                (job_type,),
            )
        else:
            cur = self.conn.execute(
                """
                UPDATE jobs
                SET status = 'pending',
                    attempts = 0,
                    last_error = NULL,
                    locked_by = NULL,
                    locked_at = NULL,
                    heartbeat_at = NULL,
                    updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
                WHERE status = 'failed'
                """
            )
        self.conn.commit()
        return cur.rowcount

    def stats(self) -> dict[str, Any]:
        """Return job counts grouped by job_type and status, plus recent errors."""
        rows = self.conn.execute(
            """
            SELECT job_type, status, COUNT(*) AS cnt
            FROM jobs
            GROUP BY job_type, status
            """
        ).fetchall()

        by_type: dict[str, dict[str, int]] = {}
        by_status: dict[str, int] = {}
        total = 0

        for row in rows:
            jtype = row["job_type"]
            status = row["status"]
            cnt = int(row["cnt"])
            by_type.setdefault(jtype, {})[status] = cnt
            by_status[status] = by_status.get(status, 0) + cnt
            total += cnt

        error_rows = self.conn.execute(
            """
            SELECT id, job_type, photo_id, attempts, last_error, updated_at
            FROM jobs
            WHERE status = 'failed' OR last_error IS NOT NULL
            ORDER BY updated_at DESC, id DESC
            LIMIT 10
            """
        ).fetchall()

        return {
            "total": total,
            "by_status": by_status,
            "by_type": by_type,
            "recent_errors": [dict(r) for r in error_rows],
        }
