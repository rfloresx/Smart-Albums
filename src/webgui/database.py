"""Async SQLite database layer for persistent storage.

Provides tables for:
- jobs: Pipeline execution history
- settings: Key-value store for global/user settings
- users: Authentication credentials
- sessions: Active login sessions
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from webgui.models import JobRecord, JobStatus, ScheduleRecord

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """\
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    pipeline TEXT NOT NULL,
    pipeline_settings TEXT NOT NULL DEFAULT '{}',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    error TEXT,
    log_output TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS presets (
    id TEXT PRIMARY KEY,
    pipeline TEXT NOT NULL,
    name TEXT NOT NULL,
    pipeline_settings TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS schedules (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    pipeline TEXT NOT NULL,
    pipeline_settings TEXT NOT NULL DEFAULT '{}',
    cron_expression TEXT NOT NULL,
    enabled INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_run_at TEXT,
    next_run_at TEXT,
    last_job_id TEXT
);

CREATE TABLE IF NOT EXISTS user_pipelines (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    steps TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_created_at ON jobs(created_at);
CREATE INDEX IF NOT EXISTS idx_presets_pipeline ON presets(pipeline);
CREATE INDEX IF NOT EXISTS idx_schedules_enabled ON schedules(enabled);
"""


class Database:
    """Async SQLite database wrapper for the web GUI."""

    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def connect(self) -> None:
        """Open the database connection and ensure schema exists."""
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.executescript(_SCHEMA_SQL)
        await self._db.commit()
        logger.info("Database connected: %s", self._db_path)

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    # ------------------------------------------------------------------
    # Users
    # ------------------------------------------------------------------

    async def get_user(self, username: str) -> Optional[dict[str, Any]]:
        """Fetch a user by username."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM users WHERE username=?", (username,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "username": row["username"],
            "password_hash": row["password_hash"],
            "created_at": row["created_at"],
        }

    async def create_user(self, user_id: str, username: str, password_hash: str) -> None:
        """Insert a new user."""
        assert self._db is not None
        await self._db.execute(
            "INSERT INTO users (id, username, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (user_id, username, password_hash, datetime.now(timezone.utc).isoformat()),
        )
        await self._db.commit()

    async def update_user_password(self, username: str, password_hash: str) -> None:
        """Update a user's password hash."""
        assert self._db is not None
        await self._db.execute(
            "UPDATE users SET password_hash=? WHERE username=?",
            (password_hash, username),
        )
        await self._db.commit()

    async def user_count(self) -> int:
        """Return the number of users in the database."""
        assert self._db is not None
        cursor = await self._db.execute("SELECT COUNT(*) as cnt FROM users")
        row = await cursor.fetchone()
        assert row is not None
        return int(row["cnt"])

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    async def create_job(self, record: JobRecord) -> None:
        """Insert a new job record."""
        assert self._db is not None
        await self._db.execute(
            """INSERT INTO jobs (id, pipeline, pipeline_settings, status,
               created_at, started_at, completed_at, error, log_output)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id,
                record.pipeline,
                json.dumps(record.pipeline_settings),
                record.status.value,
                record.created_at.isoformat(),
                record.started_at.isoformat() if record.started_at else None,
                record.completed_at.isoformat() if record.completed_at else None,
                record.error,
                record.log_output,
            ),
        )
        await self._db.commit()

    async def update_job(self, record: JobRecord) -> None:
        """Update an existing job record."""
        assert self._db is not None
        await self._db.execute(
            """UPDATE jobs SET status=?, started_at=?, completed_at=?,
               error=?, log_output=? WHERE id=?""",
            (
                record.status.value,
                record.started_at.isoformat() if record.started_at else None,
                record.completed_at.isoformat() if record.completed_at else None,
                record.error,
                record.log_output,
                record.id,
            ),
        )
        await self._db.commit()

    async def get_job(self, job_id: str) -> Optional[JobRecord]:
        """Fetch a single job by ID."""
        assert self._db is not None
        cursor = await self._db.execute("SELECT * FROM jobs WHERE id=?", (job_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_job(row)

    async def list_jobs(self) -> list[JobRecord]:
        """List all jobs, most recent first."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC"
        )
        rows = await cursor.fetchall()
        return [self._row_to_job(row) for row in rows]

    async def delete_job(self, job_id: str) -> bool:
        """Delete a job record. Returns True if deleted."""
        assert self._db is not None
        cursor = await self._db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        await self._db.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_job(row: aiosqlite.Row) -> JobRecord:
        """Convert a database row to a JobRecord."""
        return JobRecord(
            id=row["id"],
            pipeline=row["pipeline"],
            pipeline_settings=json.loads(row["pipeline_settings"]),
            status=JobStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            started_at=(
                datetime.fromisoformat(row["started_at"])
                if row["started_at"]
                else None
            ),
            completed_at=(
                datetime.fromisoformat(row["completed_at"])
                if row["completed_at"]
                else None
            ),
            error=row["error"],
            log_output=row["log_output"],
        )

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    async def get_setting(self, key: str) -> Optional[Any]:
        """Get a setting value by key. Returns None if not found."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT value FROM settings WHERE key=?", (key,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return json.loads(row["value"])

    async def set_setting(self, key: str, value: Any) -> None:
        """Set a setting value (upsert)."""
        assert self._db is not None
        await self._db.execute(
            """INSERT INTO settings (key, value, updated_at)
               VALUES (?, ?, ?)
               ON CONFLICT(key) DO UPDATE SET value=excluded.value,
               updated_at=excluded.updated_at""",
            (key, json.dumps(value), datetime.now(timezone.utc).isoformat()),
        )
        await self._db.commit()

    async def delete_setting(self, key: str) -> bool:
        """Delete a setting. Returns True if deleted."""
        assert self._db is not None
        cursor = await self._db.execute(
            "DELETE FROM settings WHERE key=?", (key,)
        )
        await self._db.commit()
        return cursor.rowcount > 0

    async def list_settings(self) -> dict[str, Any]:
        """Get all settings as a dictionary."""
        assert self._db is not None
        cursor = await self._db.execute("SELECT key, value FROM settings")
        rows = await cursor.fetchall()
        return {row["key"]: json.loads(row["value"]) for row in rows}

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------

    async def create_preset(
        self,
        preset_id: str,
        pipeline: str,
        name: str,
        pipeline_settings: dict[str, Any],
    ) -> None:
        """Insert a new preset."""
        assert self._db is not None
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """INSERT INTO presets (id, pipeline, name, pipeline_settings, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (preset_id, pipeline, name, json.dumps(pipeline_settings), now, now),
        )
        await self._db.commit()

    async def update_preset(
        self,
        preset_id: str,
        pipeline_settings: dict[str, Any],
    ) -> None:
        """Update an existing preset's settings."""
        assert self._db is not None
        await self._db.execute(
            "UPDATE presets SET pipeline_settings=?, updated_at=? WHERE id=?",
            (json.dumps(pipeline_settings), datetime.now(timezone.utc).isoformat(), preset_id),
        )
        await self._db.commit()

    async def list_presets(self, pipeline: str | None = None) -> list[dict[str, Any]]:
        """List presets, optionally filtered by pipeline name."""
        assert self._db is not None
        if pipeline:
            cursor = await self._db.execute(
                "SELECT * FROM presets WHERE pipeline=? ORDER BY name",
                (pipeline,),
            )
        else:
            cursor = await self._db.execute("SELECT * FROM presets ORDER BY pipeline, name")
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "pipeline": row["pipeline"],
                "name": row["name"],
                "pipeline_settings": json.loads(row["pipeline_settings"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    async def get_preset(self, preset_id: str) -> dict[str, Any] | None:
        """Fetch a single preset by ID."""
        assert self._db is not None
        cursor = await self._db.execute("SELECT * FROM presets WHERE id=?", (preset_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "pipeline": row["pipeline"],
            "name": row["name"],
            "pipeline_settings": json.loads(row["pipeline_settings"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    async def delete_preset(self, preset_id: str) -> bool:
        """Delete a preset. Returns True if deleted."""
        assert self._db is not None
        cursor = await self._db.execute("DELETE FROM presets WHERE id=?", (preset_id,))
        await self._db.commit()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Schedules
    # ------------------------------------------------------------------

    async def create_schedule(self, record: ScheduleRecord) -> None:
        """Insert a new schedule record."""
        assert self._db is not None
        await self._db.execute(
            """INSERT INTO schedules
               (id, name, pipeline, pipeline_settings, cron_expression,
                enabled, created_at, updated_at, last_run_at, next_run_at, last_job_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.id,
                record.name,
                record.pipeline,
                json.dumps(record.pipeline_settings),
                record.cron_expression,
                1 if record.enabled else 0,
                record.created_at.isoformat(),
                record.updated_at.isoformat(),
                record.last_run_at.isoformat() if record.last_run_at else None,
                record.next_run_at.isoformat() if record.next_run_at else None,
                record.last_job_id,
            ),
        )
        await self._db.commit()

    async def update_schedule(self, record: ScheduleRecord) -> None:
        """Update an existing schedule record."""
        assert self._db is not None
        await self._db.execute(
            """UPDATE schedules SET name=?, pipeline=?, pipeline_settings=?,
               cron_expression=?, enabled=?, updated_at=?,
               last_run_at=?, next_run_at=?, last_job_id=?
               WHERE id=?""",
            (
                record.name,
                record.pipeline,
                json.dumps(record.pipeline_settings),
                record.cron_expression,
                1 if record.enabled else 0,
                record.updated_at.isoformat(),
                record.last_run_at.isoformat() if record.last_run_at else None,
                record.next_run_at.isoformat() if record.next_run_at else None,
                record.last_job_id,
                record.id,
            ),
        )
        await self._db.commit()

    async def get_schedule(self, schedule_id: str) -> ScheduleRecord | None:
        """Fetch a single schedule by ID."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM schedules WHERE id=?", (schedule_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_schedule(row)

    async def list_schedules(self) -> list[ScheduleRecord]:
        """List all schedules, ordered by name."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM schedules ORDER BY name"
        )
        rows = await cursor.fetchall()
        return [self._row_to_schedule(row) for row in rows]

    async def list_enabled_schedules(self) -> list[ScheduleRecord]:
        """List only enabled schedules."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM schedules WHERE enabled=1 ORDER BY name"
        )
        rows = await cursor.fetchall()
        return [self._row_to_schedule(row) for row in rows]

    async def delete_schedule(self, schedule_id: str) -> bool:
        """Delete a schedule. Returns True if deleted."""
        assert self._db is not None
        cursor = await self._db.execute(
            "DELETE FROM schedules WHERE id=?", (schedule_id,)
        )
        await self._db.commit()
        return cursor.rowcount > 0

    @staticmethod
    def _row_to_schedule(row: aiosqlite.Row) -> ScheduleRecord:
        """Convert a database row to a ScheduleRecord."""
        return ScheduleRecord(
            id=row["id"],
            name=row["name"],
            pipeline=row["pipeline"],
            pipeline_settings=json.loads(row["pipeline_settings"]),
            cron_expression=row["cron_expression"],
            enabled=bool(row["enabled"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_run_at=(
                datetime.fromisoformat(row["last_run_at"])
                if row["last_run_at"]
                else None
            ),
            next_run_at=(
                datetime.fromisoformat(row["next_run_at"])
                if row["next_run_at"]
                else None
            ),
            last_job_id=row["last_job_id"],
        )

    # ------------------------------------------------------------------
    # User Pipelines
    # ------------------------------------------------------------------

    async def create_user_pipeline(
        self,
        pipeline_id: str,
        name: str,
        description: str,
        steps: list[Any],
    ) -> None:
        """Insert a new user-defined pipeline."""
        assert self._db is not None
        now = datetime.now(timezone.utc).isoformat()
        await self._db.execute(
            """INSERT INTO user_pipelines (id, name, description, steps, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (pipeline_id, name, description, json.dumps(steps), now, now),
        )
        await self._db.commit()

    async def update_user_pipeline(
        self,
        pipeline_id: str,
        name: str,
        description: str,
        steps: list[Any],
    ) -> None:
        """Update an existing user-defined pipeline."""
        assert self._db is not None
        await self._db.execute(
            """UPDATE user_pipelines SET name=?, description=?, steps=?, updated_at=?
               WHERE id=?""",
            (name, description, json.dumps(steps), datetime.now(timezone.utc).isoformat(), pipeline_id),
        )
        await self._db.commit()

    async def get_user_pipeline(self, pipeline_id: str) -> dict[str, Any] | None:
        """Fetch a single user pipeline by ID."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM user_pipelines WHERE id=?", (pipeline_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "id": row["id"],
            "name": row["name"],
            "description": row["description"],
            "steps": json.loads(row["steps"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    async def list_user_pipelines(self) -> list[dict[str, Any]]:
        """List all user-defined pipelines, ordered by name."""
        assert self._db is not None
        cursor = await self._db.execute(
            "SELECT * FROM user_pipelines ORDER BY name"
        )
        rows = await cursor.fetchall()
        return [
            {
                "id": row["id"],
                "name": row["name"],
                "description": row["description"],
                "steps": json.loads(row["steps"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    async def delete_user_pipeline(self, pipeline_id: str) -> bool:
        """Delete a user pipeline. Returns True if deleted."""
        assert self._db is not None
        cursor = await self._db.execute(
            "DELETE FROM user_pipelines WHERE id=?", (pipeline_id,)
        )
        await self._db.commit()
        return cursor.rowcount > 0
