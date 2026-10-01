"""Shared application state singleton.

Holds the global config, database connection, and in-memory tracking
for running jobs. Initialized once at startup and accessible from all
pages and services.
"""

from __future__ import annotations

import asyncio
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from webgui.config import AppConfig, AuthConfig, load_config, providers_from_dict, resolve_provider_paths
from webgui.database import Database
from webgui.models import JobRecord, JobStatus

PROVIDERS_SETTINGS_KEY = "providers"
AUTH_SETTINGS_KEY = "auth"

logger = logging.getLogger(__name__)


class AppState:
    """Global application state — initialized once, shared across pages."""

    def __init__(self) -> None:
        self.config: AppConfig = AppConfig()
        self.db: Database = Database("")
        self.pipeline_schemas: dict[str, Any] = {}
        self.running_jobs: dict[str, JobRecord] = {}
        self.running_processes: dict[str, asyncio.subprocess.Process] = {}

    async def init(self) -> None:
        """Load config, connect to DB, seed admin user if needed."""
        self.config = load_config()
        self.db = Database(self.config.db_path)
        await self.db.connect()
        await self.reload_providers()

        # Mark any jobs stuck in "running" or "pending" as failed (server
        # restart). Previously only "running" was handled; a job that was
        # queued behind the concurrency semaphore (status "pending") when
        # the process died had no in-memory task to resume it and no code
        # path ever revisited it, so it stayed "pending" forever — showing
        # up in the Jobs page indistinguishable from a job genuinely
        # waiting for a free slot (WG-20).
        jobs = await self.db.list_jobs()
        for job in jobs:
            if job.status in (JobStatus.running, JobStatus.pending):
                job.status = JobStatus.failed
                job.error = "Server restarted while job was running or queued"
                job.completed_at = datetime.now(timezone.utc)
                await self.db.update_job(job)

        # Seed admin user from env vars if auth enabled and no users exist
        if self.config.auth.enabled:
            count = await self.db.user_count()
            if count == 0:
                username = os.environ.get("ADMIN_USERNAME", "")
                password = os.environ.get("ADMIN_PASSWORD", "")
                if username and password:
                    from webgui.auth import hash_password

                    pw_hash = hash_password(password)
                    await self.db.create_user(str(uuid.uuid4()), username, pw_hash)
                    logger.info("Created admin user from env vars: %s", username)
                else:
                    logger.info(
                        "No users exist and ADMIN_USERNAME/ADMIN_PASSWORD not set. "
                        "Account setup will be required via the web UI."
                    )
    async def reload_providers(self) -> None:
        """Overlay DB-stored provider settings onto ``self.config.providers``.

        Provider configuration is persisted in the ``settings`` table (under
        the ``providers`` key) rather than in ``config.yaml``, so it can be
        changed at runtime without touching the config file. Any providers
        section in ``config.yaml`` only acts as the initial default on a
        fresh deployment, until an admin saves settings via the UI.
        """
        stored = await self.db.get_setting(PROVIDERS_SETTINGS_KEY)
        if stored:
            self.config.providers.update(providers_from_dict(stored))
            resolve_provider_paths(self.config)

        # Overlay auth settings from the DB if present
        auth_stored = await self.db.get_setting(AUTH_SETTINGS_KEY)
        if auth_stored:
            self.config.auth = AuthConfig(**auth_stored)

    async def shutdown(self) -> None:
        """Close database on shutdown."""
        await self.db.close()


# Module-level singleton
state = AppState()


async def init_state() -> None:
    """Initialize the global state. Called once at startup."""
    await state.init()


async def shutdown_state() -> None:
    """Shutdown the global state. Called on app shutdown."""
    await state.shutdown()
