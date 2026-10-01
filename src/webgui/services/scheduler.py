"""Schedule management and background scheduler loop.

Provides CRUD operations for cron schedules and an asyncio-based background
loop that checks every 30 seconds whether any enabled schedule is due to fire.
When a schedule's next_run_at has passed, the scheduler triggers a pipeline
job via the jobs service and updates the schedule record.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from webgui.models import ScheduleRecord
from webgui.services.cron import CronParseError, next_run_time, validate_cron
from webgui.services.template_vars import resolve_template_variables
from webgui.state import state

logger = logging.getLogger(__name__)

_scheduler_task: Optional[asyncio.Task[None]] = None
_CHECK_INTERVAL_SECONDS = 30

# Held while a maintenance operation (e.g. restoring a database backup) needs
# exclusive access to the DB connection. The scheduler loop skips firing
# schedules while this is set, so it never queries a connection that is
# mid-close/reconnect.
_maintenance_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------


async def create_schedule(
    name: str,
    pipeline: str,
    pipeline_settings: dict[str, Any],
    cron_expression: str,
    enabled: bool = True,
    template_variables: Optional[dict[str, str]] = None,
) -> ScheduleRecord:
    """Create a new cron schedule.

    Args:
        name: Human-friendly name for this schedule.
        pipeline: Registered pipeline name.
        pipeline_settings: Alias-keyed config overrides (same as job form).
        cron_expression: 5-field cron expression.
        enabled: Whether the schedule is active.
        template_variables: Variable definitions (e.g. {"YEAR": "2025"})
            carried over from the preset this schedule was configured from,
            if any. Resolved into pipeline_settings at fire/run-now time —
            see ``_resolve_template_variables``.

    Returns:
        The newly created ScheduleRecord.

    Raises:
        ValueError: If the cron expression is invalid or name is empty.
    """
    if not name.strip():
        raise ValueError("Schedule name cannot be empty")

    error = validate_cron(cron_expression)
    if error:
        raise ValueError(f"Invalid cron expression: {error}")

    now = datetime.now(timezone.utc)
    next_run = next_run_time(cron_expression, after=now) if enabled else None

    record = ScheduleRecord(
        id=str(uuid.uuid4())[:8],
        name=name.strip(),
        pipeline=pipeline,
        pipeline_settings=pipeline_settings,
        template_variables=template_variables or {},
        cron_expression=cron_expression,
        enabled=enabled,
        created_at=now,
        updated_at=now,
        next_run_at=next_run,
    )
    await state.db.create_schedule(record)
    logger.info(
        "Created schedule %s (%s) — next run: %s",
        record.id,
        record.name,
        record.next_run_at,
    )
    return record


async def update_schedule(
    schedule_id: str,
    *,
    name: Optional[str] = None,
    pipeline: Optional[str] = None,
    pipeline_settings: Optional[dict[str, Any]] = None,
    cron_expression: Optional[str] = None,
    enabled: Optional[bool] = None,
    template_variables: Optional[dict[str, str]] = None,
) -> ScheduleRecord:
    """Update an existing schedule.

    Only non-None arguments are applied.

    Raises:
        ValueError: If the schedule doesn't exist or cron expression is invalid.
    """
    record = await state.db.get_schedule(schedule_id)
    if record is None:
        raise ValueError(f"Schedule {schedule_id} not found")

    if name is not None:
        if not name.strip():
            raise ValueError("Schedule name cannot be empty")
        record.name = name.strip()

    if pipeline is not None:
        record.pipeline = pipeline

    if pipeline_settings is not None:
        record.pipeline_settings = pipeline_settings

    if template_variables is not None:
        record.template_variables = template_variables

    if cron_expression is not None:
        error = validate_cron(cron_expression)
        if error:
            raise ValueError(f"Invalid cron expression: {error}")
        record.cron_expression = cron_expression

    if enabled is not None:
        record.enabled = enabled

    # Recalculate next_run_at
    if record.enabled:
        record.next_run_at = next_run_time(record.cron_expression, after=datetime.now(timezone.utc))
    else:
        record.next_run_at = None

    record.updated_at = datetime.now(timezone.utc)
    await state.db.update_schedule(record)
    logger.info("Updated schedule %s (%s)", record.id, record.name)
    return record


async def delete_schedule(schedule_id: str) -> None:
    """Delete a schedule by ID.

    Raises:
        ValueError: If the schedule doesn't exist.
    """
    deleted = await state.db.delete_schedule(schedule_id)
    if not deleted:
        raise ValueError(f"Schedule {schedule_id} not found")
    logger.info("Deleted schedule %s", schedule_id)


async def toggle_schedule(schedule_id: str) -> ScheduleRecord:
    """Toggle a schedule's enabled state.

    Raises:
        ValueError: If the schedule doesn't exist.
    """
    record = await state.db.get_schedule(schedule_id)
    if record is None:
        raise ValueError(f"Schedule {schedule_id} not found")
    return await update_schedule(schedule_id, enabled=not record.enabled)


async def list_schedules() -> list[ScheduleRecord]:
    """List all schedules."""
    return await state.db.list_schedules()


async def get_schedule(schedule_id: str) -> ScheduleRecord | None:
    """Get a single schedule by ID."""
    return await state.db.get_schedule(schedule_id)


# ---------------------------------------------------------------------------
# Background scheduler loop
# ---------------------------------------------------------------------------


async def _scheduler_loop() -> None:
    """Background loop that checks for due schedules and triggers jobs.

    Runs every _CHECK_INTERVAL_SECONDS. For each enabled schedule whose
    next_run_at has passed, it creates a job and advances next_run_at.
    """
    logger.info("Scheduler loop started (check interval: %ds)", _CHECK_INTERVAL_SECONDS)

    while True:
        try:
            await asyncio.sleep(_CHECK_INTERVAL_SECONDS)
            if _maintenance_lock.locked():
                logger.debug("Skipping schedule check — maintenance in progress")
                continue
            await _check_and_fire()
        except asyncio.CancelledError:
            logger.info("Scheduler loop cancelled")
            break
        except Exception:
            logger.exception("Error in scheduler loop")
            # Keep running despite transient errors


async def _advance_or_disable(schedule: ScheduleRecord, now: datetime) -> ScheduleRecord:
    """Compute the schedule's next run, disabling it if that's not possible.

    Pulls a fresh copy of the schedule row first and writes back only the
    scheduler-owned columns (``last_run_at``, ``next_run_at``,
    ``last_job_id``, ``enabled``, ``updated_at``) via
    ``update_schedule_fields`` instead of overwriting every column from the
    stale in-memory snapshot ``_check_and_fire`` loaded at the top of its
    pass. Without this, a user edit (name, pipeline_settings, cron
    expression) or a pause made in the UI while this schedule's job was
    running would be silently discarded by this write, and a paused
    schedule could even be re-enabled by it.

    If ``next_run_time`` still can't find a match (a combinatorially
    impossible expression that validate_cron didn't catch, or one that
    changed by direct DB edit), the schedule is disabled with its
    next_run_at cleared instead of being left pointing at a past time —
    which would otherwise cause it to be picked up and "fired" again on
    every subsequent 30s tick forever.
    """
    fresh = await state.db.get_schedule(schedule.id)
    if fresh is None:
        # Deleted concurrently; nothing to update.
        return schedule

    try:
        fresh.next_run_at = next_run_time(schedule.cron_expression, after=now)
    except CronParseError:
        logger.error(
            "Schedule %s (%s) has a cron expression with no reachable next "
            "run (%r) — disabling it instead of leaving it in a fire loop.",
            schedule.id, schedule.name, schedule.cron_expression,
        )
        fresh.enabled = False
        fresh.next_run_at = None

    fresh.last_run_at = schedule.last_run_at
    fresh.last_job_id = schedule.last_job_id
    fresh.updated_at = now
    await state.db.update_schedule_fields(fresh)
    return fresh


async def _check_and_fire() -> None:
    """Check all enabled schedules and fire any that are due.

    Schedules whose last fired job is still pending/running are skipped
    (not fired again) until that job reaches a terminal state, so a run
    that takes longer than the schedule's interval doesn't stack duplicate
    jobs behind the concurrency semaphore.
    """
    from webgui.models import JobStatus
    from webgui.services import jobs as jobs_service

    now = datetime.now(timezone.utc)
    schedules = await state.db.list_enabled_schedules()

    for schedule in schedules:
        try:
            if schedule.next_run_at is None:
                # Compute next_run_at if missing
                await _advance_or_disable(schedule, now)
                continue

            if schedule.next_run_at > now:
                continue

            if schedule.last_job_id:
                last_job = await state.db.get_job(schedule.last_job_id)
                if last_job is not None and last_job.status in (
                    JobStatus.pending, JobStatus.running,
                ):
                    logger.info(
                        "Schedule %s (%s) is due but job %s is still %s — "
                        "skipping this tick.",
                        schedule.id, schedule.name, last_job.id, last_job.status.value,
                    )
                    continue

            # Schedule is due — fire it
            logger.info(
                "Firing schedule %s (%s) — was due at %s",
                schedule.id,
                schedule.name,
                schedule.next_run_at,
            )

            try:
                settings_to_run = resolve_template_variables(
                    schedule.pipeline_settings, schedule.template_variables
                )
                record = await jobs_service.create_job(
                    schedule.pipeline, settings_to_run
                )
                schedule.last_run_at = now
                schedule.last_job_id = record.id
                logger.info(
                    "Schedule %s fired job %s",
                    schedule.id,
                    record.id,
                )
            except Exception:
                logger.exception("Failed to fire schedule %s", schedule.id)

            # Advance next_run_at regardless of success/failure
            await _advance_or_disable(schedule, now)

        except Exception:
            logger.exception(
                "Error processing schedule %s (%s), skipping",
                schedule.id,
                schedule.name,
            )


def start_scheduler() -> None:
    """Start the background scheduler task.

    Safe to call multiple times — only one loop will run.
    """
    global _scheduler_task
    if _scheduler_task is not None and not _scheduler_task.done():
        logger.debug("Scheduler already running")
        return

    _scheduler_task = asyncio.create_task(_scheduler_loop())
    logger.info("Scheduler background task started")


def maintenance_lock() -> asyncio.Lock:
    """Return the shared lock used to pause scheduling during maintenance.

    Callers that need exclusive access to the database connection (e.g.
    restoring a backup) should hold this lock for the duration of the
    operation so the scheduler loop does not fire jobs against a connection
    that is being closed/reconnected concurrently.
    """
    return _maintenance_lock


def stop_scheduler() -> None:
    """Stop the background scheduler task."""
    global _scheduler_task
    if _scheduler_task is not None and not _scheduler_task.done():
        _scheduler_task.cancel()
        logger.info("Scheduler background task stop requested")
    _scheduler_task = None
