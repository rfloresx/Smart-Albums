"""Job lifecycle management — create, run (subprocess), cancel, poll.

Jobs are pipeline executions spawned as child processes via the ``smart-albums``
CLI. The webgui writes a temporary config JSON and invokes the CLI with
``--progress-json`` so that structured progress events can be parsed from
stdout in real time. Regular log lines are captured separately for display
in the UI.

The subprocess model provides:
- Crash isolation (a failed pipeline never takes down the GUI server)
- Memory isolation (GPU models are loaded/freed with the subprocess)
- Identical behavior to CLI usage (no subtle runtime differences)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import signal
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from webgui.models import JobRecord, JobStatus
from webgui.services import prompts as prompts_service
from webgui.state import state

logger = logging.getLogger(__name__)

PROGRESS_PREFIX = "@@PROGRESS@@"
_CONFIG_DIR = Path("/tmp/smart-albums-jobs")
_PROMPT_DIR = _CONFIG_DIR / "prompts"
_LOGS_DIR = Path("/tmp/smart-albums-jobs/logs")

# StreamReader buffer limit for the subprocess's stdout, well above the
# asyncio default of 64 KiB so a large (but not absurd) progress JSON line
# or traceback doesn't overrun it.
_STREAM_LINE_LIMIT = 8 * 1024 * 1024

# How long to wait for a killed subprocess to exit before giving up on it.
_PROCESS_KILL_TIMEOUT_SECONDS = 10

# Maximum number of pipeline subprocesses that can run concurrently.
# Additional jobs wait in the queue until a slot becomes available.
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "4"))
_job_semaphore: asyncio.Semaphore | None = None

# Strong references to in-flight job tasks, so they aren't garbage-collected
# mid-run (see create_job()).
_background_tasks: set[asyncio.Task[None]] = set()

# Job IDs that cancel_job() has been asked to cancel. Checked by _run_job at
# two points: right after acquiring the concurrency semaphore (so a pending
# job never spawns a subprocess at all), and in the finally block (so the
# terminal status/error reflects "cancelled", not whatever exit code the
# killed subprocess happened to produce).
_cancelled_jobs: set[str] = set()


def _get_job_semaphore() -> asyncio.Semaphore:
    """Lazily initialize the job semaphore on the running event loop."""
    global _job_semaphore
    if _job_semaphore is None:
        _job_semaphore = asyncio.Semaphore(MAX_CONCURRENT_JOBS)
    return _job_semaphore


def _write_config_file(path: Path, data: dict[str, Any]) -> None:
    """Write a config file with restricted permissions (owner-only read/write).

    This prevents other local users from reading secrets (API keys) that are
    embedded in per-job config files.
    """
    import os
    import stat

    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)  # 0600


def _pipeline_has_export_context(steps: list[Any]) -> bool:
    """Check if a pipeline definition contains an export.context stage."""
    for step in steps:
        if isinstance(step, (list, tuple)) and len(step) >= 1:
            stage_name = step[0]
            if stage_name == "export.context":
                return True
    return False


def _inject_export_directory(cli_config: dict[str, Any], job_id: str) -> None:
    """Inject the managed export directory into pipeline_settings for export.context.

    Only modifies the config if the pipeline includes an export.context stage.
    For user pipelines, checks the embedded steps. For built-in pipelines,
    checks if the alias already exists in pipeline_settings.
    """
    export_dir = str(Path(state.config.exports_dir) / job_id)
    settings = cli_config.get("pipeline_settings", {})

    # User pipelines: check steps definition
    pipeline_steps = cli_config.get("pipeline")
    if pipeline_steps and _pipeline_has_export_context(pipeline_steps):
        export_cfg = settings.setdefault("export.context", {})
        export_cfg["export_directory"] = export_dir
        settings["export.context"] = export_cfg
        cli_config["pipeline_settings"] = settings
        return

    # Built-in pipelines: only inject if the alias already exists in settings
    # (user explicitly configured it) — otherwise the pipeline doesn't have the stage
    if "export.context" in settings:
        settings["export.context"]["export_directory"] = export_dir
        cli_config["pipeline_settings"] = settings


def _build_cli_config(pipeline_settings: dict[str, Any], job_id: str) -> dict[str, Any]:
    """Build the config.json dict the CLI expects from form settings.

    Resolves provider configuration from the in-memory state (which reflects
    the DB-persisted settings), maps ``prompt_file`` references to absolute
    paths in ``prompts_dir``, and handles inline custom prompts by writing
    them to a temp file.
    """
    settings = dict(pipeline_settings)
    custom_prompt = settings.pop("_custom_prompt", None)
    log_level = settings.pop("_log_level", "INFO")

    # Resolve prompt_file path
    score_cfg = settings.get("score", {})
    if custom_prompt:
        _PROMPT_DIR.mkdir(parents=True, exist_ok=True)
        custom_path = _PROMPT_DIR / f"{job_id}.md"
        custom_path.write_text(custom_prompt, encoding="utf-8")
        score_cfg["prompt_file"] = str(custom_path)
        settings["score"] = score_cfg
    elif score_cfg.get("prompt_file"):
        pf = score_cfg["prompt_file"]
        try:
            # Route through the same traversal guard the Prompts page uses
            # (services.prompts._safe_path) instead of a bare `Path(...) /
            # pf` join. An absolute path (e.g. "/etc/passwd") or a "../"
            # value previously escaped prompts_dir; worse, if the escaped
            # path didn't exist, the *raw* value was still forwarded to the
            # CLI unchanged, which read it and sent its contents to the
            # configured LLM (WG-14). Any invalid reference now fails the
            # job up front with a clear error instead.
            user_path = prompts_service.resolve_prompt_path(pf)
        except prompts_service.PromptError as exc:
            raise ValueError(f"Invalid prompt_file reference {pf!r}: {exc}") from exc
        if not user_path.exists():
            raise ValueError(f"Prompt file not found: {pf!r}")
        score_cfg["prompt_file"] = str(user_path)
        settings["score"] = score_cfg

    cfg: dict[str, Any] = {
        "log_level": log_level,
        "pipeline_settings": settings,
    }

    # Resolve each provider into CLI config format
    for protocol_name, prov_cfg in state.config.providers.items():
        if not prov_cfg.selected:
            continue
        cfg[protocol_name] = prov_cfg.selected
        for param_key, param_value in prov_cfg.params.items():
            cfg[f"{protocol_name}.{prov_cfg.selected}.{param_key}"] = param_value

    return cfg


def _update_progress(record: JobRecord, event: dict[str, Any]) -> None:
    """Update the job record's progress field from a parsed progress event."""
    event_type = event.get("type")

    if event_type == "pipeline_start":
        record.progress = {
            "total_stages": event.get("total_stages", 0),
            "stage_index": 0,
            "stage_name": None,
            "stage_completed": 0,
            "stage_total": 0,
        }
    elif event_type == "stage_enter":
        if record.progress is None:
            record.progress = {}
        record.progress["stage_index"] = event.get("stage_index", 0)
        record.progress["total_stages"] = event.get("total_stages", 0)
        record.progress["stage_name"] = event.get("stage_name")
        record.progress["stage_completed"] = 0
        record.progress["stage_total"] = 0
    elif event_type == "stage_start":
        if record.progress is None:
            record.progress = {}
        record.progress["stage_name"] = event.get("stage")
        record.progress["stage_total"] = event.get("total", 0)
        record.progress["stage_completed"] = 0
    elif event_type == "advance":
        if record.progress is None:
            record.progress = {}
        record.progress["stage_completed"] = event.get("completed", 0)
        record.progress["stage_total"] = event.get("total", 0)
    elif event_type == "stage_finish":
        if record.progress is None:
            record.progress = {}
        record.progress["stage_completed"] = event.get("completed", 0)
        record.progress["stage_total"] = event.get("total", 0)


async def create_job(pipeline_name: str, pipeline_settings: dict[str, Any]) -> JobRecord:
    """Create a job record and kick off the pipeline subprocess.

    Args:
        pipeline_name: Registered pipeline name (e.g. ``"best-of-year"``)
            or a user pipeline reference (``"user:<id>"``).
        pipeline_settings: Alias-keyed config overrides from the form.

    Returns:
        The newly created :class:`JobRecord` in *pending* status.
    """
    job_id = str(uuid.uuid4())[:8]
    record = JobRecord(
        id=job_id,
        pipeline=pipeline_name,
        pipeline_settings=pipeline_settings,
    )
    await state.db.create_job(record)
    task = asyncio.create_task(_run_job(job_id))
    # Keep a strong reference so the task isn't garbage-collected mid-run
    # (asyncio only holds a weak reference internally) and so an unexpected
    # exception is at least logged via the done-callback instead of being
    # silently dropped.
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return record


async def cancel_job(job_id: str) -> None:
    """Cancel a job, whether it's queued (pending) or already running.

    For a pending job (still waiting on the concurrency semaphore, no
    subprocess started yet), this sets a flag that ``_run_job`` checks
    right after acquiring its semaphore slot, so the job never spawns a
    subprocess at all. For a running job, the subprocess (and its whole
    process group, so grandchildren are included) is sent SIGTERM; if it
    hasn't exited after a grace period, SIGKILL follows.

    The cancellation reason is recorded via ``_cancelled_jobs`` rather than
    writing the DB record directly, so ``_run_job``'s own ``finally`` block
    (which also writes a terminal status/error) doesn't race with this
    function and overwrite "Cancelled by user" with a generic
    "Process exited with code -15".

    Raises:
        ValueError: If the job doesn't exist or has already finished.
    """
    record = await state.db.get_job(job_id)
    if record is None:
        raise ValueError(f"Job {job_id} not found")
    if record.status not in (JobStatus.pending, JobStatus.running):
        raise ValueError("Job is not pending or running")

    _cancelled_jobs.add(job_id)

    proc = state.running_processes.get(job_id)
    if proc is not None and proc.returncode is None:
        try:
            # Kill the whole process group (see start_new_session=True in
            # _run_job), not just the direct child, so a pipeline that
            # spawned its own subprocesses doesn't leave them running.
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            proc.terminate()

        async def _escalate() -> None:
            try:
                await asyncio.wait_for(proc.wait(), timeout=_PROCESS_KILL_TIMEOUT_SECONDS)
            except asyncio.TimeoutError:
                logger.warning("Job %s: did not exit after SIGTERM; sending SIGKILL", job_id)
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.kill()

        task = asyncio.create_task(_escalate())
        _background_tasks.add(task)
        task.add_done_callback(_background_tasks.discard)
    else:
        # Still pending (no subprocess yet) — _run_job will see the flag
        # and finalize the record itself once it wakes up from the
        # semaphore. Nothing further to do here.
        pass


async def delete_job(job_id: str) -> None:
    """Delete a completed or failed job record.

    Also removes any exported data from the managed export directory.

    Raises:
        ValueError: If the job is still running or doesn't exist.
    """
    record = await state.db.get_job(job_id)
    if record is None:
        raise ValueError(f"Job {job_id} not found")
    if record.status == JobStatus.running:
        raise ValueError("Cannot delete a running job")
    await state.db.delete_job(job_id)

    # Clean up export artifacts
    export_dir = Path(state.config.exports_dir) / job_id
    if export_dir.is_dir():
        shutil.rmtree(export_dir, ignore_errors=True)

    # Clean up log file
    log_file = get_log_file(job_id)
    log_file.unlink(missing_ok=True)


def get_live_job(job_id: str) -> JobRecord | None:
    """Return the in-memory job record if the job is currently running."""
    return state.running_jobs.get(job_id)


async def list_jobs(*, limit: int | None = None) -> list[JobRecord]:
    """List all jobs, overlaying live progress for running ones."""
    jobs = await state.db.list_jobs(limit=limit)
    return [state.running_jobs.get(j.id, j) for j in jobs]


async def list_job_summaries(*, limit: int | None = None) -> list[JobRecord]:
    """List jobs without ``log_output``, overlaying live progress for running ones.

    Intended for frequent polling (see pages/jobs.py's 3-second timer).
    Running jobs are overlaid with the in-memory record from
    ``state.running_jobs``, which does carry ``log_output`` for the tail
    currently held in memory — only the DB-backed historical jobs skip it.
    """
    jobs = await state.db.list_job_summaries(limit=limit)
    return [state.running_jobs.get(j.id, j) for j in jobs]


def get_export_dir(job_id: str) -> Path:
    """Return the managed export directory path for a job."""
    return Path(state.config.exports_dir) / job_id


def has_export_output(job_id: str) -> bool:
    """Check if a job produced downloadable export data."""
    export_dir = get_export_dir(job_id)
    try:
        return export_dir.is_dir() and any(export_dir.iterdir())
    except OSError:
        return False


def get_log_file(job_id: str) -> Path:
    """Return the path to the full log file for a job."""
    return _LOGS_DIR / f"{job_id}.log"


def has_log_file(job_id: str) -> bool:
    """Check if a full log file exists for a job."""
    return get_log_file(job_id).is_file()


async def _run_job(job_id: str) -> None:
    """Execute the pipeline as a subprocess (background task).

    Acquires a slot from the job concurrency semaphore before spawning the
    subprocess. If all slots are occupied, the job waits in 'pending' status
    until one becomes available.
    """
    semaphore = _get_job_semaphore()

    record = await state.db.get_job(job_id)
    if record is None:
        logger.error("Job %s not found in database", job_id)
        return

    # Wait for a concurrency slot before starting the subprocess
    async with semaphore:
        # Re-fetch in case the job was cancelled while waiting
        record = await state.db.get_job(job_id)
        if record is None or record.status != JobStatus.pending:
            return

        if job_id in _cancelled_jobs:
            # Cancelled while still queued — finalize without ever
            # spawning a subprocess. Previously a pending job had no
            # cancel path at all (cancel_job only accepted "running").
            _cancelled_jobs.discard(job_id)
            record.status = JobStatus.failed
            record.error = "Cancelled by user"
            record.completed_at = datetime.now(timezone.utc)
            await state.db.update_job(record)
            return

        record.status = JobStatus.running
        record.started_at = datetime.now(timezone.utc)
        await state.db.update_job(record)
        state.running_jobs[job_id] = record

        # Everything from here on (config building, directory setup, and
        # the subprocess itself) runs inside try/finally. Previously, the
        # config-building and mkdir calls ran *before* this point, so a
        # failure there (disk full, a permission error, a bad nested
        # setting) left the job's status stuck at "running" forever, both
        # in the DB and in ``state.running_jobs`` — nothing ever reached the
        # code that sets a terminal status. Moving that work inside the try
        # ensures every failure path is finalized the same way.
        proc: asyncio.subprocess.Process | None = None
        config_file = _CONFIG_DIR / f"{job_id}.json"
        try:
            cli_config = _build_cli_config(record.pipeline_settings, job_id)

            _CONFIG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
            _LOGS_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
            log_file = get_log_file(job_id)
            # For built-in pipelines, inject export dir if applicable
            _inject_export_directory(cli_config, job_id)
            _write_config_file(config_file, cli_config)

            log_level = record.pipeline_settings.get("_log_level", "INFO")

            # Determine CLI invocation based on pipeline type
            from webgui.services.user_pipelines import is_user_pipeline, get_user_pipeline_id

            smart_albums_bin = str(Path(sys.executable).parent / "smart-albums")

            if is_user_pipeline(record.pipeline):
                # User-defined pipeline: use the 'run' command with pipeline
                # definition embedded in the config JSON
                user_pipeline_id = get_user_pipeline_id(record.pipeline)
                user_pipeline = await state.db.get_user_pipeline(user_pipeline_id)
                if user_pipeline is None:
                    raise ValueError(
                        f"User pipeline '{user_pipeline_id}' not found"
                    )
                # Inject the pipeline steps into the config
                cli_config["pipeline"] = user_pipeline["steps"]
                _inject_export_directory(cli_config, job_id)
                _write_config_file(config_file, cli_config)

                proc = await asyncio.create_subprocess_exec(
                    smart_albums_bin,
                    "run",
                    "--config", str(config_file),
                    "--log-level", log_level,
                    "--progress-json",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    limit=_STREAM_LINE_LIMIT,
                    # Own process group so cancel_job() can kill the whole
                    # tree (the CLI and anything it spawns), not just this
                    # direct child.
                    start_new_session=True,
                )
            else:
                # Built-in pipeline: use the pipeline name as command
                proc = await asyncio.create_subprocess_exec(
                    smart_albums_bin,
                    record.pipeline,
                    "--config", str(config_file),
                    "--log-level", log_level,
                    "--progress-json",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    limit=_STREAM_LINE_LIMIT,
                    start_new_session=True,
                )

            state.running_processes[job_id] = proc

            output_lines: list[str] = []
            _log_update_counter = 0
            assert proc.stdout is not None
            # buffering=1 (line-buffered) rather than flushing explicitly
            # after every line — flush() is itself a blocking syscall, and
            # doing it on every single stdout line from the subprocess
            # (which can be thousands for a large pipeline run) adds up to
            # real time spent blocking the event loop (WG-21). Line
            # buffering keeps the log file reasonably current on disk
            # without a syscall per line.
            with open(log_file, "w", encoding="utf-8", buffering=1) as lf:
                # readline() (which `async for` on StreamReader uses) can
                # still raise LimitOverrunError/IncompleteReadError for a
                # single line longer than `limit` even with `limit` raised
                # generously above. Read defensively so one oversized line
                # (e.g. a huge traceback) fails that read instead of
                # silently leaving the subprocess's pipe undrained, which
                # would otherwise block the child forever once its stdout
                # buffer fills up.
                while True:
                    try:
                        line = await proc.stdout.readline()
                    except (asyncio.LimitOverrunError, ValueError) as exc:
                        logger.warning(
                            "Job %s: oversized output line, truncating: %s",
                            job_id, exc,
                        )
                        # Drain and discard the rest of the oversized line
                        # so the stream can resync on the next line.
                        try:
                            await proc.stdout.read(_STREAM_LINE_LIMIT)
                        except Exception:
                            pass
                        continue
                    if not line:
                        break
                    decoded = line.decode("utf-8", errors="replace")

                    if decoded.startswith(PROGRESS_PREFIX):
                        try:
                            event = json.loads(decoded[len(PROGRESS_PREFIX):])
                            _update_progress(record, event)
                        except (json.JSONDecodeError, KeyError) as exc:
                            logger.debug("Failed to parse progress event: %s", exc)
                        continue

                    # Write full log to file (no truncation)
                    lf.write(decoded)
                    lf.flush()

                    # Keep tail in memory for UI display
                    output_lines.append(decoded)
                    if len(output_lines) > 2000:
                        output_lines = output_lines[-2000:]
                    _log_update_counter += 1
                    if _log_update_counter % 20 == 0:
                        record.log_output = "".join(output_lines)

            # Final join to capture any remaining lines
            record.log_output = "".join(output_lines)

            await proc.wait()

            if job_id in _cancelled_jobs:
                # cancel_job() already sent the signal; make sure the
                # final status says "cancelled" rather than reporting
                # whatever exit code the killed process happened to exit
                # with (e.g. "Process exited with code -15"), and rather
                # than reporting "completed" if the CLI caught SIGTERM and
                # exited 0 on its own.
                record.status = JobStatus.failed
                record.error = "Cancelled by user"
            elif proc.returncode == 0:
                record.status = JobStatus.completed
            else:
                record.status = JobStatus.failed
                record.error = f"Process exited with code {proc.returncode}"

        except Exception as exc:
            if job_id in _cancelled_jobs:
                record.status = JobStatus.failed
                record.error = "Cancelled by user"
            else:
                record.status = JobStatus.failed
                record.error = str(exc)
                logger.exception("Job %s failed", job_id)
        finally:
            # If a subprocess was started but is still alive at this point
            # (an exception was raised while streaming output, or the job
            # was cancelled), make sure it doesn't outlive this task. Left
            # unattended, an orphaned child with a full stdout pipe and no
            # reader blocks forever. Kills the whole process group (see
            # start_new_session=True above), not just the direct child.
            if proc is not None and proc.returncode is None:
                try:
                    os.killpg(proc.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=_PROCESS_KILL_TIMEOUT_SECONDS)
                except asyncio.TimeoutError:
                    logger.warning("Job %s: process did not exit after terminate(); killing", job_id)
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        proc.kill()
                    await proc.wait()
                except ProcessLookupError:
                    pass

            _cancelled_jobs.discard(job_id)
            record.completed_at = datetime.now(timezone.utc)
            record.progress = None
            await state.db.update_job(record)
            state.running_processes.pop(job_id, None)
            state.running_jobs.pop(job_id, None)
            config_file.unlink(missing_ok=True)
            custom_prompt = _PROMPT_DIR / f"{job_id}.md"
            custom_prompt.unlink(missing_ok=True)
