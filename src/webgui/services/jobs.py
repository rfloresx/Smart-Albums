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
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from webgui.models import JobRecord, JobStatus
from webgui.state import state

logger = logging.getLogger(__name__)

PROGRESS_PREFIX = "@@PROGRESS@@"
_CONFIG_DIR = Path("/tmp/smart-albums-jobs")
_PROMPT_DIR = _CONFIG_DIR / "prompts"

# Maximum number of pipeline subprocesses that can run concurrently.
# Additional jobs wait in the queue until a slot becomes available.
MAX_CONCURRENT_JOBS = int(os.environ.get("MAX_CONCURRENT_JOBS", "4"))
_job_semaphore: asyncio.Semaphore | None = None


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
        user_path = Path(state.config.prompts_dir) / pf
        if user_path.exists():
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
    asyncio.create_task(_run_job(job_id))
    return record


async def cancel_job(job_id: str) -> None:
    """Terminate a running job's subprocess.

    Raises:
        ValueError: If the job is not currently running.
    """
    record = await state.db.get_job(job_id)
    if record is None:
        raise ValueError(f"Job {job_id} not found")
    if record.status != JobStatus.running:
        raise ValueError("Job is not running")
    proc = state.running_processes.get(job_id)
    if proc:
        proc.terminate()
    record.status = JobStatus.failed
    record.error = "Cancelled by user"
    record.completed_at = datetime.now(timezone.utc)
    await state.db.update_job(record)
    state.running_jobs.pop(job_id, None)
    state.running_processes.pop(job_id, None)


async def delete_job(job_id: str) -> None:
    """Delete a completed or failed job record.

    Raises:
        ValueError: If the job is still running or doesn't exist.
    """
    record = await state.db.get_job(job_id)
    if record is None:
        raise ValueError(f"Job {job_id} not found")
    if record.status == JobStatus.running:
        raise ValueError("Cannot delete a running job")
    await state.db.delete_job(job_id)


def get_live_job(job_id: str) -> JobRecord | None:
    """Return the in-memory job record if the job is currently running."""
    return state.running_jobs.get(job_id)


async def list_jobs() -> list[JobRecord]:
    """List all jobs, overlaying live progress for running ones."""
    jobs = await state.db.list_jobs()
    return [state.running_jobs.get(j.id, j) for j in jobs]


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

        record.status = JobStatus.running
        record.started_at = datetime.now(timezone.utc)
        await state.db.update_job(record)
        state.running_jobs[job_id] = record

        cli_config = _build_cli_config(record.pipeline_settings, job_id)

        _CONFIG_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
        config_file = _CONFIG_DIR / f"{job_id}.json"
        _write_config_file(config_file, cli_config)

        log_level = record.pipeline_settings.get("_log_level", "INFO")

        # Determine CLI invocation based on pipeline type
        from webgui.services.user_pipelines import is_user_pipeline, get_user_pipeline_id

        try:
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
                _write_config_file(config_file, cli_config)

                proc = await asyncio.create_subprocess_exec(
                    smart_albums_bin,
                    "run",
                    "--config", str(config_file),
                    "--log-level", log_level,
                    "--progress-json",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
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
                )

            state.running_processes[job_id] = proc

            output_lines: list[str] = []
            _log_update_counter = 0
            assert proc.stdout is not None
            async for line in proc.stdout:
                decoded = line.decode("utf-8", errors="replace")

                if decoded.startswith(PROGRESS_PREFIX):
                    try:
                        event = json.loads(decoded[len(PROGRESS_PREFIX):])
                        _update_progress(record, event)
                    except (json.JSONDecodeError, KeyError) as exc:
                        logger.debug("Failed to parse progress event: %s", exc)
                    continue

                output_lines.append(decoded)
                if len(output_lines) > 2000:
                    output_lines = output_lines[-2000:]
                _log_update_counter += 1
                if _log_update_counter % 20 == 0:
                    record.log_output = "".join(output_lines)

            # Final join to capture any remaining lines
            record.log_output = "".join(output_lines)

            await proc.wait()

            if proc.returncode == 0:
                record.status = JobStatus.completed
            else:
                record.status = JobStatus.failed
                record.error = f"Process exited with code {proc.returncode}"

        except Exception as exc:
            record.status = JobStatus.failed
            record.error = str(exc)
            logger.exception("Job %s failed", job_id)
        finally:
            record.completed_at = datetime.now(timezone.utc)
            record.progress = None
            await state.db.update_job(record)
            state.running_processes.pop(job_id, None)
            state.running_jobs.pop(job_id, None)
            config_file.unlink(missing_ok=True)
            custom_prompt = _PROMPT_DIR / f"{job_id}.md"
            custom_prompt.unlink(missing_ok=True)
