"""Job history page — list, monitor, and manage pipeline runs.

Shows all jobs (newest first) with live progress for running ones. Each job
card shows status, timing, and expandable log output. Running jobs can be
cancelled; completed/failed jobs can be deleted. Completed jobs with export
data offer a download button.
"""

from __future__ import annotations

import logging
import zipfile
from datetime import datetime
from io import BytesIO
from pathlib import Path

from nicegui import app, ui
from starlette.responses import Response

from webgui.components.navbar import build_navbar
from webgui.models import JobStatus
from webgui.services import jobs as jobs_service

logger = logging.getLogger(__name__)

# Maximum total size of export directory allowed for on-the-fly ZIP (500 MB)
_MAX_EXPORT_ZIP_BYTES = 500 * 1024 * 1024


def _status_color(status: JobStatus) -> str:
    return {
        JobStatus.pending: "grey",
        JobStatus.running: "blue",
        JobStatus.completed: "positive",
        JobStatus.failed: "negative",
    }.get(status, "grey")


def _status_icon(status: JobStatus) -> str:
    return {
        JobStatus.pending: "hourglass_empty",
        JobStatus.running: "sync",
        JobStatus.completed: "check_circle",
        JobStatus.failed: "error",
    }.get(status, "help")


def _fmt_time(dt: datetime | None) -> str:
    if dt is None:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _build_header() -> None:
    build_navbar()


# ---------------------------------------------------------------------------
# Export download endpoint (FastAPI)
# ---------------------------------------------------------------------------


def _create_export_zip(export_dir: Path) -> bytes:
    """Create a ZIP archive from the export directory contents."""
    # Check total size first
    total_size = sum(f.stat().st_size for f in export_dir.rglob("*") if f.is_file())
    if total_size > _MAX_EXPORT_ZIP_BYTES:
        raise ValueError(
            f"Export too large for download ({total_size // (1024 * 1024)} MB "
            f"exceeds {_MAX_EXPORT_ZIP_BYTES // (1024 * 1024)} MB limit)"
        )

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(export_dir.rglob("*")):
            if file_path.is_file():
                arcname = str(file_path.relative_to(export_dir))
                zf.write(file_path, arcname)
    return buffer.getvalue()


@app.get("/api/jobs/{job_id}/download")
async def download_export(job_id: str) -> Response:
    """Download the exported data for a completed job as a ZIP archive."""
    from webgui.state import state

    record = await state.db.get_job(job_id)
    if record is None:
        return Response(content="Job not found", status_code=404)
    if record.status != JobStatus.completed:
        return Response(content="Job has not completed yet", status_code=400)

    export_dir = jobs_service.get_export_dir(job_id)
    if not export_dir.is_dir():
        return Response(
            content="Export data not available (pipeline may not include export.context)",
            status_code=404,
        )

    # If directory has no files
    if not any(export_dir.rglob("*")):
        return Response(content="No files to download", status_code=404)

    # Check if a pre-built ZIP exists in the export directory root
    existing_zips = list(export_dir.glob("*.zip"))
    if existing_zips:
        # Serve the first ZIP found directly
        zip_path = existing_zips[0]
        return Response(
            content=zip_path.read_bytes(),
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{zip_path.name}"'
            },
        )

    # Create ZIP on-the-fly
    try:
        zip_bytes = _create_export_zip(export_dir)
    except ValueError as exc:
        return Response(content=str(exc), status_code=413)

    filename = f"job-{job_id}-export.zip"
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/api/jobs/{job_id}/logs")
async def download_logs(job_id: str) -> Response:
    """Download the full log file for a job."""
    from webgui.state import state

    record = await state.db.get_job(job_id)
    if record is None:
        return Response(content="Job not found", status_code=404)

    log_file = jobs_service.get_log_file(job_id)
    if not log_file.is_file():
        # Fall back to the in-memory log_output stored in the DB
        if record.log_output:
            return Response(
                content=record.log_output,
                media_type="text/plain",
                headers={
                    "Content-Disposition": f'attachment; filename="job-{job_id}.log"'
                },
            )
        return Response(content="No log data available for this job", status_code=404)

    return Response(
        content=log_file.read_bytes(),
        media_type="text/plain",
        headers={
            "Content-Disposition": f'attachment; filename="job-{job_id}.log"'
        },
    )


def _build_job_card(job, *, on_refresh) -> dict:
    """Build a single job card and return references to updatable elements.

    Returns a dict with keys for elements that may need live updates:
    - 'status_icon', 'status_badge', 'progress_container', 'timing_row'
    """
    refs: dict = {"job_id": job.id, "status": job.status}

    with ui.card().classes("w-full") as card:
        refs["card"] = card
        with ui.row().classes("w-full items-center justify-between"):
            with ui.row().classes("items-center gap-2"):
                refs["status_icon"] = ui.icon(_status_icon(job.status)).props(
                    f"color={_status_color(job.status)}"
                )
                ui.label(f"{job.pipeline}").classes("text-subtitle1")
                refs["status_badge"] = ui.badge(job.status.value).props(
                    f"color={_status_color(job.status)}"
                )
            ui.label(f"#{job.id}").classes("text-caption text-grey")

        # Timing info
        with ui.row().classes("gap-4 text-caption text-grey") as timing_row:
            refs["timing_row"] = timing_row
            refs["created_label"] = ui.label(f"Created: {_fmt_time(job.created_at)}")
            refs["started_label"] = ui.label(
                f"Started: {_fmt_time(job.started_at)}"
            )
            refs["started_label"].set_visibility(job.started_at is not None)
            refs["finished_label"] = ui.label(
                f"Finished: {_fmt_time(job.completed_at)}"
            )
            refs["finished_label"].set_visibility(job.completed_at is not None)

        # Progress section (only content changes on update)
        with ui.column().classes("w-full gap-1") as progress_container:
            refs["progress_container"] = progress_container
            _render_progress(progress_container, job)

        # Error display
        refs["error_label"] = ui.label(f"Error: {job.error}").classes(
            "text-caption text-negative q-mt-sm"
        )
        refs["error_label"].set_visibility(bool(job.error))

        # Log output (expandable)
        if job.log_output:
            with ui.expansion("Log output").classes("w-full q-mt-sm").props("dense"):
                ui.code(job.log_output[-4000:]).classes(
                    "w-full text-xs"
                ).style("max-height: 300px; overflow-y: auto;")

        # Actions
        with ui.row().classes("w-full justify-end gap-2 q-mt-sm") as actions_row:
            refs["actions_row"] = actions_row
            _render_actions(actions_row, job, on_refresh=on_refresh)

    return refs


def _render_progress(container, job) -> None:
    """Render progress content inside the given container."""
    container.clear()
    if job.status != JobStatus.running or not job.progress:
        return

    with container:
        stage_name = job.progress.get("stage_name") or "Initializing…"
        completed = job.progress.get("stage_completed", 0)
        total = job.progress.get("stage_total", 0)
        stage_idx = job.progress.get("stage_index", 0)
        total_stages = job.progress.get("total_stages", 0)

        ui.label(
            f"Stage {stage_idx + 1}/{total_stages}: {stage_name}"
        ).classes("text-caption q-mt-sm")
        if total > 0:
            ui.linear_progress(
                value=completed / total, show_value=False
            ).props("color=blue rounded")
            ui.label(f"{completed}/{total}").classes("text-caption text-grey")
        else:
            ui.spinner(size="sm")


def _render_actions(container, job, *, on_refresh) -> None:
    """Render action buttons inside the given container."""
    container.clear()
    with container:
        if job.status == JobStatus.running:
            async def do_cancel(jid=job.id) -> None:
                try:
                    await jobs_service.cancel_job(jid)
                    ui.notify("Job cancelled", type="warning")
                except ValueError as e:
                    ui.notify(str(e), type="negative")
                await on_refresh()

            ui.button(
                "Cancel", icon="stop", on_click=do_cancel
            ).props("flat color=negative size=sm")
        elif job.status in (JobStatus.completed, JobStatus.failed):
            # Download button for completed jobs with export data
            if job.status == JobStatus.completed and jobs_service.has_export_output(job.id):
                ui.button(
                    "Download",
                    icon="download",
                    on_click=lambda jid=job.id: ui.download(f"/api/jobs/{jid}/download"),
                ).props("flat color=primary size=sm")

            # Download full log file
            if jobs_service.has_log_file(job.id) or job.log_output:
                ui.button(
                    "Logs",
                    icon="article",
                    on_click=lambda jid=job.id: ui.download(f"/api/jobs/{jid}/logs"),
                ).props("flat color=secondary size=sm")

            async def do_delete(jid=job.id) -> None:
                try:
                    await jobs_service.delete_job(jid)
                    ui.notify("Job deleted", type="info")
                except ValueError as e:
                    ui.notify(str(e), type="negative")
                await on_refresh()

            ui.button(
                "Delete", icon="delete", on_click=do_delete
            ).props("flat color=grey size=sm")


@ui.page("/jobs")
async def jobs_page() -> None:
    """Job history and monitoring page."""
    await ui.context.client.connected()
    ui.dark_mode(True)
    _build_header()

    container = ui.column().classes("w-full max-w-4xl mx-auto q-pa-md gap-3")
    # Track card refs by job ID for surgical updates
    card_refs: dict[int, dict] = {}

    async def full_refresh() -> None:
        """Full rebuild of the job list (used on status changes, delete, etc.)."""
        container.clear()
        card_refs.clear()
        jobs = await jobs_service.list_jobs()
        with container:
            if not jobs:
                ui.label("No jobs yet — start one from the New Job page.").classes(
                    "text-caption text-grey"
                )
                return
            for job in jobs:
                refs = _build_job_card(job, on_refresh=full_refresh)
                card_refs[job.id] = refs

    await full_refresh()

    async def poll() -> None:
        """Poll for updates — only re-render changed parts."""
        jobs = await jobs_service.list_jobs()
        has_running = any(j.status == JobStatus.running for j in jobs)
        if not has_running and not any(
            r["status"] == JobStatus.running for r in card_refs.values()
        ):
            # Nothing running and nothing was running — no update needed
            return

        # Check if any job changed status (needs full refresh for layout changes)
        needs_full_refresh = False
        current_ids = {j.id for j in jobs}
        tracked_ids = set(card_refs.keys())

        if current_ids != tracked_ids:
            needs_full_refresh = True
        else:
            for job in jobs:
                tracked = card_refs.get(job.id)
                if tracked and tracked["status"] != job.status:
                    needs_full_refresh = True
                    break

        if needs_full_refresh:
            await full_refresh()
            return

        # Only update progress for running jobs (no flicker)
        for job in jobs:
            if job.status != JobStatus.running:
                continue
            refs = card_refs.get(job.id)
            if refs is None:
                continue
            _render_progress(refs["progress_container"], job)

    ui.timer(3.0, poll)
