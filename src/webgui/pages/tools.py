"""Admin tools page — run maintenance scripts and manage backups.

Provides:
- A list of available admin scripts that can be executed from the UI.
- Database backup (download as .zip) and restore (upload .zip) functionality.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Response
from nicegui import app, ui

from webgui.components.navbar import build_navbar
from webgui.state import state

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Script registry — scripts that admins can run from the Tools page
# ---------------------------------------------------------------------------

_SCRIPTS: list[dict[str, str]] = [
    {
        "id": "migrate_best_of_year",
        "name": "Migrate Best-of-Year Presets",
        "description": (
            "Updates old best-of-year pipeline_settings to use the new composite "
            "node layout (DedupPHashComposite, DedupSimilar, DedupScenes)."
        ),
        "script": "scripts/migrate_best_of_year_presets.py",
    },
]


async def _run_script(script_id: str, log_area: Any) -> None:
    """Execute a registered script against the current database."""
    script = next((s for s in _SCRIPTS if s["id"] == script_id), None)
    if script is None:
        ui.notify(f"Unknown script: {script_id}", type="negative")
        return

    db_path = state.config.db_path
    log_area.set_value("")

    try:
        # Run the script file directly to avoid module resolution issues
        import sys
        script_path = str(Path(script["script"]).resolve())
        proc = await asyncio.create_subprocess_exec(
            sys.executable, script_path, db_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode() if stdout else ""
        log_area.set_value(output)

        if proc.returncode == 0:
            ui.notify("Script completed successfully.", type="positive")
        else:
            ui.notify(f"Script exited with code {proc.returncode}.", type="warning")

    except Exception as exc:
        logger.exception("Failed to run script %s", script_id)
        log_area.set_value(f"Error: {exc}")
        ui.notify(f"Script failed: {exc}", type="negative")


# ---------------------------------------------------------------------------
# Backup & Restore
# ---------------------------------------------------------------------------

def _get_backup_paths() -> list[Path]:
    """Collect all paths that should be included in a backup."""
    paths: list[Path] = []

    # Database
    db_path = Path(state.config.db_path)
    if db_path.exists():
        paths.append(db_path)

    # Prompts directory
    prompts_dir = Path(state.config.prompts_dir)
    if prompts_dir.exists():
        for f in prompts_dir.rglob("*"):
            if f.is_file():
                paths.append(f)

    return paths


def _create_backup_zip() -> bytes:
    """Create a zip archive containing the database and prompts."""
    buf = io.BytesIO()
    db_path = Path(state.config.db_path)
    prompts_dir = Path(state.config.prompts_dir)

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Add database
        if db_path.exists():
            zf.write(db_path, "db/smart_albums.db")

        # Add prompts
        if prompts_dir.exists():
            for f in prompts_dir.rglob("*"):
                if f.is_file():
                    arcname = f"prompts/{f.relative_to(prompts_dir)}"
                    zf.write(f, arcname)

        # Add metadata
        meta = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "db_path": str(db_path),
            "prompts_dir": str(prompts_dir),
        }
        zf.writestr("backup_meta.json", json.dumps(meta, indent=2))

    return buf.getvalue()


async def _restore_from_zip(content: bytes) -> str:
    """Restore database and prompts from a backup zip.

    Returns a summary message of what was restored.
    """
    restored: list[str] = []

    with zipfile.ZipFile(io.BytesIO(content), "r") as zf:
        names = zf.namelist()

        # Restore database
        if "db/smart_albums.db" in names:
            db_path = Path(state.config.db_path)
            # Close current connection, write new file, reconnect
            await state.db.close()
            db_data = zf.read("db/smart_albums.db")
            db_path.parent.mkdir(parents=True, exist_ok=True)
            db_path.write_bytes(db_data)
            await state.db.connect()
            await state.reload_providers()
            restored.append("database")

        # Restore prompts
        prompts_dir = Path(state.config.prompts_dir)
        prompt_files = [n for n in names if n.startswith("prompts/") and not n.endswith("/")]
        if prompt_files:
            for name in prompt_files:
                data = zf.read(name)
                # Strip the "prompts/" prefix
                rel_path = name[len("prompts/"):]
                target = prompts_dir / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            restored.append(f"{len(prompt_files)} prompt file(s)")

    return ", ".join(restored) if restored else "nothing"


# ---------------------------------------------------------------------------
# Backup download endpoint (FastAPI)
# ---------------------------------------------------------------------------

@app.get("/api/tools/backup")
async def download_backup() -> Response:
    """Generate and return a backup zip file."""
    zip_bytes = _create_backup_zip()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"smart_albums_backup_{timestamp}.zip"
    return Response(
        content=zip_bytes,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

@ui.page("/tools")
async def tools_page() -> None:
    """Admin tools page — scripts, backup, and restore."""
    await ui.context.client.connected()
    ui.dark_mode(True)
    build_navbar()

    with ui.column().classes("w-full max-w-3xl mx-auto q-pa-md gap-6"):
        ui.label("Admin Tools").classes("text-h4")

        # --- Scripts Section ---
        ui.separator()
        ui.label("Maintenance Scripts").classes("text-h5")
        ui.label(
            "Run database maintenance and migration scripts. "
            "Scripts operate on the current database."
        ).classes("text-body2 text-grey")

        for script in _SCRIPTS:
            with ui.card().classes("w-full"):
                with ui.row().classes("items-center justify-between w-full"):
                    with ui.column().classes("gap-0"):
                        ui.label(script["name"]).classes("text-subtitle1 text-weight-medium")
                        ui.label(script["description"]).classes("text-caption text-grey")
                    run_btn = ui.button("Run", icon="play_arrow").props("color=primary")

                log_area = ui.textarea(
                    label="Output",
                ).props("readonly outlined").classes("w-full font-mono").style(
                    "display: none"
                )

                async def _on_run(s=script, log=log_area):
                    log.style("display: block")
                    await _run_script(s["id"], log)

                run_btn.on_click(_on_run)

        # --- Backup & Restore Section ---
        ui.separator()
        ui.label("Backup & Restore").classes("text-h5")
        ui.label(
            "Download a complete backup (database + prompts) as a .zip file, "
            "or restore from a previously downloaded backup."
        ).classes("text-body2 text-grey")

        with ui.card().classes("w-full"):
            with ui.row().classes("items-center gap-4"):
                # Download backup
                ui.button(
                    "Download Backup",
                    icon="download",
                    on_click=lambda: ui.download("/api/tools/backup"),
                ).props("color=primary")

                ui.separator().props("vertical")

                # Restore from backup
                with ui.column().classes("gap-2"):
                    ui.label("Restore from backup").classes("text-subtitle2")

                    restore_status = ui.label("").classes("text-caption")

                    async def _handle_upload(e) -> None:
                        if not e.content:
                            return
                        restore_status.set_text("Restoring...")
                        try:
                            content = e.content.read()
                            result = await _restore_from_zip(content)
                            restore_status.set_text(f"✓ Restored: {result}")
                            ui.notify(f"Backup restored: {result}", type="positive")
                        except Exception as exc:
                            logger.exception("Restore failed")
                            restore_status.set_text(f"✗ Failed: {exc}")
                            ui.notify(f"Restore failed: {exc}", type="negative")

                    ui.upload(
                        label="Upload .zip backup",
                        on_upload=_handle_upload,
                        auto_upload=True,
                    ).props("accept=.zip flat bordered").classes("max-w-xs")
