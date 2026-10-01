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
import os
import shutil
import sqlite3
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Response
from nicegui import app, ui

from webgui.components.navbar import build_navbar
from webgui.services.scheduler import maintenance_lock
from webgui.state import state

logger = logging.getLogger(__name__)

# Zip-bomb / resource-exhaustion guards for restoring an uploaded backup.
_MAX_RESTORE_ENTRY_BYTES = 200 * 1024 * 1024  # 200 MB per member
_MAX_RESTORE_TOTAL_BYTES = 1024 * 1024 * 1024  # 1 GB uncompressed total
_MAX_RESTORE_ENTRIES = 5000


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


# Scripts are resolved relative to the package root, not the process CWD
# (WG-24) — the previous `Path(script["script"]).resolve()` only worked
# when the server happened to be started with CWD == the repo/package
# root (e.g. `/app` in the Docker image), and silently failed with
# FileNotFoundError anywhere else.
#
# Candidate roots the ``scripts/`` directory may live under, highest
# priority first (DK-03): an explicit ``SMART_ALBUMS_SCRIPTS_DIR`` override,
# then ``/app`` (where the Docker image copies ``scripts/`` even though the
# package itself is imported from the installed wheel, not from /app), then
# the source-checkout layout (repo root is 3 parents up from this file).
def _resolve_script(rel_path: str) -> Path | None:
    """Find a registered maintenance script across supported layouts.

    Returns the first existing candidate path, or ``None`` if the script
    can't be located (so the caller reports a clear error rather than
    shelling out to a non-existent path).
    """
    import os

    candidates: list[Path] = []
    env_dir = os.environ.get("SMART_ALBUMS_SCRIPTS_DIR")
    if env_dir:
        # The env var points at the scripts dir; rel_path is "scripts/<name>".
        candidates.append(Path(env_dir) / Path(rel_path).name)
    candidates.append(Path("/app") / rel_path)
    candidates.append(Path(__file__).resolve().parents[3] / rel_path)
    candidates.append(Path.cwd() / rel_path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None

# A migration script that hangs (e.g. on a locked database file) would
# otherwise block this request — and the asyncio event loop for every
# other connected client — forever, since `proc.communicate()` had no
# timeout (WG-24).
_SCRIPT_TIMEOUT_SECONDS = 120.0


async def _run_script(
    script_id: str, log_area: Any, *, dry_run: bool = False
) -> None:
    """Execute a registered script against the current database.

    Takes a backup snapshot of the live database before a non-dry-run
    invocation (WG-24) — mirrors the pattern used by the "Backup & Restore"
    section's `_create_backup_zip`, but as a plain `.bak` file next to the
    live database so a bad migration can be undone without going through
    the full zip restore flow.
    """
    script = next((s for s in _SCRIPTS if s["id"] == script_id), None)
    if script is None:
        ui.notify(f"Unknown script: {script_id}", type="negative")
        return

    db_path = state.config.db_path
    log_area.set_value("")

    script_path = _resolve_script(script["script"])
    if script_path is None:
        msg = (
            f"Script not found: {script['script']} "
            "(looked under SMART_ALBUMS_SCRIPTS_DIR, /app, the package root, and CWD)"
        )
        log_area.set_value(f"Error: {msg}")
        ui.notify(msg, type="negative")
        return

    backup_note = ""
    if not dry_run:
        try:
            backup_path = f"{db_path}.bak-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
            await state.db.backup_to(backup_path)
            backup_note = f"Pre-run backup: {backup_path}\n\n"
        except Exception:
            logger.exception("Failed to create pre-run backup before running %s", script_id)
            ui.notify("Could not create a pre-run backup — aborting.", type="negative")
            return

    try:
        import sys
        args = [sys.executable, str(script_path), db_path]
        if dry_run:
            args.append("--dry-run")
        # The script does its own backup (see scripts/migrate_best_of_year_presets.py's
        # `backup_database`) — skip the redundant second one for a live run
        # since this handler already took one above.
        if not dry_run:
            args.append("--skip-backup")

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            stdout, _ = await asyncio.wait_for(
                proc.communicate(), timeout=_SCRIPT_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            log_area.set_value(
                f"{backup_note}Error: script timed out after "
                f"{_SCRIPT_TIMEOUT_SECONDS:.0f}s and was killed."
            )
            ui.notify("Script timed out and was killed.", type="negative")
            return

        output = stdout.decode() if stdout else ""
        log_area.set_value(backup_note + output)

        if proc.returncode == 0:
            ui.notify(
                "Dry run completed." if dry_run else "Script completed successfully.",
                type="positive",
            )
        else:
            ui.notify(f"Script exited with code {proc.returncode}.", type="warning")

    except Exception as exc:
        logger.exception("Failed to run script %s", script_id)
        log_area.set_value(f"{backup_note}Error: {exc}")
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


async def _create_backup_zip() -> bytes:
    """Create a zip archive containing a consistent DB snapshot and prompts.

    The database is captured via ``Database.backup_to`` (SQLite's online
    backup API) into a temp file rather than by zipping the live file
    directly. In WAL mode (see ``Database.connect``), copying the raw file
    can miss committed data that is still sitting in the ``-wal`` file, or
    produce a torn/corrupt copy if a checkpoint runs mid-copy. The backup
    API produces a correct, self-contained snapshot regardless.
    """
    buf = io.BytesIO()
    prompts_dir = Path(state.config.prompts_dir)

    with tempfile.TemporaryDirectory() as tmp_dir:
        snapshot_path = Path(tmp_dir) / "smart_albums_snapshot.db"
        await state.db.backup_to(str(snapshot_path))

        def _build_zip() -> bytes:
            # Zipping and file I/O are synchronous; keep them off the event
            # loop since the DB snapshot and prompts can be sizeable.
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                if snapshot_path.exists():
                    zf.write(snapshot_path, "db/smart_albums.db")

                if prompts_dir.exists():
                    for f in prompts_dir.rglob("*"):
                        if f.is_file():
                            arcname = f"prompts/{f.relative_to(prompts_dir)}"
                            zf.write(f, arcname)

                meta = {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "db_path": str(state.config.db_path),
                    "prompts_dir": str(prompts_dir),
                    "format_version": 2,
                }
                zf.writestr("backup_meta.json", json.dumps(meta, indent=2))
            return buf.getvalue()

        return await asyncio.to_thread(_build_zip)


class RestoreError(Exception):
    """Raised when an uploaded backup zip fails validation."""


def _safe_extract_path(base_dir: Path, member_name: str) -> Path:
    """Resolve a zip member name to a path confined to ``base_dir``.

    Rejects absolute paths and any ``..`` traversal so a crafted zip entry
    (zip-slip) cannot write outside the intended directory.
    """
    candidate = (base_dir / member_name).resolve()
    base_resolved = base_dir.resolve()
    if candidate != base_resolved and base_resolved not in candidate.parents:
        raise RestoreError(f"Refusing to extract unsafe path from backup: {member_name!r}")
    return candidate


def _validate_zip_contents(zf: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    """Validate size/entry-count limits before extracting anything.

    Guards against zip bombs (a small file that expands to a huge size) and
    against a backup with an unreasonable number of entries.
    """
    infos = zf.infolist()
    if len(infos) > _MAX_RESTORE_ENTRIES:
        raise RestoreError(f"Backup has too many entries ({len(infos)})")
    total = 0
    for info in infos:
        if info.file_size > _MAX_RESTORE_ENTRY_BYTES:
            raise RestoreError(f"Backup entry too large: {info.filename}")
        total += info.file_size
        if total > _MAX_RESTORE_TOTAL_BYTES:
            raise RestoreError("Backup is too large (uncompressed size limit exceeded)")
    return infos


def _validate_sqlite_file(path: Path) -> None:
    """Verify ``path`` is a well-formed SQLite database with the expected schema."""
    try:
        conn = sqlite3.connect(str(path))
        try:
            (result,) = conn.execute("PRAGMA integrity_check").fetchone()
            if result != "ok":
                raise RestoreError(f"Database integrity check failed: {result}")
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }
            required = {"users", "jobs", "settings"}
            missing = required - tables
            if missing:
                raise RestoreError(f"Backup database is missing expected tables: {sorted(missing)}")
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        raise RestoreError(f"Not a valid SQLite database: {exc}") from exc


async def _restore_from_zip(content: bytes) -> str:
    """Restore database and prompts from a backup zip.

    The restore is designed to fail safe:
    - The uploaded zip is validated (size limits, safe paths) before any
      file on disk is touched.
    - The candidate database is validated (integrity check + expected
      tables) *before* the live database is touched.
    - The current database is kept as a ``.bak`` copy so a bad restore can
      be reverted by hand, and the swap uses ``os.replace`` (atomic on the
      same filesystem) instead of overwriting the live file in place.
    - The scheduler is paused for the duration so it can't fire a job
      against a connection that is being closed/reconnected.
    - If reconnecting to the restored database fails, the pre-restore
      database is restored automatically so the server doesn't end up
      bricked.

    Returns a summary message of what was restored.
    """
    restored: list[str] = []

    with zipfile.ZipFile(io.BytesIO(content), "r") as zf:
        infos = _validate_zip_contents(zf)
        names = {info.filename for info in infos}

        prompts_dir = Path(state.config.prompts_dir)
        prompt_infos = [
            info for info in infos
            if info.filename.startswith("prompts/") and not info.filename.endswith("/")
        ]
        # Pre-validate every prompt path before writing anything.
        prompt_targets = [
            (_safe_extract_path(prompts_dir, info.filename[len("prompts/"):]), info)
            for info in prompt_infos
        ]

        async with maintenance_lock():
            if "db/smart_albums.db" in names:
                db_path = Path(state.config.db_path)
                db_data = zf.read("db/smart_albums.db")

                with tempfile.TemporaryDirectory() as tmp_dir:
                    candidate_path = Path(tmp_dir) / "candidate.db"
                    candidate_path.write_bytes(db_data)
                    _validate_sqlite_file(candidate_path)

                    backup_path = db_path.with_suffix(db_path.suffix + ".bak")
                    await state.db.close()
                    try:
                        db_path.parent.mkdir(parents=True, exist_ok=True)
                        if db_path.exists():
                            shutil.copy2(db_path, backup_path)
                        # Drop stale WAL/SHM sidecar files so SQLite doesn't
                        # try to replay old write-ahead frames onto the
                        # freshly restored database file.
                        for suffix in ("-wal", "-shm"):
                            sidecar = Path(str(db_path) + suffix)
                            sidecar.unlink(missing_ok=True)
                        os.replace(candidate_path, db_path)
                        await state.db.connect()
                        await state.reload_providers()
                    except Exception:
                        logger.exception("Restore failed; rolling back to pre-restore database")
                        if backup_path.exists():
                            for suffix in ("-wal", "-shm"):
                                Path(str(db_path) + suffix).unlink(missing_ok=True)
                            os.replace(backup_path, db_path)
                            await state.db.connect()
                            await state.reload_providers()
                        raise
                restored.append("database")

            if prompt_targets:
                for target, info in prompt_targets:
                    data = zf.read(info.filename)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(data)
                restored.append(f"{len(prompt_targets)} prompt file(s)")

    return ", ".join(restored) if restored else "nothing"


# ---------------------------------------------------------------------------
# Backup download endpoint (FastAPI)
# ---------------------------------------------------------------------------

@app.get("/api/tools/backup")
async def download_backup() -> Response:
    """Generate and return a backup zip file."""
    zip_bytes = await _create_backup_zip()
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
                    with ui.row().classes("items-center gap-2"):
                        dry_run_btn = ui.button(
                            "Dry Run", icon="visibility"
                        ).props("outline color=primary")
                        run_btn = ui.button("Run", icon="play_arrow").props(
                            "color=primary"
                        )

                log_area = ui.textarea(
                    label="Output",
                ).props("readonly outlined").classes("w-full font-mono").style(
                    "display: none"
                )

                async def _execute(
                    s=script, log=log_area, btn=run_btn, dry_btn=dry_run_btn, *, dry_run: bool
                ) -> None:
                    # Guard against double-submit — a double-click would
                    # otherwise run the migration script twice concurrently
                    # against the same database (WG-23).
                    if btn.props.get("disable") or dry_btn.props.get("disable"):
                        return
                    btn.props("disable loading")
                    dry_btn.props("disable loading")
                    log.style("display: block")
                    try:
                        await _run_script(s["id"], log, dry_run=dry_run)
                    finally:
                        btn.props(remove="disable loading")
                        dry_btn.props(remove="disable loading")

                async def _on_dry_run(s=script) -> None:
                    await _execute(dry_run=True)

                async def _on_run(s=script) -> None:
                    # Running the script writes to the live database — even
                    # though a pre-run backup is now taken automatically
                    # (WG-24), require an explicit confirmation rather than
                    # letting a single accidental click run it live. "Dry
                    # Run" above lets an admin preview the changes first.
                    with ui.dialog() as dialog, ui.card():
                        ui.label(f'Run "{s["name"]}"?').classes("text-subtitle1")
                        ui.label(
                            "This runs the script against the live database. "
                            "A backup snapshot is taken automatically before "
                            "it runs, but review the dry run output first if "
                            "you haven't already."
                        ).classes("text-body2 text-grey")
                        with ui.row().classes("justify-end w-full gap-2"):
                            ui.button("Cancel", on_click=dialog.close).props("flat")

                            async def _confirm() -> None:
                                dialog.close()
                                await _execute(dry_run=False)

                            ui.button(
                                "Run", color="negative", on_click=_confirm
                            )
                    dialog.open()

                dry_run_btn.on_click(_on_dry_run)
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

                    async def _do_restore(content: bytes) -> None:
                        restore_status.set_text("Restoring...")
                        try:
                            result = await _restore_from_zip(content)
                            restore_status.set_text(f"✓ Restored: {result}")
                            ui.notify(f"Backup restored: {result}", type="positive")
                        except RestoreError as exc:
                            logger.warning("Restore rejected: %s", exc)
                            restore_status.set_text(f"✗ Rejected: {exc}")
                            ui.notify(f"Restore rejected: {exc}", type="negative")
                        except Exception as exc:
                            logger.exception("Restore failed")
                            restore_status.set_text(f"✗ Failed: {exc}")
                            ui.notify(f"Restore failed: {exc}", type="negative")

                    async def _handle_upload(e) -> None:
                        if not e.content:
                            return
                        content = e.content.read()

                        # Restoring is destructive (it replaces the live
                        # database, after keeping a .bak copy) — confirm
                        # before proceeding rather than acting on upload
                        # alone, which can happen from a stray drag-and-drop.
                        with ui.dialog() as dialog, ui.card():
                            ui.label("Restore from backup?").classes("text-subtitle1")
                            ui.label(
                                "This replaces the current database and prompts. "
                                "The current database is kept as a .bak file, but "
                                "any changes made after this backup was taken will "
                                "be lost."
                            ).classes("text-body2 text-grey")
                            with ui.row().classes("justify-end w-full gap-2"):
                                ui.button("Cancel", on_click=dialog.close).props("flat")

                                async def _confirm_restore() -> None:
                                    # Guard against double-submit — a second
                                    # click before the dialog closes could
                                    # otherwise race two restores against
                                    # the live database (WG-23).
                                    if restore_btn.props.get("disable"):
                                        return
                                    restore_btn.props("disable loading")
                                    dialog.close()
                                    try:
                                        await _do_restore(content)
                                    finally:
                                        restore_btn.props(remove="disable loading")

                                restore_btn = ui.button(
                                    "Restore",
                                    color="negative",
                                    on_click=_confirm_restore,
                                )
                        dialog.open()

                    ui.upload(
                        label="Upload .zip backup",
                        on_upload=_handle_upload,
                        auto_upload=True,
                        max_file_size=_MAX_RESTORE_TOTAL_BYTES,
                    ).props("accept=.zip flat bordered").classes("max-w-xs")
