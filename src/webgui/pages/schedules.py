"""Schedules page — create, view, and manage cron-based pipeline schedules.

Allows users to configure pipelines to run automatically on recurring
schedules using standard cron expressions. Each schedule can be configured
with full pipeline settings (via the schema-driven form) or by selecting
a saved preset.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from nicegui import ui

from webgui.components.navbar import build_navbar
from webgui.models import ScheduleRecord
from webgui.pages.new_job import _apply_settings, _collect_values, _render_steps
from webgui.services import pipelines as pipelines_service
from webgui.services import presets as presets_service
from webgui.services import scheduler as scheduler_service
from webgui.services.cron import CronExpression, CronParseError
from webgui.services.template_vars import resolve_template_variables

logger = logging.getLogger(__name__)

# Common cron presets for the quick-select dropdown
# Keys are the cron expression (the value written into the field),
# values are the human-readable labels shown in the dropdown.
_CRON_PRESETS = {
    "0 * * * *": "Every hour",
    "0 0 * * *": "Every day at midnight",
    "0 1 * * *": "Every day at 1 AM",
    "0 6 * * *": "Every day at 6 AM",
    "0 */6 * * *": "Every 6 hours",
    "0 */12 * * *": "Every 12 hours",
    "0 9 * * 0": "Every Monday at 9 AM",
    "0 8 * * 0-4": "Every weekday at 8 AM",
    "0 2 * * 6": "Every Sunday at 2 AM",
    "0 0 1 * *": "First day of month at midnight",
}


def _fmt_time(dt: datetime | None) -> str:
    """Format a datetime for display.

    All schedule times are computed and stored in UTC (see
    ``services/cron.py``); the "UTC" suffix makes that explicit instead of
    looking like an unqualified local time.
    """
    if dt is None:
        return "—"
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _describe_cron(expression: str) -> str:
    """Get a human-readable description of a cron expression."""
    try:
        cron = CronExpression(expression)
        return cron.describe()
    except (CronParseError, ValueError, IndexError):
        return expression


def _summarize_settings(pipeline_settings: dict[str, Any]) -> str:
    """Create a brief summary of pipeline_settings for display."""
    if not pipeline_settings:
        return "Default settings"
    # Show up to 3 key-value pairs
    parts: list[str] = []
    for alias, params in pipeline_settings.items():
        if alias.startswith("_"):
            continue
        if isinstance(params, dict):
            for k, v in params.items():
                parts.append(f"{alias}.{k}={v}")
        else:
            parts.append(f"{alias}={params}")
    if len(parts) > 3:
        return ", ".join(parts[:3]) + f" (+{len(parts) - 3} more)"
    return ", ".join(parts) if parts else "Default settings"


# ---------------------------------------------------------------------------
# Schedule card
# ---------------------------------------------------------------------------


def _build_schedule_card(schedule: ScheduleRecord, *, on_refresh, schemas: dict[str, Any] | None = None) -> None:
    """Render a single schedule card."""
    if schemas is None:
        schemas = pipelines_service.get_all_schemas()
    pipeline_info = schemas.get(schedule.pipeline, {})
    pipeline_label = pipeline_info.get("label", schedule.pipeline)

    status_color = "positive" if schedule.enabled else "grey"
    status_text = "Active" if schedule.enabled else "Paused"

    with ui.card().classes("w-full"):
        # Header row
        with ui.row().classes("w-full items-center justify-between"):
            with ui.row().classes("items-center gap-2"):
                ui.icon(
                    "schedule" if schedule.enabled else "pause_circle",
                    color=status_color,
                )
                ui.label(schedule.name).classes("text-subtitle1 text-weight-medium")
                ui.badge(status_text, color=status_color).props("outline")
            with ui.row().classes("items-center gap-1"):
                ui.label(f"#{schedule.id}").classes("text-caption text-grey")

        # Details
        with ui.grid(columns=2).classes("w-full gap-x-8 gap-y-1 q-mt-sm"):
            ui.label("Pipeline:").classes("text-caption text-grey")
            ui.label(pipeline_label).classes("text-caption")

            ui.label("Schedule:").classes("text-caption text-grey")
            with ui.row().classes("items-center gap-2"):
                ui.code(schedule.cron_expression).classes("text-xs")
                ui.label(f"({_describe_cron(schedule.cron_expression)})").classes(
                    "text-caption text-grey"
                )

            ui.label("Settings:").classes("text-caption text-grey")
            ui.label(_summarize_settings(schedule.pipeline_settings)).classes(
                "text-caption"
            )

            ui.label("Next run:").classes("text-caption text-grey")
            ui.label(_fmt_time(schedule.next_run_at)).classes("text-caption")

            ui.label("Last run:").classes("text-caption text-grey")
            with ui.row().classes("items-center gap-2"):
                ui.label(_fmt_time(schedule.last_run_at)).classes("text-caption")
                if schedule.last_job_id:
                    ui.button(
                        f"Job #{schedule.last_job_id}",
                        on_click=lambda jid=schedule.last_job_id: ui.navigate.to(
                            "/jobs"
                        ),
                    ).props("flat dense size=xs color=primary no-caps")

        # Actions
        with ui.row().classes("w-full justify-end gap-2 q-mt-sm"):

            async def do_toggle(sid=schedule.id) -> None:
                try:
                    await scheduler_service.toggle_schedule(sid)
                    await on_refresh()
                except ValueError as e:
                    ui.notify(str(e), type="negative")

            async def do_edit(sid=schedule.id) -> None:
                await _open_edit_dialog(sid, on_refresh=on_refresh)

            async def do_delete(sid=schedule.id) -> None:
                await _confirm_delete(sid, on_refresh=on_refresh)

            async def do_run_now(s=schedule) -> None:
                from webgui.services import jobs as jobs_service

                # Guard against double-submit — a double-click would
                # otherwise start two identical jobs for the same schedule
                # (WG-23).
                if run_now_btn.props.get("disable"):
                    return
                run_now_btn.props("disable loading")
                try:
                    settings_to_run = resolve_template_variables(
                        s.pipeline_settings, s.template_variables
                    )
                    record = await jobs_service.create_job(
                        s.pipeline, settings_to_run
                    )
                    ui.notify(f"Job {record.id} started", type="positive")
                except Exception as exc:
                    ui.notify(f"Failed: {exc}", type="negative")
                finally:
                    run_now_btn.props(remove="disable loading")

            toggle_label = "Pause" if schedule.enabled else "Enable"
            toggle_icon = "pause" if schedule.enabled else "play_arrow"

            run_now_btn = ui.button(
                "Run Now", icon="play_arrow", on_click=do_run_now
            ).props("flat size=sm color=primary")
            ui.button(
                toggle_label, icon=toggle_icon, on_click=do_toggle
            ).props("flat size=sm")
            ui.button(
                "Edit", icon="edit", on_click=do_edit
            ).props("flat size=sm")
            ui.button(
                "Delete", icon="delete", on_click=do_delete
            ).props("flat size=sm color=negative")


# ---------------------------------------------------------------------------
# Pipeline settings configuration component
# ---------------------------------------------------------------------------


def _build_pipeline_config_section(
    pipeline_name: str,
    existing_settings: dict[str, Any] | None = None,
    existing_template_variables: dict[str, str] | None = None,
    *,
    schemas: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the pipeline configuration section with preset selection and
    manual configuration via the schema-driven form.

    Args:
        schemas: The merged built-in + user-pipeline schema dict (from
            ``get_all_schemas_with_user_pipelines``). Previously this
            function always called the built-in-only ``get_all_schemas()``,
            so selecting a user-defined pipeline ("user:<id>") always
            rendered "No configurable parameters" even though the pipeline
            dropdown itself listed user pipelines (WG-28). Callers should
            fetch the merged schemas once and pass them in here.

    Returns a dict with:
        - 'get_settings': callable that returns the current pipeline_settings
        - 'get_template_variables': callable that returns the variable
          definitions to resolve at fire time (non-empty only when a preset
          carrying variables is selected)
        - 'container': the UI container element
    """
    if schemas is None:
        schemas = pipelines_service.get_all_schemas()
    schema_info = schemas.get(pipeline_name, {})
    schema_tree = schema_info.get("schema", [])

    # State for collecting form values
    current_sections: list[dict[str, Any]] = []
    config_mode = {"value": "preset"}  # 'preset' or 'manual'
    preset_settings: dict[str, Any] = {}
    preset_template_variables: dict[str, str] = dict(existing_template_variables or {})

    ui.label("Pipeline Configuration").classes(
        "text-subtitle2 text-weight-medium q-mt-md"
    )

    # Mode toggle
    with ui.row().classes("items-center gap-4"):
        mode_toggle = ui.toggle(
            {"preset": "Use Preset", "manual": "Configure Manually"},
            value="preset" if not existing_settings else "manual",
        ).props("dense no-caps")

    # Preset selection area
    preset_container = ui.column().classes("w-full gap-2")
    # Manual config area
    manual_container = ui.column().classes("w-full gap-1")

    async def build_preset_section() -> None:
        """Build the preset selection dropdown."""
        preset_container.clear()
        presets = await presets_service.list_presets(pipeline_name)
        with preset_container:
            if not presets:
                ui.label(
                    "No presets saved for this pipeline. "
                    "Save a preset from the Pipeline page first, or switch to manual configuration."
                ).classes("text-caption text-grey")
                return

            options = {p["id"]: p["name"] for p in presets}

            def on_preset_selected(e) -> None:
                selected = next(
                    (p for p in presets if p["id"] == e.value), None
                )
                if selected:
                    preset_settings.clear()
                    preset_settings.update(selected["pipeline_settings"])
                    preset_template_variables.clear()
                    preset_template_variables.update(
                        selected.get("template_variables") or {}
                    )

            select = ui.select(
                options=options,
                label="Select a preset",
                value=None,
                on_change=on_preset_selected,
            ).classes("w-full").props("outlined dense")

            # If existing settings match a preset, pre-select it
            if existing_settings:
                for p in presets:
                    if p["pipeline_settings"] == existing_settings:
                        select.value = p["id"]
                        preset_settings.update(p["pipeline_settings"])
                        preset_template_variables.update(
                            p.get("template_variables") or {}
                        )
                        break

    def build_manual_section() -> None:
        """Build the schema-driven pipeline configuration form."""
        nonlocal current_sections
        manual_container.clear()
        if not schema_tree:
            with manual_container:
                ui.label("No configurable parameters for this pipeline.").classes(
                    "text-caption text-grey"
                )
            return
        with manual_container:
            current_sections = _render_steps(schema_tree)
            # Apply existing settings if any
            if existing_settings:
                _apply_settings(current_sections, existing_settings)

    def on_mode_change(e) -> None:
        config_mode["value"] = e.value
        preset_container.set_visibility(e.value == "preset")
        manual_container.set_visibility(e.value == "manual")
        if e.value == "manual" and not manual_container.default_slot.children:
            build_manual_section()

    mode_toggle.on_value_change(on_mode_change)

    # Initial state
    if existing_settings:
        config_mode["value"] = "manual"
        preset_container.set_visibility(False)
        manual_container.set_visibility(True)
        build_manual_section()
    else:
        preset_container.set_visibility(True)
        manual_container.set_visibility(False)
        # Trigger async preset load
        ui.timer(0.1, build_preset_section, once=True)

    def get_settings() -> dict[str, Any]:
        """Return the pipeline_settings based on current mode."""
        if config_mode["value"] == "preset":
            return dict(preset_settings)
        else:
            return _collect_values(current_sections)

    def get_template_variables() -> dict[str, str]:
        """Return the template variable definitions to persist on the schedule.

        Only meaningful in preset mode (manual configuration has no
        placeholders to resolve, since the form always collects concrete
        values).
        """
        if config_mode["value"] == "preset":
            return dict(preset_template_variables)
        return {}

    return {
        "get_settings": get_settings,
        "get_template_variables": get_template_variables,
        "rebuild_presets": build_preset_section,
        "rebuild_manual": build_manual_section,
    }


# ---------------------------------------------------------------------------
# Dialogs
# ---------------------------------------------------------------------------


async def _open_create_dialog(*, on_refresh) -> None:
    """Open a dialog to create a new schedule."""
    schemas = await pipelines_service.get_all_schemas_with_user_pipelines()
    pipeline_options = {name: info["label"] for name, info in schemas.items()}

    with ui.dialog() as dlg, ui.card().classes("w-full max-w-2xl").style(
        "max-height: 90vh; overflow-y: auto"
    ):
        ui.label("Create Schedule").classes("text-h6")
        ui.label(
            "Configure a pipeline to run automatically on a cron schedule."
        ).classes("text-caption text-grey q-mb-sm")

        name_input = ui.input(
            label="Schedule Name",
            placeholder="e.g. Nightly Rotation",
        ).classes("w-full").props("outlined dense")

        # Pipeline selection
        first_pipeline = list(pipeline_options.keys())[0] if pipeline_options else None
        pipeline_select = ui.select(
            options=pipeline_options,
            label="Pipeline",
            value=first_pipeline,
        ).classes("w-full").props("outlined dense")

        # Cron expression
        ui.label("Schedule (UTC)").classes("text-subtitle2 text-weight-medium q-mt-md")
        with ui.row().classes("w-full items-end gap-2"):
            cron_input = ui.input(
                label="Cron Expression, UTC (min hour dom month dow)",
                placeholder="0 1 * * *",
                value="0 1 * * *",
            ).classes("w-full").props("outlined dense")

        preset_select = ui.select(
            options=_CRON_PRESETS,
            label="Quick presets",
            value=None,
        ).classes("w-full").props("outlined dense clearable")

        def on_cron_preset_change(e) -> None:
            if e.value:
                cron_input.value = e.value
                validate_cron_input()

        preset_select.on_value_change(on_cron_preset_change)

        # Cron validation feedback
        cron_hint = ui.label("").classes("text-caption")

        def validate_cron_input() -> None:
            expr = cron_input.value or ""
            if not expr.strip():
                cron_hint.text = ""
                return
            try:
                cron = CronExpression(expr)
                next_dt = cron.next_run()
                cron_hint.text = f"✓ Next run: {_fmt_time(next_dt)} — {cron.describe()}"
                cron_hint.classes(replace="text-caption text-positive")
            except CronParseError as e:
                cron_hint.text = f"✗ {e}"
                cron_hint.classes(replace="text-caption text-negative")

        cron_input.on("blur", lambda _: validate_cron_input())
        validate_cron_input()

        # Pipeline configuration section
        config_section_container = ui.column().classes("w-full")

        config_ref: dict[str, Any] = {}

        def build_config_section() -> None:
            config_section_container.clear()
            with config_section_container:
                result = _build_pipeline_config_section(
                    pipeline_select.value or first_pipeline,
                    schemas=schemas,
                )
                config_ref.clear()
                config_ref.update(result)

        def on_pipeline_change(_) -> None:
            build_config_section()

        pipeline_select.on_value_change(on_pipeline_change)
        build_config_section()

        ui.separator().classes("q-mt-md")

        error_label = ui.label("").classes("text-negative")
        error_label.set_visibility(False)

        async def do_create() -> None:
            # Guard against double-submit (WG-23).
            if create_btn.props.get("disable"):
                return
            create_btn.props("disable loading")
            try:
                pipeline_settings = config_ref["get_settings"]() if config_ref else {}
                template_variables = (
                    config_ref["get_template_variables"]() if config_ref else {}
                )
                try:
                    await scheduler_service.create_schedule(
                        name=name_input.value or "",
                        pipeline=pipeline_select.value or "",
                        pipeline_settings=pipeline_settings,
                        cron_expression=cron_input.value or "",
                        enabled=True,
                        template_variables=template_variables,
                    )
                    ui.notify("Schedule created", type="positive")
                    dlg.close()
                    await on_refresh()
                except ValueError as exc:
                    error_label.text = str(exc)
                    error_label.set_visibility(True)
            finally:
                create_btn.props(remove="disable loading")

        with ui.row().classes("w-full justify-end gap-2 q-mt-md"):
            ui.button("Cancel", on_click=dlg.close).props("flat")
            create_btn = ui.button("Create", on_click=do_create).props("color=primary")

    dlg.open()


async def _open_edit_dialog(schedule_id: str, *, on_refresh) -> None:
    """Open a dialog to edit an existing schedule."""
    schedule = await scheduler_service.get_schedule(schedule_id)
    if schedule is None:
        ui.notify("Schedule not found", type="negative")
        return

    schemas = await pipelines_service.get_all_schemas_with_user_pipelines()
    pipeline_options = {name: info["label"] for name, info in schemas.items()}

    with ui.dialog() as dlg, ui.card().classes("w-full max-w-2xl").style(
        "max-height: 90vh; overflow-y: auto"
    ):
        ui.label("Edit Schedule").classes("text-h6")

        name_input = ui.input(
            label="Schedule Name",
            value=schedule.name,
        ).classes("w-full").props("outlined dense")

        pipeline_select = ui.select(
            options=pipeline_options,
            label="Pipeline",
            value=schedule.pipeline,
        ).classes("w-full").props("outlined dense")

        # Cron expression
        ui.label("Schedule (UTC)").classes("text-subtitle2 text-weight-medium q-mt-md")
        cron_input = ui.input(
            label="Cron Expression, UTC (min hour dom month dow)",
            value=schedule.cron_expression,
        ).classes("w-full").props("outlined dense")

        cron_preset_select = ui.select(
            options=_CRON_PRESETS,
            label="Quick presets",
            value=None,
        ).classes("w-full").props("outlined dense clearable")

        def on_cron_preset_change(e) -> None:
            if e.value:
                cron_input.value = e.value
                validate_cron_input()

        cron_preset_select.on_value_change(on_cron_preset_change)

        cron_hint = ui.label("").classes("text-caption")

        def validate_cron_input() -> None:
            expr = cron_input.value or ""
            if not expr.strip():
                cron_hint.text = ""
                return
            try:
                cron = CronExpression(expr)
                next_dt = cron.next_run()
                cron_hint.text = f"✓ Next run: {_fmt_time(next_dt)} — {cron.describe()}"
                cron_hint.classes(replace="text-caption text-positive")
            except CronParseError as e:
                cron_hint.text = f"✗ {e}"
                cron_hint.classes(replace="text-caption text-negative")

        cron_input.on("blur", lambda _: validate_cron_input())
        validate_cron_input()

        # Pipeline configuration section — pre-filled with existing settings
        config_section_container = ui.column().classes("w-full")
        config_ref: dict[str, Any] = {}

        def build_config_section() -> None:
            config_section_container.clear()
            # Only pass existing_settings when pipeline matches
            existing = (
                schedule.pipeline_settings
                if pipeline_select.value == schedule.pipeline
                else None
            )
            existing_vars = (
                schedule.template_variables
                if pipeline_select.value == schedule.pipeline
                else None
            )
            with config_section_container:
                result = _build_pipeline_config_section(
                    pipeline_select.value or schedule.pipeline,
                    existing_settings=existing,
                    existing_template_variables=existing_vars,
                    schemas=schemas,
                )
                config_ref.clear()
                config_ref.update(result)

        def on_pipeline_change(_) -> None:
            build_config_section()

        pipeline_select.on_value_change(on_pipeline_change)
        build_config_section()

        ui.separator().classes("q-mt-md")

        enabled_switch = ui.switch("Enabled", value=schedule.enabled)

        error_label = ui.label("").classes("text-negative")
        error_label.set_visibility(False)

        async def do_save() -> None:
            # Guard against double-submit (WG-23).
            if save_btn.props.get("disable"):
                return
            save_btn.props("disable loading")
            try:
                pipeline_settings = config_ref["get_settings"]() if config_ref else {}
                template_variables = (
                    config_ref["get_template_variables"]() if config_ref else {}
                )
                try:
                    await scheduler_service.update_schedule(
                        schedule_id,
                        name=name_input.value,
                        pipeline=pipeline_select.value,
                        pipeline_settings=pipeline_settings,
                        cron_expression=cron_input.value,
                        enabled=enabled_switch.value,
                        template_variables=template_variables,
                    )
                    ui.notify("Schedule updated", type="positive")
                    dlg.close()
                    await on_refresh()
                except ValueError as exc:
                    error_label.text = str(exc)
                    error_label.set_visibility(True)
            finally:
                save_btn.props(remove="disable loading")

        with ui.row().classes("w-full justify-end gap-2 q-mt-md"):
            ui.button("Cancel", on_click=dlg.close).props("flat")
            save_btn = ui.button("Save", on_click=do_save).props("color=primary")

    dlg.open()


async def _confirm_delete(schedule_id: str, *, on_refresh) -> None:
    """Show a confirmation dialog before deleting a schedule."""
    schedule = await scheduler_service.get_schedule(schedule_id)
    if schedule is None:
        ui.notify("Schedule not found", type="negative")
        return

    with ui.dialog() as dlg, ui.card().classes("w-80"):
        ui.label("Delete Schedule").classes("text-h6")
        ui.label(
            f'Are you sure you want to delete "{schedule.name}"? '
            "This action cannot be undone."
        ).classes("q-mt-sm")

        async def do_delete() -> None:
            try:
                await scheduler_service.delete_schedule(schedule_id)
                ui.notify("Schedule deleted", type="info")
                dlg.close()
                await on_refresh()
            except ValueError as exc:
                ui.notify(str(exc), type="negative")

        with ui.row().classes("w-full justify-end gap-2 q-mt-md"):
            ui.button("Cancel", on_click=dlg.close).props("flat")
            ui.button("Delete", on_click=do_delete).props("color=negative")

    dlg.open()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


@ui.page("/schedules")
async def schedules_page() -> None:
    """Cron schedules management page."""
    await ui.context.client.connected()
    ui.dark_mode(True)
    build_navbar()

    container = ui.column().classes("w-full max-w-4xl mx-auto q-pa-md gap-3")

    async def refresh() -> None:
        """Rebuild the schedule list."""
        container.clear()
        schedules = await scheduler_service.list_schedules()
        schemas = await pipelines_service.get_all_schemas_with_user_pipelines()
        with container:
            # Header
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Schedules").classes("text-h4")
                ui.button(
                    "New Schedule",
                    icon="add",
                    on_click=lambda: _open_create_dialog(on_refresh=refresh),
                ).props("color=primary")

            ui.label(
                "Configure pipelines to run automatically on a cron schedule. "
                "Use standard cron expressions (minute hour day month weekday). "
                "All times are UTC."
            ).classes("text-caption text-grey")

            if not schedules:
                with ui.card().classes("w-full q-pa-lg items-center"):
                    ui.icon("schedule", size="48px", color="grey")
                    ui.label("No schedules configured").classes(
                        "text-subtitle1 text-grey q-mt-sm"
                    )
                    ui.label(
                        "Create a schedule to run pipelines automatically."
                    ).classes("text-caption text-grey")
                return

            for schedule in schedules:
                _build_schedule_card(schedule, on_refresh=refresh, schemas=schemas)

    await refresh()
