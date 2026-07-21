"""New Job page — configure and start pipeline runs.

Renders a schema-driven form for the selected pipeline. The form is built
recursively from the pipeline's config schema (same introspection used by
the CLI) so adding a new stage or pipeline requires zero UI changes.

After submission the job is tracked as an async subprocess (see
``webgui.services.jobs``) and the user is redirected to the job history page.
"""

from __future__ import annotations

import logging
from typing import Any

from nicegui import ui

from webgui.components.navbar import build_navbar
from webgui.services import jobs as jobs_service
from webgui.services import pipelines as pipelines_service
from webgui.services import presets as presets_service
from webgui.services import prompts as prompts_service

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema → form renderer
# ---------------------------------------------------------------------------


def _widget_kind(param: dict[str, Any]) -> str:
    """Classify a serialized param dict into a widget kind for rendering."""
    type_str = param.get("type", "str").lower()

    if param.get("choices"):
        return "select"
    widget = param.get("widget") or {}
    if widget.get("source") == "prompts":
        return "prompt_file"
    if "bool" in type_str:
        return "bool"
    if "float" in type_str:
        return "float"
    if "int" in type_str and "none" not in type_str:
        return "int"
    if "int" in type_str:
        return "int_optional"
    if "date" in type_str:
        return "date"
    return "str"


class _FormField:
    """A single parameter field in the pipeline form."""

    __slots__ = ("key", "param", "element", "kind")

    def __init__(self, key: str, param: dict[str, Any], element: Any, kind: str) -> None:
        self.key = key
        self.param = param
        self.element = element
        self.kind = kind

    def value(self) -> Any:
        """Return the coerced value for this field, or None if empty/default."""
        raw = self.element.value
        if raw is None or raw == "":
            return None
        if self.kind == "bool":
            return bool(raw)
        if self.kind in ("int", "int_optional"):
            try:
                return int(raw)
            except (TypeError, ValueError):
                return None
        if self.kind == "float":
            try:
                return float(raw)
            except (TypeError, ValueError):
                return None
        return raw


def _render_param(param: dict[str, Any]) -> _FormField | None:
    """Create a NiceGUI widget for a serialized ConfigParam dict."""
    key = param["key"]
    kind = _widget_kind(param)
    default = param.get("default")
    label = key.replace("_", " ").title()
    required = param.get("required", False)
    desc = param.get("description") or ""

    if required:
        label += " *"

    if kind == "select":
        choices = param["choices"]
        element = ui.select(
            options=choices,
            value=default if default in choices else (choices[0] if choices else None),
            label=label,
        ).classes("w-full").props("outlined dense")

    elif kind == "prompt_file":
        prompts = prompts_service.list_prompts()
        options = {p.name: p.label for p in prompts}
        element = ui.select(
            options=options,
            value=default,
            label=label,
        ).classes("w-full").props("outlined dense clearable")

    elif kind == "bool":
        element = ui.switch(label, value=bool(default) if default is not None else False)

    elif kind in ("int", "int_optional"):
        element = ui.number(
            label=label,
            value=default,
            min=param.get("min"),
            max=param.get("max"),
        ).classes("w-full").props("outlined dense")

    elif kind == "float":
        element = ui.number(
            label=label,
            value=default,
            min=param.get("min"),
            max=param.get("max"),
            step=0.01,
        ).classes("w-full").props("outlined dense")

    elif kind == "date":
        element = ui.input(
            label=label,
            value=str(default) if default else "",
            placeholder="YYYY-MM-DD",
        ).classes("w-full").props("outlined dense")

    else:
        element = ui.input(
            label=label,
            value=str(default) if default is not None else "",
        ).classes("w-full").props("outlined dense")

    if desc:
        ui.label(desc).classes("text-caption text-grey q-mt-none q-mb-xs")

    return _FormField(key=key, param=param, element=element, kind=kind)


def _render_steps(steps: list[dict[str, Any]], path_prefix: str = "") -> list[dict[str, Any]]:
    """Recursively render pipeline steps into the current NiceGUI container.

    Top-level steps are rendered as labeled cards (always visible).
    Composite branch children are nested inside expansion panels.

    Returns a list of section descriptors for collecting values at submit time.
    """
    sections: list[dict[str, Any]] = []

    for step in steps:
        alias = step.get("alias") or step.get("stage", "")
        full_alias = f"{path_prefix}{alias}" if path_prefix else alias
        params = step.get("params", {})
        children = step.get("children")

        if not params and not children:
            continue

        section: dict[str, Any] = {
            "alias": alias,
            "full_alias": full_alias,
            "fields": [],
            "children": [],
        }

        with ui.card().classes("w-full q-pa-sm"):
            ui.label(alias).classes("text-subtitle2 text-weight-bold")
            ui.label(step.get("stage", "")).classes("text-caption text-grey")

            # Render parameter fields
            with ui.column().classes("w-full gap-1 q-mt-xs"):
                for _key, param in params.items():
                    field = _render_param(param)
                    if field:
                        section["fields"].append(field)

            # Render composite children (fork branches)
            if children:
                for branch in children.get("branches", []):
                    branch_prefix = f"{full_alias}.{children['key']}.{branch['name']}."
                    with ui.expansion(
                        f"Branch: {branch['name']}",
                    ).classes("w-full"):
                        child_sections = _render_steps(
                            branch.get("steps", []),
                            path_prefix=branch_prefix,
                        )
                        section["children"].extend(child_sections)

        sections.append(section)

    return sections


def _collect_values(sections: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Walk the section tree and collect non-default field values."""
    result: dict[str, dict[str, Any]] = {}

    for section in sections:
        values: dict[str, Any] = {}
        for field in section["fields"]:
            v = field.value()
            if v is not None and (
                field.param.get("required") or v != field.param.get("default")
            ):
                values[field.key] = v
        if values:
            result[section["full_alias"]] = values

        # Recurse into children
        child_values = _collect_values(section["children"])
        result.update(child_values)

    return result


def _apply_settings(
    sections: list[dict[str, Any]],
    pipeline_settings: dict[str, dict[str, Any]],
) -> None:
    """Apply saved pipeline_settings values back into form fields.

    Walks the section tree and sets field element values for matching
    alias keys from the preset's saved settings dict.
    """
    for section in sections:
        alias_settings = pipeline_settings.get(section["full_alias"], {})
        for field in section["fields"]:
            if field.key in alias_settings:
                field.element.value = alias_settings[field.key]
        # Recurse into children
        _apply_settings(section["children"], pipeline_settings)


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


@ui.page("/jobs/new")
async def new_job_page() -> None:
    """Pipeline configuration and job submission page."""
    await ui.context.client.connected()
    ui.dark_mode(True)
    build_navbar()

    schemas = await pipelines_service.get_all_schemas_with_user_pipelines()
    pipeline_names = list(schemas.keys())

    if not pipeline_names:
        with ui.column().classes("w-full max-w-4xl mx-auto q-pa-md"):
            ui.label("No pipelines registered.").classes("text-h5 text-warning")
        return

    # Mutable state shared between render and submit
    current_sections: list[dict[str, Any]] = []

    with ui.column().classes("w-full max-w-4xl mx-auto q-pa-md gap-4"):
        ui.label("Start a Pipeline").classes("text-h4")
        ui.label(
            "Select a pipeline, configure its stages, then submit. "
            "The job runs as a background process — you can navigate away."
        ).classes("text-caption text-grey")

        # Pipeline selector
        pipeline_options = {name: schemas[name]["label"] for name in pipeline_names}
        pipeline_select = ui.select(
            options=pipeline_options,
            value=pipeline_names[0],
            label="Pipeline",
        ).classes("w-80").props("outlined dense")

        # Description
        desc_label = ui.label(schemas[pipeline_names[0]].get("description", "")).classes(
            "text-caption text-grey"
        )

        # --- Preset bar ---
        preset_row = ui.row().classes("w-full items-center gap-2")
        preset_select = None
        preset_data: dict[str, dict[str, Any]] = {}

        async def refresh_presets() -> None:
            nonlocal preset_select, preset_data
            preset_row.clear()
            pipeline_name = pipeline_select.value
            presets = await presets_service.list_presets(pipeline_name)
            preset_data = {p["id"]: p for p in presets}
            options = {p["id"]: p["name"] for p in presets}
            with preset_row:
                if options:
                    preset_select = ui.select(
                        options=options,
                        label="Load preset",
                        value=None,
                    ).classes("w-64").props("outlined dense clearable")
                    ui.button(
                        "Load", icon="download", on_click=load_preset
                    ).props("flat size=sm")
                    ui.button(
                        "Delete", icon="delete", on_click=delete_preset
                    ).props("flat size=sm color=negative")
                else:
                    preset_select = None
                    ui.label("No presets saved for this pipeline.").classes(
                        "text-caption text-grey"
                    )
                ui.button(
                    "Save as preset", icon="save", on_click=save_preset
                ).props("flat size=sm")

        async def load_preset() -> None:
            if preset_select is None or not preset_select.value:
                ui.notify("Select a preset first", type="warning")
                return
            preset = preset_data.get(preset_select.value)
            if not preset:
                return
            # Rebuild form, then apply saved values
            build_form()
            _apply_settings(current_sections, preset["pipeline_settings"])
            ui.notify(f"Loaded preset: {preset['name']}", type="info")

        async def save_preset() -> None:
            pipeline_settings = _collect_values(current_sections)
            pipeline_name = pipeline_select.value

            # Dialog for preset name
            with ui.dialog() as dlg, ui.card().classes("w-80"):
                ui.label("Save Preset").classes("text-h6")
                name_input = ui.input(label="Preset name").classes("w-full").props(
                    "outlined dense autofocus"
                )
                error_lbl = ui.label("").classes("text-negative")
                error_lbl.set_visibility(False)

                async def do_save() -> None:
                    try:
                        await presets_service.save_preset(
                            pipeline_name, name_input.value or "", pipeline_settings
                        )
                        ui.notify("Preset saved", type="positive")
                        dlg.close()
                        await refresh_presets()
                    except presets_service.PresetError as exc:
                        error_lbl.text = str(exc)
                        error_lbl.set_visibility(True)

                with ui.row().classes("w-full justify-end gap-2 q-mt-sm"):
                    ui.button("Cancel", on_click=dlg.close).props("flat")
                    ui.button("Save", on_click=do_save).props("color=primary")
            dlg.open()

        async def delete_preset() -> None:
            if preset_select is None or not preset_select.value:
                ui.notify("Select a preset first", type="warning")
                return
            preset = preset_data.get(preset_select.value)
            if not preset:
                return
            try:
                await presets_service.delete_preset(preset["id"])
                ui.notify(f"Deleted preset: {preset['name']}", type="info")
                await refresh_presets()
            except presets_service.PresetError as exc:
                ui.notify(str(exc), type="negative")

        # Form area — use a column that we clear and rebuild
        form_container = ui.column().classes("w-full gap-1")

        def build_form() -> None:
            nonlocal current_sections
            form_container.clear()
            name = pipeline_select.value
            desc_label.text = schemas[name].get("description", "")
            schema_tree = schemas[name]["schema"]
            with form_container:
                current_sections = _render_steps(schema_tree)

        async def on_pipeline_change(_) -> None:
            build_form()
            await refresh_presets()

        pipeline_select.on_value_change(on_pipeline_change)
        build_form()
        # Initial preset load (async, after page renders)
        ui.timer(0.1, refresh_presets, once=True)

        ui.separator()

        # Advanced options
        with ui.expansion("⚙️ Advanced", icon="settings").classes("w-full"):
            log_level_select = ui.select(
                options=["DEBUG", "INFO", "WARNING", "ERROR"],
                value="INFO",
                label="Log Level",
            ).classes("w-64").props("outlined dense")

        # Submit button
        async def submit() -> None:
            pipeline_settings = _collect_values(current_sections)
            pipeline_settings["_log_level"] = log_level_select.value
            pipeline_name = pipeline_select.value
            logger.info(
                "Submitting job: pipeline=%s settings=%s",
                pipeline_name, pipeline_settings,
            )
            try:
                record = await jobs_service.create_job(pipeline_name, pipeline_settings)
                ui.notify(f"Job {record.id} started", type="positive")
                ui.navigate.to("/jobs")
            except Exception as exc:
                logger.exception("Failed to start job")
                ui.notify(f"Failed: {exc}", type="negative")

        with ui.row().classes("w-full justify-end"):
            ui.button("Start job", icon="play_arrow", on_click=submit).props(
                "color=primary"
            )
