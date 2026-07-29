"""New Job page — configure and start pipeline runs.

Renders a schema-driven form for the selected pipeline. The form is built
recursively from the pipeline's config schema (same introspection used by
the CLI) so adding a new stage or pipeline requires zero UI changes.

Template Variables
------------------
Presets can define template variables (e.g. ``YEAR``) whose values are
substituted into pipeline_settings at run time. Any string value containing
``{VAR_NAME}`` will have placeholders replaced before the job is submitted.

After submission the job is tracked as an async subprocess (see
``webgui.services.jobs``) and the user is redirected to the job history page.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from typing import Any

from nicegui import ui

from webgui.components.navbar import build_navbar
from webgui.services import jobs as jobs_service
from webgui.services import pipelines as pipelines_service
from webgui.services import presets as presets_service
from webgui.services import prompts as prompts_service

logger = logging.getLogger(__name__)

# Pattern matching a single {VAR_NAME} placeholder (used across validation and resolution)
_TEMPLATE_PATTERN = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Pattern matching a valid variable name (without braces)
_VAR_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_numeric(value: str) -> bool:
    """Return True if the string represents a valid number."""
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _step_value(element: Any, step: float, *, min_val: float | None = None, max_val: float | None = None) -> None:
    """Increment or decrement a numeric input field's value by step, respecting optional bounds."""
    raw = element.value
    if not raw or not _is_numeric(str(raw)):
        return
    current = float(raw)
    new_val = current + step
    # Clamp to bounds if provided
    if min_val is not None and new_val < min_val:
        new_val = min_val
    if max_val is not None and new_val > max_val:
        new_val = max_val
    # Keep as int if step is integer-sized
    if step == int(step) and current == int(current):
        element.value = str(int(new_val))
    else:
        # Round relative to step precision to avoid IEEE 754 artifacts
        decimals = max(0, len(str(step).rstrip("0").split(".")[-1])) if "." in str(step) else 0
        element.value = str(round(new_val, max(decimals, 2)))


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
        # Allow template variable references (e.g. "{YEAR}") in any field type
        if isinstance(raw, str) and _TEMPLATE_PATTERN.fullmatch(raw):
            return raw
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


def _render_param(param: dict[str, Any], vars_getter: Callable[[], dict[str, str]] | None = None) -> _FormField | None:
    """Create a NiceGUI widget for a serialized ConfigParam dict."""
    key = param["key"]
    kind = _widget_kind(param)
    default = param.get("default")
    label = key.replace("_", " ").title()
    required = param.get("required", False)
    desc = param.get("description") or ""

    if required:
        label += " *"

    def _validate_numeric_or_var(value: str) -> bool:
        """Allow empty, numeric, or a {VAR} that is defined in the variables table."""
        if not value:
            return True
        if _is_numeric(value):
            return True
        if not _TEMPLATE_PATTERN.fullmatch(value):
            return False
        # Check that the variable is defined
        if vars_getter:
            var_name = value[1:-1]  # strip { and }
            defined = vars_getter()
            return var_name in defined
        return True

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
        step = 1
        min_val = param.get("min")
        max_val = param.get("max")
        element = ui.input(
            label=label,
            value=str(default) if default is not None else "",
            validation={"Must be a number or a defined {VARIABLE}": _validate_numeric_or_var},
        ).classes("w-full").props("outlined dense")
        with element.add_slot("append"):
            ui.button(
                icon="arrow_drop_up",
                on_click=lambda e, el=element, s=step, mn=min_val, mx=max_val: _step_value(el, s, min_val=mn, max_val=mx),
            ).props("flat dense round size=xs")
            ui.button(
                icon="arrow_drop_down",
                on_click=lambda e, el=element, s=step, mn=min_val, mx=max_val: _step_value(el, -s, min_val=mn, max_val=mx),
            ).props("flat dense round size=xs")

    elif kind == "float":
        step = param.get("step") or 0.01
        min_val = param.get("min")
        max_val = param.get("max")
        element = ui.input(
            label=label,
            value=str(default) if default is not None else "",
            validation={"Must be a number or a defined {VARIABLE}": _validate_numeric_or_var},
        ).classes("w-full").props("outlined dense")
        with element.add_slot("append"):
            ui.button(
                icon="arrow_drop_up",
                on_click=lambda e, el=element, s=step, mn=min_val, mx=max_val: _step_value(el, s, min_val=mn, max_val=mx),
            ).props("flat dense round size=xs")
            ui.button(
                icon="arrow_drop_down",
                on_click=lambda e, el=element, s=step, mn=min_val, mx=max_val: _step_value(el, -s, min_val=mn, max_val=mx),
            ).props("flat dense round size=xs")

    elif kind == "date":
        element = ui.input(
            label=label,
            value=str(default) if default else "",
            placeholder="YYYY-MM-DD",
        ).classes("w-full").props("outlined dense")

    else:
        def _validate_str_vars(value: str) -> bool:
            """For string fields, check that any {VAR} references are defined."""
            if not value or not vars_getter:
                return True
            defined = vars_getter()
            for match in _TEMPLATE_PATTERN.finditer(value):
                if match.group(1) not in defined:
                    return False
            return True

        element = ui.input(
            label=label,
            value=str(default) if default is not None else "",
            validation={"Undefined {VARIABLE} reference": _validate_str_vars},
        ).classes("w-full").props("outlined dense")

    if desc:
        ui.label(desc).classes("text-caption text-grey q-mt-none q-mb-xs")

    return _FormField(key=key, param=param, element=element, kind=kind)


def _render_steps(steps: list[dict[str, Any]], path_prefix: str = "", vars_getter: Callable[[], dict[str, str]] | None = None) -> list[dict[str, Any]]:
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
                    field = _render_param(param, vars_getter=vars_getter)
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
                            vars_getter=vars_getter,
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
# Template variable resolution
# ---------------------------------------------------------------------------


def _resolve_variables(
    pipeline_settings: dict[str, dict[str, Any]],
    variables: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Replace {VAR} placeholders in all string values within pipeline_settings.

    Only replaces variables that are defined in the variables dict.
    Unresolved placeholders are left as-is.

    Note: Only resolves one level deep (alias → flat key/value dict).
    Nested dict values are not recursed into.
    """
    if not variables:
        return pipeline_settings

    def _substitute(value: Any) -> Any:
        if isinstance(value, str):
            def _replace(m: re.Match) -> str:
                var_name = m.group(1)
                return variables.get(var_name, m.group(0))
            return _TEMPLATE_PATTERN.sub(_replace, value)
        return value

    resolved: dict[str, dict[str, Any]] = {}
    for alias, settings in pipeline_settings.items():
        if isinstance(settings, dict):
            resolved[alias] = {k: _substitute(v) for k, v in settings.items()}
        else:
            resolved[alias] = _substitute(settings)
    return resolved


def _find_undefined_variables(
    pipeline_settings: dict[str, Any],
    variables: dict[str, str],
) -> set[str]:
    """Return the set of {VAR} names used in settings that are not defined in variables."""
    defined = set(variables.keys())
    used: set[str] = set()

    for _alias, settings in pipeline_settings.items():
        if isinstance(settings, dict):
            for v in settings.values():
                if isinstance(v, str):
                    used.update(_TEMPLATE_PATTERN.findall(v))
        elif isinstance(settings, str):
            used.update(_TEMPLATE_PATTERN.findall(settings))

    return used - defined


# ---------------------------------------------------------------------------
# Template Variables UI component
# ---------------------------------------------------------------------------


class _TemplateVariablesEditor:
    """Expandable section for editing template variables (name/value pairs)."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

        with ui.expansion("Template Variables", icon="data_object").classes("w-full"):
            self._container = ui.column().classes("w-full gap-2")
            with ui.row().classes("w-full justify-start q-mt-sm"):
                ui.button("Add variable", icon="add", on_click=self._add_row).props(
                    "flat size=sm color=primary"
                )

    @staticmethod
    def _validate_var_name(value: str) -> bool:
        """Validate that variable name matches [A-Za-z_][A-Za-z0-9_]*."""
        if not value:
            return True  # empty is handled by get_variables (skips empty)
        return bool(_VAR_NAME_PATTERN.match(value))

    def _add_row(self, name: str = "", value: str = "") -> None:
        """Add a variable row to the editor."""
        row_data: dict[str, Any] = {}

        with self._container:
            with ui.row().classes("w-full items-center gap-2") as row_el:
                name_input = ui.input(
                    label="Variable name",
                    value=name,
                    placeholder="e.g. YEAR",
                    validation={
                        "Must be letters, digits, underscores (start with letter/_)":
                            self._validate_var_name
                    },
                ).classes("w-40").props("outlined dense")
                value_input = ui.input(
                    label="Value",
                    value=value,
                    placeholder="e.g. 2025",
                ).classes("flex-grow").props("outlined dense")
                ui.button(
                    icon="remove",
                    on_click=lambda r=row_data: self._remove_row(r),
                ).props("flat round size=sm color=negative")

        row_data["row_el"] = row_el
        row_data["name_input"] = name_input
        row_data["value_input"] = value_input
        self._rows.append(row_data)

    def _remove_row(self, row_data: dict[str, Any]) -> None:
        """Remove a variable row."""
        self._container.remove(row_data["row_el"])
        self._rows.remove(row_data)

    def get_variables(self) -> dict[str, str]:
        """Collect current variable name→value mappings (skipping empty names)."""
        variables: dict[str, str] = {}
        for row in self._rows:
            name = (row["name_input"].value or "").strip()
            value = row["value_input"].value or ""
            if name:
                variables[name] = value
        return variables

    def set_variables(self, variables: dict[str, str]) -> None:
        """Replace the current rows with the given variables dict."""
        # Clear existing rows
        self._container.clear()
        self._rows.clear()
        # Add rows for each variable
        for name, value in variables.items():
            self._add_row(name, value)


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
            # Load template variables from preset
            template_vars_editor.set_variables(preset.get("template_variables") or {})
            ui.notify(f"Loaded preset: {preset['name']}", type="info")

        async def save_preset() -> None:
            pipeline_settings = _collect_values(current_sections)
            template_variables = template_vars_editor.get_variables()
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
                            pipeline_name, name_input.value or "", pipeline_settings,
                            template_variables=template_variables,
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

        # --- Template Variables editor (between preset bar and form) ---
        template_vars_editor = _TemplateVariablesEditor()

        # Form area — use a column that we clear and rebuild
        form_container = ui.column().classes("w-full gap-1")

        def build_form() -> None:
            nonlocal current_sections
            form_container.clear()
            name = pipeline_select.value
            desc_label.text = schemas[name].get("description", "")
            schema_tree = schemas[name]["schema"]
            with form_container:
                current_sections = _render_steps(
                    schema_tree,
                    vars_getter=template_vars_editor.get_variables,
                )

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
            pipeline_name = pipeline_select.value

            # Verify all {VAR} references are defined in the variables table
            variables = template_vars_editor.get_variables()
            undefined = _find_undefined_variables(pipeline_settings, variables)
            if undefined:
                names = ", ".join(f"{{{v}}}" for v in sorted(undefined))
                ui.notify(
                    f"Undefined template variables: {names}",
                    type="negative",
                )
                return

            # Resolve template variables in settings
            if variables:
                pipeline_settings = _resolve_variables(pipeline_settings, variables)

            # Inject log level after variable resolution (not a user-configurable template field)
            pipeline_settings["_log_level"] = log_level_select.value

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
