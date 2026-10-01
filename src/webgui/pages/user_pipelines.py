"""User Pipelines page — create, view, edit, and delete custom pipelines.

Allows users to compose arbitrary sequences of registered stages into
reusable pipeline definitions that can be run from the Pipeline page
or attached to schedules. Supports nested fork/fork.by_selection branches.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from nicegui import ui

from webgui.components.navbar import build_navbar
from webgui.services import user_pipelines as user_pipelines_service

logger = logging.getLogger(__name__)

# Stages that use branches (composite/fork stages)
_FORK_STAGES = frozenset({"fork", "fork.by_selection"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_stage_options() -> dict[str, str]:
    """Return a dict of stage_name -> description for the stage selector."""
    stages = user_pipelines_service.get_available_stages()
    return {s["name"]: f"{s['name']} — {s['description'][:80]}" if s["description"] else s["name"] for s in stages}


def _get_stage_params(stage_name: str) -> list[dict[str, Any]]:
    """Return the parameter schema for a stage."""
    stages = user_pipelines_service.get_available_stages()
    for s in stages:
        if s["name"] == stage_name:
            return s["params"]
    return []


# ---------------------------------------------------------------------------
# Branch editor (nested pipeline within a fork step)
# ---------------------------------------------------------------------------


class _BranchEditor:
    """Editor for a single fork branch containing a nested pipeline."""

    def __init__(
        self,
        index: int,
        stage_options: dict[str, str],
        *,
        initial_name: str = "",
        initial_steps: list[list[Any]] | None = None,
        on_remove=None,
    ) -> None:
        self.index = index
        self._step_editors: list[_StepEditor] = []

        with ui.card().classes("w-full q-pa-sm").style(
            "border-left: 3px solid #1976d2"
        ):
            with ui.row().classes("w-full items-center justify-between"):
                with ui.row().classes("items-center gap-2"):
                    ui.icon("call_split", color="primary", size="sm")
                    self._name_input = ui.input(
                        label="Branch name *",
                        value=initial_name,
                        placeholder="e.g. curation",
                    ).classes("w-48").props("outlined dense")
                if on_remove:
                    ui.button(icon="delete", on_click=on_remove).props(
                        "flat round dense size=sm color=negative"
                    )

            self._steps_container = ui.column().classes("w-full gap-2 q-mt-xs q-pl-sm")
            self._stage_options = stage_options

            # Build initial steps
            self._rebuild_steps(initial_steps or [["", {}]])

            ui.button("Add Step", icon="add", on_click=self._add_step).props(
                "flat size=xs color=primary"
            ).classes("q-mt-xs")

    def _rebuild_steps(self, steps_data: list[list[Any]]) -> None:
        """Rebuild step editors from step data."""
        self._steps_container.clear()
        self._step_editors = []

        with self._steps_container:
            for i, step_data in enumerate(steps_data):
                stage = step_data[0] if len(step_data) > 0 else ""
                config = step_data[1] if len(step_data) > 1 else {}
                alias = step_data[2] if len(step_data) > 2 else ""

                # NOTE: remove/move handlers below collect via
                # ``_collect_steps_raw`` (not the filtering ``_collect_steps``),
                # so index `idx` here always matches the editor actually
                # sitting at UI position `idx` — including steps whose stage
                # hasn't been picked yet. Using the filtering collector would
                # silently drop those in-progress steps and make `pop(idx)`
                # remove the wrong one (WG-26).
                def make_remove(idx=i):
                    def remove():
                        current = self._collect_steps_raw()
                        current.pop(idx)
                        if not current:
                            current = [["", {}]]
                        self._rebuild_steps(current)
                    return remove

                def make_move_up(idx=i):
                    def move_up():
                        current = self._collect_steps_raw()
                        if idx > 0:
                            current[idx - 1], current[idx] = current[idx], current[idx - 1]
                            self._rebuild_steps(current)
                    return move_up

                def make_move_down(idx=i):
                    def move_down():
                        current = self._collect_steps_raw()
                        if idx < len(current) - 1:
                            current[idx], current[idx + 1] = current[idx + 1], current[idx]
                            self._rebuild_steps(current)
                    return move_down

                editor = _StepEditor(
                    index=i,
                    stage_options=self._stage_options,
                    initial_stage=stage,
                    initial_config=config if isinstance(config, dict) else {},
                    initial_alias=alias or "",
                    on_remove=make_remove(),
                    on_move_up=make_move_up() if i > 0 else None,
                    on_move_down=make_move_down() if i < len(steps_data) - 1 else None,
                    nested=True,
                )
                self._step_editors.append(editor)

    def _add_step(self) -> None:
        """Add a new empty step."""
        # Raw collection — using the filtering collector here would drop
        # any other in-progress (stage-not-yet-selected) step every time a
        # new one is added (WG-26).
        current = self._collect_steps_raw()
        current.append(["", {}])
        self._rebuild_steps(current)

    def _collect_steps(self) -> list[list[Any]]:
        """Collect steps from editors, dropping steps with no stage selected.

        Used only for the final save path (via ``get_branch``), where an
        incomplete step should be treated as "not part of the branch yet"
        rather than crashing. Reorder/remove handlers must NOT use this —
        see ``_collect_steps_raw``.
        """
        steps = []
        for editor in self._step_editors:
            step = editor.get_step()
            if step is not None:
                steps.append(step)
        return steps

    def _collect_steps_raw(self) -> list[list[Any]]:
        """Collect steps from editors without dropping incomplete ones.

        Keeps a 1:1 correspondence with ``self._step_editors`` (and thus
        with the UI), which is required for remove/move-by-index to target
        the right editor (WG-26).
        """
        return [editor.to_raw_list() for editor in self._step_editors]

    def get_branch(self) -> list[Any] | None:
        """Return [branch_name, steps] or None if invalid."""
        name = (self._name_input.value or "").strip()
        if not name:
            return None
        steps = self._collect_steps()
        return [name, steps]

    def get_branch_raw(self) -> list[Any]:
        """Return [branch_name, steps] reflecting current UI state verbatim.

        Unlike ``get_branch``, never signals "invalid" for a blank name —
        used by the parent step's remove/move handlers so popping/swapping
        by index stays aligned with the branch editors actually on screen,
        even while a branch name is still being typed (WG-26).
        """
        name = (self._name_input.value or "").strip()
        return [name, self._collect_steps_raw()]


# ---------------------------------------------------------------------------
# Step editor component
# ---------------------------------------------------------------------------


class _StepEditor:
    """Interactive editor for a single pipeline step, with fork support."""

    def __init__(
        self,
        index: int,
        stage_options: dict[str, str],
        *,
        initial_stage: str = "",
        initial_config: dict[str, Any] | None = None,
        initial_alias: str = "",
        on_remove=None,
        on_move_up=None,
        on_move_down=None,
        nested: bool = False,
    ) -> None:
        self.index = index
        self.stage_name = initial_stage
        self.config = initial_config or {}
        self.alias = initial_alias
        self._param_fields: dict[str, Any] = {}
        self._branch_editors: list[_BranchEditor] = []
        self._stage_options = stage_options
        self._nested = nested

        card_classes = "w-full q-pa-sm"
        if nested:
            card_classes += " q-pa-xs"

        with ui.card().classes(card_classes) as self.card:
            with ui.row().classes("w-full items-center justify-between"):
                with ui.row().classes("items-center gap-2"):
                    label_text = f"Step {index + 1}"
                    ui.label(label_text).classes("text-subtitle2 text-weight-bold")
                    self._alias_input = ui.input(
                        label="Alias (optional)",
                        value=initial_alias,
                        placeholder="e.g. my_filter",
                    ).classes("w-48").props("outlined dense")

                with ui.row().classes("items-center gap-1"):
                    if on_move_up:
                        ui.button(icon="arrow_upward", on_click=on_move_up).props(
                            "flat round dense size=sm"
                        )
                    if on_move_down:
                        ui.button(icon="arrow_downward", on_click=on_move_down).props(
                            "flat round dense size=sm"
                        )
                    if on_remove:
                        ui.button(icon="delete", on_click=on_remove).props(
                            "flat round dense size=sm color=negative"
                        )

            # `ui.select(value=...)` raises ValueError if the value isn't
            # among `options` — which happens for a step referencing a
            # stage that has since been removed/renamed in the registry
            # (e.g. after an upgrade). Fall back to unset rather than
            # crashing the whole editor (WG-27).
            initial_stage_value = (
                initial_stage if initial_stage in stage_options else None
            )
            self._stage_select = ui.select(
                options=stage_options,
                value=initial_stage_value,
                label="Stage",
                on_change=self._on_stage_change,
            ).classes("w-full").props("outlined dense")
            if initial_stage and initial_stage_value is None:
                ui.label(
                    f"⚠ Unknown stage '{initial_stage}' — select a replacement."
                ).classes("text-caption text-warning")

            self._params_container = ui.column().classes("w-full gap-1 q-mt-xs")
            self._branches_container = ui.column().classes("w-full gap-2 q-mt-xs")

            # Render params/branches if stage is already set
            if initial_stage:
                self._render_content(initial_stage)

    def _on_stage_change(self, e) -> None:
        """Rebuild when stage changes."""
        self.stage_name = e.value or ""
        self.config = {}
        self._render_content(self.stage_name)

    def _render_content(self, stage_name: str) -> None:
        """Render parameters or branch editors depending on stage type."""
        self._params_container.clear()
        self._branches_container.clear()
        self._param_fields.clear()
        self._branch_editors = []

        if not stage_name:
            return

        if stage_name in _FORK_STAGES:
            self._render_branches(stage_name)
        else:
            self._render_params(stage_name)

    def _render_branches(self, stage_name: str) -> None:
        """Render branch editors for fork stages."""
        branches_data = self.config.get("branches", [])

        with self._branches_container:
            ui.label("Branches").classes("text-caption text-weight-bold")
            ui.label(
                "Each branch runs its own sub-pipeline. "
                + ("Assets not selected by earlier branches pass to the next."
                   if stage_name == "fork.by_selection"
                   else "All branches receive a copy of the input.")
            ).classes("text-caption text-grey q-mb-xs")

            self._branches_list_container = ui.column().classes("w-full gap-2")
            self._rebuild_branches(branches_data or [])

            ui.button("Add Branch", icon="add", on_click=self._add_branch).props(
                "flat size=sm color=primary"
            ).classes("q-mt-xs")

    def _rebuild_branches(self, branches_data: list[Any]) -> None:
        """Rebuild branch editors from data."""
        self._branches_list_container.clear()
        self._branch_editors = []

        # Normalize branches_data
        if isinstance(branches_data, dict):
            branches_data = [[name, steps] for name, steps in branches_data.items()]

        if not branches_data:
            branches_data = [["branch_1", [["", {}]]]]

        with self._branches_list_container:
            for i, branch in enumerate(branches_data):
                if isinstance(branch, (list, tuple)) and len(branch) == 2:
                    branch_name, branch_steps = branch
                else:
                    branch_name = f"branch_{i + 1}"
                    branch_steps = [["", {}]]

                # Uses the raw (non-filtering) collector so `pop(idx)`
                # targets the branch editor actually at UI position `idx`,
                # even if an earlier branch currently has a blank name
                # (WG-26).
                def make_remove(idx=i):
                    def remove():
                        current = self._collect_branches_raw()
                        current.pop(idx)
                        self._rebuild_branches(current)
                    return remove

                editor = _BranchEditor(
                    index=i,
                    stage_options=self._stage_options,
                    initial_name=branch_name,
                    initial_steps=branch_steps if isinstance(branch_steps, list) else [],
                    on_remove=make_remove() if len(branches_data) > 1 else None,
                )
                self._branch_editors.append(editor)

    def _add_branch(self) -> None:
        """Add a new empty branch."""
        current = self._collect_branches_raw()
        n = len(current) + 1
        current.append([f"branch_{n}", [["", {}]]])
        self._rebuild_branches(current)

    def _collect_branches(self) -> list[list[Any]]:
        """Collect branch data from editors, dropping unnamed branches.

        Used only for the final save path. Reorder/remove handlers must
        use ``_collect_branches_raw`` instead — see WG-26.
        """
        branches = []
        for editor in self._branch_editors:
            branch = editor.get_branch()
            if branch is not None:
                branches.append(branch)
        return branches

    def _collect_branches_raw(self) -> list[list[Any]]:
        """Collect branch data from editors without dropping unnamed ones.

        Keeps 1:1 correspondence with ``self._branch_editors`` so
        remove-by-index stays aligned with what's on screen (WG-26).
        """
        return [editor.get_branch_raw() for editor in self._branch_editors]

    def _render_params(self, stage_name: str) -> None:
        """Render parameter input fields for the selected stage."""
        self._params_container.clear()
        self._param_fields.clear()

        params = _get_stage_params(stage_name)
        if not params:
            with self._params_container:
                ui.label("No configurable parameters.").classes("text-caption text-grey")
            return

        with self._params_container:
            for param in params:
                key = param["key"]
                default = param.get("default")
                current_value = self.config.get(key, default)
                label = key.replace("_", " ").title()
                desc = param.get("description") or ""
                required = param.get("required", False)

                if required:
                    label += " *"

                type_str = param.get("type", "str").lower()

                if param.get("choices"):
                    choices = param["choices"]
                    element = ui.select(
                        options=choices,
                        value=current_value if current_value in choices else None,
                        label=label,
                    ).classes("w-full").props("outlined dense clearable")
                elif "bool" in type_str:
                    element = ui.switch(
                        label, value=bool(current_value) if current_value is not None else False
                    )
                elif "int" in type_str or "float" in type_str:
                    step = 0.01 if "float" in type_str else 1
                    element = ui.number(
                        label=label,
                        value=current_value,
                        min=param.get("min"),
                        max=param.get("max"),
                        step=step,
                    ).classes("w-full").props("outlined dense")
                else:
                    element = ui.input(
                        label=label,
                        value=str(current_value) if current_value is not None else "",
                    ).classes("w-full").props("outlined dense")

                if desc:
                    ui.label(desc).classes("text-caption text-grey q-mt-none q-mb-xs")

                self._param_fields[key] = (element, param)

    def get_step(self) -> list[Any] | None:
        """Return the step definition, or None if invalid.

        For fork stages, the config includes a "branches" key with nested pipelines.
        """
        stage = self._stage_select.value
        if not stage:
            return None

        config: dict[str, Any] = {}

        if stage in _FORK_STAGES:
            # Collect branches
            branches = self._collect_branches()
            if branches:
                config["branches"] = branches
        else:
            # Collect param fields
            for key, (element, param) in self._param_fields.items():
                raw = element.value
                if raw is None or raw == "":
                    continue
                type_str = param.get("type", "str").lower()
                if "bool" in type_str:
                    config[key] = bool(raw)
                elif "int" in type_str:
                    try:
                        config[key] = int(raw)
                    except (TypeError, ValueError):
                        pass
                elif "float" in type_str:
                    try:
                        config[key] = float(raw)
                    except (TypeError, ValueError):
                        pass
                else:
                    config[key] = raw

        alias = (self._alias_input.value or "").strip()
        if alias:
            return [stage, config, alias]
        return [stage, config]

    def to_raw_list(self) -> list[Any]:
        """Return this step's current UI state verbatim, even if incomplete.

        Unlike ``get_step`` (which returns ``None`` for a step with no
        stage selected yet, so it can be filtered out at save time), this
        always returns a step entry — used by remove/move handlers that
        need their by-index collection to match the on-screen editor list
        1:1 (WG-26).
        """
        stage = self._stage_select.value or ""
        if stage in _FORK_STAGES:
            config = {"branches": self._collect_branches_raw()}
        else:
            config = self.get_step()[1] if stage else dict(self.config)
        alias = (self._alias_input.value or "").strip()
        if alias:
            return [stage, config, alias]
        return [stage, config]


# ---------------------------------------------------------------------------
# Pipeline editor dialog
# ---------------------------------------------------------------------------


async def _open_pipeline_editor(
    *,
    pipeline: dict[str, Any] | None = None,
    on_save,
) -> None:
    """Open a full-screen dialog for creating/editing a user pipeline."""
    is_edit = pipeline is not None
    title = "Edit Pipeline" if is_edit else "Create Pipeline"

    stage_options = _get_stage_options()
    step_editors: list[_StepEditor] = []

    with ui.dialog() as dlg, ui.card().classes("w-full max-w-4xl").style(
        "max-height: 90vh; overflow-y: auto"
    ):
        ui.label(title).classes("text-h5")

        name_input = ui.input(
            label="Pipeline Name *",
            value=pipeline["name"] if pipeline else "",
            placeholder="e.g. My Custom Pipeline",
        ).classes("w-full").props("outlined dense")

        desc_input = ui.input(
            label="Description",
            value=pipeline["description"] if pipeline else "",
            placeholder="What this pipeline does...",
        ).classes("w-full").props("outlined dense")

        ui.separator().classes("q-my-sm")
        ui.label("Pipeline Steps").classes("text-subtitle1 text-weight-medium")
        ui.label(
            "Add stages in the order they should execute. "
            "Use fork/fork.by_selection stages to create branching pipelines."
        ).classes("text-caption text-grey q-mb-sm")

        steps_container = ui.column().classes("w-full gap-2")

        def rebuild_steps(initial_steps: list[list[Any]] | None = None) -> None:
            """Rebuild all step editor cards."""
            nonlocal step_editors
            steps_container.clear()
            step_editors = []

            steps_data = initial_steps or []

            with steps_container:
                for i, step_data in enumerate(steps_data):
                    stage = step_data[0] if len(step_data) > 0 else ""
                    config = step_data[1] if len(step_data) > 1 else {}
                    alias = step_data[2] if len(step_data) > 2 else ""

                    # Reorder/remove use the raw (non-filtering) collector
                    # so index `idx` always matches the editor at UI
                    # position `idx`, including steps with no stage chosen
                    # yet — the filtering collector would silently drop
                    # those and make `pop(idx)` act on the wrong step, or
                    # raise IndexError once enough steps are incomplete
                    # (WG-26).
                    def make_remove(idx=i):
                        def remove():
                            current = _collect_steps_raw()
                            current.pop(idx)
                            rebuild_steps(current)
                        return remove

                    def make_move_up(idx=i):
                        def move_up():
                            current = _collect_steps_raw()
                            if idx > 0:
                                current[idx - 1], current[idx] = current[idx], current[idx - 1]
                                rebuild_steps(current)
                        return move_up

                    def make_move_down(idx=i):
                        def move_down():
                            current = _collect_steps_raw()
                            if idx < len(current) - 1:
                                current[idx], current[idx + 1] = current[idx + 1], current[idx]
                                rebuild_steps(current)
                        return move_down

                    editor = _StepEditor(
                        index=i,
                        stage_options=stage_options,
                        initial_stage=stage,
                        initial_config=config if isinstance(config, dict) else {},
                        initial_alias=alias or "",
                        on_remove=make_remove(),
                        on_move_up=make_move_up() if i > 0 else None,
                        on_move_down=make_move_down() if i < len(steps_data) - 1 else None,
                    )
                    step_editors.append(editor)

        def _collect_steps() -> list[list[Any]]:
            """Collect current step definitions, dropping incomplete steps.

            Used only on the final save path — see ``_collect_steps_raw``
            for why reorder/remove handlers must not use this (WG-26).
            """
            steps = []
            for editor in step_editors:
                step = editor.get_step()
                if step is not None:
                    steps.append(step)
            return steps

        def _collect_steps_raw() -> list[list[Any]]:
            """Collect step definitions 1:1 with the on-screen editors."""
            return [editor.to_raw_list() for editor in step_editors]

        def add_step() -> None:
            """Add a new empty step at the end."""
            current = _collect_steps_raw()
            current.append(["", {}])
            rebuild_steps(current)

        # Initialize with existing steps or one empty step
        initial = pipeline["steps"] if pipeline else [["", {}]]
        rebuild_steps(initial)

        ui.button("Add Step", icon="add", on_click=add_step).props(
            "flat color=primary"
        ).classes("q-mt-sm")

        ui.separator().classes("q-my-md")

        # JSON import/export
        with ui.expansion("📋 JSON Import/Export", icon="code").classes("w-full"):
            json_area = ui.textarea(
                label="Pipeline steps as JSON",
                value=json.dumps(pipeline["steps"], indent=2) if pipeline else "[]",
            ).classes("w-full").props("outlined").style("font-family: monospace")

            def _validate_steps_structure(steps: Any, *, path: str = "steps") -> str | None:
                """Return an error message if ``steps`` isn't a well-formed step list.

                Previously only ``json.JSONDecodeError`` was caught around
                ``rebuild_steps``, so syntactically valid but structurally
                wrong JSON (e.g. a list of strings, or a branch missing its
                steps list) propagated into ``_StepEditor``/``_BranchEditor``
                construction and crashed the page instead of showing a
                notification (WG-27).
                """
                if not isinstance(steps, list):
                    return f"{path} must be a list"
                for i, step in enumerate(steps):
                    if not isinstance(step, (list, tuple)):
                        return f"{path}[{i}] must be a list of [stage, config] (or with alias)"
                    if len(step) < 1 or len(step) > 3:
                        return f"{path}[{i}] must have 1 to 3 elements: [stage, config, alias?]"
                    if not isinstance(step[0], str):
                        return f"{path}[{i}][0] (stage name) must be a string"
                    if len(step) > 1 and not isinstance(step[1], dict):
                        return f"{path}[{i}][1] (config) must be an object"
                    if len(step) > 2 and not isinstance(step[2], str):
                        return f"{path}[{i}][2] (alias) must be a string"
                    config = step[1] if len(step) > 1 and isinstance(step[1], dict) else {}
                    if step[0] in _FORK_STAGES and "branches" in config:
                        branches = config["branches"]
                        if isinstance(branches, dict):
                            branches = list(branches.items())
                        if not isinstance(branches, list):
                            return f"{path}[{i}].config.branches must be a list"
                        for j, branch in enumerate(branches):
                            if (
                                not isinstance(branch, (list, tuple))
                                or len(branch) != 2
                                or not isinstance(branch[0], str)
                            ):
                                return (
                                    f"{path}[{i}].config.branches[{j}] must be "
                                    "[branch_name, steps]"
                                )
                            err = _validate_steps_structure(
                                branch[1], path=f"{path}[{i}].branches[{j}].steps"
                            )
                            if err:
                                return err
                return None

            def import_json() -> None:
                try:
                    steps = json.loads(json_area.value or "[]")
                except json.JSONDecodeError as e:
                    ui.notify(f"Invalid JSON: {e}", type="negative")
                    return

                error = _validate_steps_structure(steps)
                if error:
                    ui.notify(f"Invalid pipeline structure: {error}", type="negative")
                    return

                try:
                    rebuild_steps(steps)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("Failed to rebuild steps from imported JSON")
                    ui.notify(f"Could not apply imported steps: {exc}", type="negative")
                    return
                ui.notify("Steps imported from JSON", type="positive")

            def export_json() -> None:
                steps = _collect_steps()
                json_area.value = json.dumps(steps, indent=2)
                ui.notify("Steps exported to JSON", type="info")

            with ui.row().classes("gap-2 q-mt-sm"):
                ui.button("Import", icon="upload", on_click=import_json).props(
                    "flat size=sm"
                )
                ui.button("Export", icon="download", on_click=export_json).props(
                    "flat size=sm"
                )

        # Error display
        error_label = ui.label("").classes("text-negative")
        error_label.set_visibility(False)

        async def do_save() -> None:
            # Guard against double-submit (WG-23).
            if save_btn.props.get("disable"):
                return
            save_btn.props("disable loading")
            try:
                error_label.set_visibility(False)
                steps = _collect_steps()
                name = (name_input.value or "").strip()
                description = desc_input.value or ""

                # The dynamic form has no real client-side validation
                # (missing names/stages were only cosmetically flagged, if
                # at all) — enforce the actual required fields here before
                # calling the service, instead of letting an incomplete
                # pipeline save silently (WG-27).
                if not name:
                    error_label.text = "Pipeline name is required."
                    error_label.set_visibility(True)
                    return
                if not steps:
                    error_label.text = (
                        "Add at least one complete step (with a stage selected)."
                    )
                    error_label.set_visibility(True)
                    return

                try:
                    if is_edit:
                        await user_pipelines_service.update_user_pipeline(
                            pipeline["id"], name, description, steps
                        )
                        ui.notify("Pipeline updated", type="positive")
                    else:
                        await user_pipelines_service.create_user_pipeline(
                            name, description, steps
                        )
                        ui.notify("Pipeline created", type="positive")
                    dlg.close()
                    await on_save()
                except user_pipelines_service.UserPipelineError as exc:
                    error_label.text = str(exc)
                    error_label.set_visibility(True)
            finally:
                save_btn.props(remove="disable loading")

        with ui.row().classes("w-full justify-end gap-2 q-mt-md"):
            ui.button("Cancel", on_click=dlg.close).props("flat")
            save_btn = ui.button(
                "Save" if is_edit else "Create",
                on_click=do_save,
            ).props("color=primary")

    dlg.open()


# ---------------------------------------------------------------------------
# Pipeline card
# ---------------------------------------------------------------------------


def _summarize_steps(steps: list[Any], depth: int = 0) -> list[tuple[str, int]]:
    """Flatten steps into (label, depth) tuples for display, including branches."""
    items: list[tuple[str, int]] = []
    for step in steps:
        if not isinstance(step, (list, tuple)) or len(step) < 2:
            continue
        stage_name = step[0]
        config = step[1] if len(step) > 1 else {}
        alias = step[2] if len(step) > 2 else None
        label = alias or stage_name
        items.append((label, depth))

        # Recurse into fork branches
        if stage_name in _FORK_STAGES and isinstance(config, dict):
            branches = config.get("branches", [])
            if isinstance(branches, list):
                for branch in branches:
                    if isinstance(branch, (list, tuple)) and len(branch) == 2:
                        branch_name, branch_steps = branch
                        items.append((f"↳ {branch_name}", depth + 1))
                        items.extend(_summarize_steps(branch_steps, depth + 2))
    return items


def _build_pipeline_card(pipeline: dict[str, Any], *, on_refresh) -> None:
    """Render a single user pipeline card."""
    steps = pipeline.get("steps", [])
    summary = _summarize_steps(steps)

    with ui.card().classes("w-full"):
        with ui.row().classes("w-full items-center justify-between"):
            with ui.row().classes("items-center gap-2"):
                ui.icon("account_tree", color="primary")
                ui.label(pipeline["name"]).classes("text-subtitle1 text-weight-medium")
                ui.badge(f"{len(steps)} steps").props("outline")
            ui.label(f"#{pipeline['id']}").classes("text-caption text-grey")

        if pipeline.get("description"):
            ui.label(pipeline["description"]).classes("text-caption text-grey q-mt-xs")

        # Step summary with branch visualization
        if summary:
            with ui.column().classes("w-full gap-0 q-mt-sm"):
                for label, depth in summary[:12]:
                    indent = "  " * depth
                    style = "text-caption" + (" text-grey" if depth > 0 else "")
                    ui.label(f"{indent}{'→ ' if depth == 0 else ''}{label}").classes(style)
                if len(summary) > 12:
                    ui.label(f"  … +{len(summary) - 12} more").classes("text-caption text-grey")

        # Actions
        with ui.row().classes("w-full justify-end gap-2 q-mt-sm"):

            async def do_run(p=pipeline) -> None:
                from webgui.services import jobs as jobs_service

                # Guard against double-submit (WG-23).
                if run_btn.props.get("disable"):
                    return
                run_btn.props("disable loading")
                try:
                    pipeline_key = f"{user_pipelines_service.USER_PIPELINE_PREFIX}{p['id']}"
                    record = await jobs_service.create_job(pipeline_key, {})
                    ui.notify(f"Job {record.id} started", type="positive")
                    ui.navigate.to("/jobs")
                except Exception as exc:
                    ui.notify(f"Failed: {exc}", type="negative")
                finally:
                    run_btn.props(remove="disable loading")

            async def do_edit(p=pipeline) -> None:
                await _open_pipeline_editor(pipeline=p, on_save=on_refresh)

            async def do_delete(p=pipeline) -> None:
                await _confirm_delete(p, on_refresh=on_refresh)

            run_btn = ui.button(
                "Run", icon="play_arrow", on_click=do_run
            ).props("flat size=sm color=primary")
            ui.button(
                "Edit", icon="edit", on_click=do_edit
            ).props("flat size=sm")
            ui.button(
                "Delete", icon="delete", on_click=do_delete
            ).props("flat size=sm color=negative")


async def _confirm_delete(pipeline: dict[str, Any], *, on_refresh) -> None:
    """Show a confirmation dialog before deleting a pipeline."""
    with ui.dialog() as dlg, ui.card().classes("w-80"):
        ui.label("Delete Pipeline").classes("text-h6")
        ui.label(
            f'Are you sure you want to delete "{pipeline["name"]}"? '
            "This action cannot be undone."
        ).classes("q-mt-sm")

        async def do_delete() -> None:
            try:
                await user_pipelines_service.delete_user_pipeline(pipeline["id"])
                ui.notify("Pipeline deleted", type="info")
                dlg.close()
                await on_refresh()
            except user_pipelines_service.UserPipelineError as exc:
                ui.notify(str(exc), type="negative")

        with ui.row().classes("w-full justify-end gap-2 q-mt-md"):
            ui.button("Cancel", on_click=dlg.close).props("flat")
            ui.button("Delete", on_click=do_delete).props("color=negative")

    dlg.open()


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


@ui.page("/pipelines")
async def user_pipelines_page() -> None:
    """User pipeline management page."""
    await ui.context.client.connected()
    ui.dark_mode(True)
    build_navbar()

    container = ui.column().classes("w-full max-w-4xl mx-auto q-pa-md gap-3")

    async def refresh() -> None:
        """Rebuild the pipeline list."""
        container.clear()
        pipelines = await user_pipelines_service.list_user_pipelines()
        with container:
            with ui.row().classes("w-full items-center justify-between"):
                ui.label("Custom Pipelines").classes("text-h4")
                ui.button(
                    "New Pipeline",
                    icon="add",
                    on_click=lambda: _open_pipeline_editor(on_save=refresh),
                ).props("color=primary")

            ui.label(
                "Create custom pipelines by combining available stages. "
                "Use fork stages to create branching workflows. "
                "Custom pipelines can be used from the Pipeline page or in schedules."
            ).classes("text-caption text-grey")

            if not pipelines:
                with ui.card().classes("w-full q-pa-lg items-center"):
                    ui.icon("account_tree", size="48px", color="grey")
                    ui.label("No custom pipelines yet").classes(
                        "text-subtitle1 text-grey q-mt-sm"
                    )
                    ui.label(
                        "Create a pipeline to compose stages into reusable workflows."
                    ).classes("text-caption text-grey")
                return

            for pipeline in pipelines:
                _build_pipeline_card(pipeline, on_refresh=refresh)

    await refresh()
