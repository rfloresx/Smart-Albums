"""User pipeline management — create, edit, delete custom pipeline definitions.

User pipelines are stored in the database and allow users to compose arbitrary
sequences of registered stages via the web UI. They are executed using the
CLI's ``run`` command, which accepts an inline pipeline definition in the
config JSON.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from smart_albums.cli.imports import import_all_stages
from smart_albums.core.registry import list_stages, has_stage, get_stage_config, get_stage_description
from smart_albums.core.node import ConfigParam

from webgui.state import state

logger = logging.getLogger(__name__)

# Prefix used to identify user pipelines in the pipeline selection UI.
# User pipeline names are stored as "user:<id>" to distinguish them from
# built-in registered pipelines.
USER_PIPELINE_PREFIX = "user:"


class UserPipelineError(ValueError):
    """Raised for invalid user pipeline operations."""


def get_available_stages() -> list[dict[str, Any]]:
    """Return all registered stages with their config schemas.

    Each item has: name, description, params (list of serialized ConfigParam).
    """
    import_all_stages()
    stages: list[dict[str, Any]] = []
    for name in list_stages():
        params = get_stage_config(name)
        serialized_params: list[dict[str, Any]] = []
        for p in params:
            if isinstance(p, ConfigParam):
                serialized_params.append({
                    "key": p.key,
                    "type": _type_to_str(p.type),
                    "default": p.default,
                    "required": p.required,
                    "description": p.description or None,
                    "min": p.min,
                    "max": p.max,
                    "choices": p.choices,
                })
        stages.append({
            "name": name,
            "description": get_stage_description(name),
            "params": serialized_params,
        })
    return stages


def _type_to_str(t: Any) -> str:
    """Convert a type annotation to a display string."""
    if isinstance(t, type):
        return t.__name__
    return str(t)


def validate_steps(steps: list[list[Any]]) -> list[str]:
    """Validate a pipeline step list, including nested fork branches.

    Args:
        steps: List of steps, each being [stage_name, config_dict] or
               [stage_name, config_dict, alias].

    Returns:
        List of validation error messages (empty if valid).
    """
    import_all_stages()
    errors: list[str] = []

    if not steps:
        errors.append("Pipeline must have at least one step.")
        return errors

    for i, step in enumerate(steps):
        if not isinstance(step, (list, tuple)):
            errors.append(f"Step {i + 1}: must be a list.")
            continue
        if len(step) < 2 or len(step) > 3:
            errors.append(f"Step {i + 1}: must have 2 or 3 elements [stage, config] or [stage, config, alias].")
            continue

        stage_name = step[0]
        config = step[1]

        if not isinstance(stage_name, str) or not stage_name.strip():
            errors.append(f"Step {i + 1}: stage name must be a non-empty string.")
            continue

        if not has_stage(stage_name):
            errors.append(f"Step {i + 1}: unknown stage '{stage_name}'.")
            continue

        if not isinstance(config, dict):
            errors.append(f"Step {i + 1}: config must be a dict.")
            continue

        if len(step) == 3:
            alias = step[2]
            if alias is not None and not isinstance(alias, str):
                errors.append(f"Step {i + 1}: alias must be a string or null.")

        # Validate fork branches recursively
        if stage_name in ("fork", "fork.by_selection") and "branches" in config:
            branch_errors = _validate_branches(config["branches"], prefix=f"Step {i + 1}")
            errors.extend(branch_errors)

    return errors


def _validate_branches(branches: Any, prefix: str) -> list[str]:
    """Validate fork branch definitions recursively.

    Accepts either:
    - A list of [name, pipeline_steps] pairs
    - A dict of {name: pipeline_steps}
    """
    errors: list[str] = []

    if isinstance(branches, list):
        if not branches:
            errors.append(f"{prefix}: fork must have at least one branch.")
            return errors
        for j, branch in enumerate(branches):
            if not isinstance(branch, (list, tuple)) or len(branch) != 2:
                errors.append(f"{prefix}, branch {j + 1}: must be [name, steps].")
                continue
            branch_name, branch_steps = branch
            if not isinstance(branch_name, str) or not branch_name.strip():
                errors.append(f"{prefix}, branch {j + 1}: name must be a non-empty string.")
                continue
            if not isinstance(branch_steps, list):
                errors.append(f"{prefix}, branch '{branch_name}': steps must be a list.")
                continue
            # Recursively validate the branch pipeline
            sub_errors = validate_steps(branch_steps)
            for err in sub_errors:
                errors.append(f"{prefix}, branch '{branch_name}': {err}")

    elif isinstance(branches, dict):
        if not branches:
            errors.append(f"{prefix}: fork must have at least one branch.")
            return errors
        for branch_name, branch_steps in branches.items():
            if not isinstance(branch_name, str) or not branch_name.strip():
                errors.append(f"{prefix}: branch name must be a non-empty string.")
                continue
            if not isinstance(branch_steps, list):
                errors.append(f"{prefix}, branch '{branch_name}': steps must be a list.")
                continue
            sub_errors = validate_steps(branch_steps)
            for err in sub_errors:
                errors.append(f"{prefix}, branch '{branch_name}': {err}")

    else:
        errors.append(f"{prefix}: 'branches' must be a list or dict.")

    return errors


async def list_user_pipelines() -> list[dict[str, Any]]:
    """Return all user-defined pipelines."""
    return await state.db.list_user_pipelines()


async def get_user_pipeline(pipeline_id: str) -> dict[str, Any] | None:
    """Fetch a single user pipeline by ID."""
    return await state.db.get_user_pipeline(pipeline_id)


async def create_user_pipeline(
    name: str,
    description: str,
    steps: list[list[Any]],
) -> dict[str, Any]:
    """Create a new user-defined pipeline.

    Args:
        name: Human-friendly pipeline name.
        description: What the pipeline does.
        steps: List of steps as [stage_name, config_dict, optional_alias].

    Returns:
        The created pipeline dict.

    Raises:
        UserPipelineError: If validation fails.
    """
    name = name.strip()
    if not name:
        raise UserPipelineError("Pipeline name cannot be empty.")

    errors = validate_steps(steps)
    if errors:
        raise UserPipelineError("\n".join(errors))

    pipeline_id = str(uuid.uuid4())[:8]
    await state.db.create_user_pipeline(pipeline_id, name, description.strip(), steps)
    logger.info("Created user pipeline %s: %s", pipeline_id, name)
    result = await state.db.get_user_pipeline(pipeline_id)
    assert result is not None
    return result


async def update_user_pipeline(
    pipeline_id: str,
    name: str,
    description: str,
    steps: list[list[Any]],
) -> None:
    """Update an existing user-defined pipeline.

    Raises:
        UserPipelineError: If the pipeline doesn't exist or validation fails.
    """
    existing = await state.db.get_user_pipeline(pipeline_id)
    if existing is None:
        raise UserPipelineError(f"Pipeline {pipeline_id!r} not found.")

    name = name.strip()
    if not name:
        raise UserPipelineError("Pipeline name cannot be empty.")

    errors = validate_steps(steps)
    if errors:
        raise UserPipelineError("\n".join(errors))

    await state.db.update_user_pipeline(pipeline_id, name, description.strip(), steps)
    logger.info("Updated user pipeline %s: %s", pipeline_id, name)


async def delete_user_pipeline(pipeline_id: str) -> None:
    """Delete a user pipeline.

    Raises:
        UserPipelineError: If the pipeline doesn't exist.
    """
    deleted = await state.db.delete_user_pipeline(pipeline_id)
    if not deleted:
        raise UserPipelineError(f"Pipeline {pipeline_id!r} not found.")
    logger.info("Deleted user pipeline %s", pipeline_id)


def is_user_pipeline(pipeline_name: str) -> bool:
    """Check whether a pipeline name refers to a user-defined pipeline."""
    return pipeline_name.startswith(USER_PIPELINE_PREFIX)


def get_user_pipeline_id(pipeline_name: str) -> str:
    """Extract the pipeline ID from a user pipeline name."""
    return pipeline_name[len(USER_PIPELINE_PREFIX):]
