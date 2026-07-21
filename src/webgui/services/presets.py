"""Preset management — save and load pipeline configuration presets.

Presets are named snapshots of ``pipeline_settings`` for a specific pipeline.
They are persisted in the ``presets`` database table and allow users to quickly
recall configurations they use frequently (e.g. "Best of 2025 — dry run" vs
"Best of 2025 — production").
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from webgui.state import state

logger = logging.getLogger(__name__)


class PresetError(ValueError):
    """Raised for invalid preset operations."""


async def list_presets(pipeline: str | None = None) -> list[dict[str, Any]]:
    """Return presets for a pipeline (or all pipelines if None).

    Each item has: id, pipeline, name, pipeline_settings, created_at, updated_at.
    """
    return await state.db.list_presets(pipeline)


async def get_preset(preset_id: str) -> dict[str, Any] | None:
    """Fetch a single preset by ID."""
    return await state.db.get_preset(preset_id)


async def save_preset(
    pipeline: str,
    name: str,
    pipeline_settings: dict[str, Any],
) -> dict[str, Any]:
    """Create a new preset.

    Args:
        pipeline: The pipeline name this preset belongs to.
        name: A human-friendly label for the preset.
        pipeline_settings: The alias-keyed config overrides to save.

    Returns:
        The created preset dict.

    Raises:
        PresetError: If the name is empty.
    """
    name = name.strip()
    if not name:
        raise PresetError("Preset name cannot be empty.")

    preset_id = str(uuid.uuid4())[:8]
    await state.db.create_preset(preset_id, pipeline, name, pipeline_settings)
    logger.info("Created preset %s for pipeline %s", preset_id, pipeline)
    return (await state.db.get_preset(preset_id))  # type: ignore[return-value]


async def update_preset(preset_id: str, pipeline_settings: dict[str, Any]) -> None:
    """Overwrite a preset's pipeline_settings.

    Args:
        preset_id: The preset ID to update.
        pipeline_settings: The new settings dict.

    Raises:
        PresetError: If the preset doesn't exist.
    """
    existing = await state.db.get_preset(preset_id)
    if existing is None:
        raise PresetError(f"Preset {preset_id!r} not found.")
    await state.db.update_preset(preset_id, pipeline_settings)
    logger.info("Updated preset %s", preset_id)


async def delete_preset(preset_id: str) -> None:
    """Delete a preset.

    Args:
        preset_id: The preset ID to delete.

    Raises:
        PresetError: If the preset doesn't exist.
    """
    deleted = await state.db.delete_preset(preset_id)
    if not deleted:
        raise PresetError(f"Preset {preset_id!r} not found.")
    logger.info("Deleted preset %s", preset_id)
