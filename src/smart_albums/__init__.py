"""smart_albums — a modular, provider-agnostic photo album curation library."""

from __future__ import annotations

from smart_albums.core.context import PipelineContext
from smart_albums.nodes.fork.fork import Fork, ForkBySelection
from protocols_system.protocols import Asset
from smart_albums.core.node import ConfigParam
from smart_albums.core.registry import (
    get_stage_config,
    get_stage_description,
    has_stage,
    list_stages,
    stage,
)
from smart_albums.core.runner import run_pipeline

__all__ = [
    "Asset",
    "ConfigParam",
    "Fork",
    "ForkBySelection",
    "PipelineContext",
    "get_stage_config",
    "get_stage_description",
    "has_stage",
    "list_stages",
    "run_pipeline",
    "stage",
]
