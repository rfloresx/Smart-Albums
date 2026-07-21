"""Core framework: models, context, protocols, registry, and runner."""

from __future__ import annotations

from smart_albums.core.builder import build_pipeline, build_alias_map, resolve_overrides
from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.models import Asset
from smart_albums.core.node import ConfigParam, PipelineNode, Stage

from smart_albums.core.protocols import (
    AlbumResult,
    AlbumSummary,
    IImageClient,
    ILLMClient,
    IProgressReporter,
)
from smart_albums.core.registry import (
    STAGE_REGISTRY,
    stage,
    list_stages,
    has_stage,
    get_stage,
    get_stage_config,
    get_stage_description,
)
from smart_albums.core.runner import run_pipeline
from smart_albums.core.spec import AliasedStep, StepSpec, StepInput, Pipeline, PipelineSpec, alias

__all__ = [
    "AliasedStep",
    "alias",
    "Asset",
    "PipelineContext",
    "ContextBatch",
    "ConfigParam",
    "PipelineNode",
    "Stage",
    "AlbumResult",
    "AlbumSummary",
    "IImageClient",
    "ILLMClient",
    "IProgressReporter",
    "STAGE_REGISTRY",
    "stage",
    "list_stages",
    "has_stage",
    "get_stage",
    "get_stage_config",
    "get_stage_description",
    "build_pipeline",
    "build_alias_map",
    "resolve_overrides",
    "run_pipeline",
    "StepSpec",
    "StepInput",
    "Pipeline",
    "PipelineSpec",
]
