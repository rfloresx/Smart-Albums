"""Pipeline registry — register and look up named pipeline definitions.

Provides the ``@register_pipeline`` decorator for declaring named pipelines
and factory functions for retrieving them at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from smart_albums.core.spec import Pipeline


@dataclass(frozen=True)
class PipelineInfo:
    """Metadata and definition for a registered pipeline."""

    label: str = ""
    description: str = ""
    pipeline: Pipeline = field(default_factory=list)


_REGISTRY: dict[str, PipelineInfo] = {}


def get_pipelines() -> list[str]:
    """Return all registered pipeline names."""
    return list(_REGISTRY.keys())


def get_pipeline(name: str) -> Optional[PipelineInfo]:
    """Return the PipelineInfo for *name*, or None if not registered."""
    return _REGISTRY.get(name)


def register_pipeline(
    name: str,
    *,
    label: str,
    description: str,
) -> Callable[[Callable[[], Pipeline]], Callable[[], Pipeline]]:
    """Register a pipeline definition under *name*.

    Usage::

        @register_pipeline("best_of_year", label="Best of Year", description="...")
        def best_of_year_pipeline() -> Pipeline:
            return [ ... ]
    """
    def decorator(fn: Callable[[], Pipeline]) -> Callable[[], Pipeline]:
        if name in _REGISTRY:
            raise ValueError(f"Pipeline {name} already registered")
        pipeline = fn if isinstance(fn, list) else fn()
        _REGISTRY[name] = PipelineInfo(
            label=label,
            description=description,
            pipeline=pipeline,
        )
        return fn
    return decorator
