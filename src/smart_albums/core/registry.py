"""Stage registry — decorator-based registration and config introspection."""

from __future__ import annotations

from typing import Any

from smart_albums.core.node import ConfigParam, CompositeParam, PipelineNode


STAGE_REGISTRY: dict[str, type[PipelineNode]] = {}


def stage(name: str, config: list[ConfigParam] | tuple[ConfigParam, ...] | None = None) -> Any:
    """Register a PipelineNode subclass under the given name.

    Can be used as a bare decorator (config defined on the class) or with
    an explicit config list passed to the decorator.
    """
    config_tuple = tuple(config) if config else ()

    def decorator(cls: type[PipelineNode]) -> type[PipelineNode]:
        if not issubclass(cls, PipelineNode):
            raise ValueError(
                f"Stage {name} must inherit from PipelineNode"
            )

        if name in STAGE_REGISTRY:
            raise ValueError(
                f"Duplicate stage registration: {name}"
            )

        cls._stage_name = name
        if config_tuple:
            cls._config_schema = config_tuple
        cls._description = cls.__doc__ or ""
        STAGE_REGISTRY[name] = cls

        return cls

    return decorator


def list_stages() -> list[str]:
    """Return all registered stage names, sorted alphabetically."""
    return sorted(STAGE_REGISTRY.keys())


def has_stage(name: str) -> bool:
    """Check whether a stage name is registered."""
    return name in STAGE_REGISTRY


def get_stage(name: str) -> type[PipelineNode] | None:
    """Return the registered stage class for the given name, or None."""
    return STAGE_REGISTRY.get(name)


def get_stage_config(name: str) -> tuple[ConfigParam | CompositeParam, ...]:
    """Return the config schema for a registered stage."""
    stage_cls = STAGE_REGISTRY.get(name)
    if stage_cls is None:
        return ()
    return stage_cls._config_schema


def get_stage_description(name: str) -> str:
    """Return the description string for a registered stage."""
    stage_cls = STAGE_REGISTRY.get(name)
    if stage_cls is None:
        return ""
    return stage_cls._description
