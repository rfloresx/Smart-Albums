"""Pipeline builder — normalizes step inputs and resolves config overrides.

Provides functions to:
- Normalize heterogeneous step inputs (strings, classes, tuples, instances)
  into a uniform ``PipelineSpec`` (list of ``StepSpec``).
- Build alias maps for config resolution.
- Apply config overrides keyed by alias to pipeline definitions.
- Introspect ``__init__`` signatures for dynamic CLI generation.
"""

from __future__ import annotations

from smart_albums.core.node import PipelineNode, ConfigParam, CompositeParam, ResolvedComposite
from smart_albums.core.spec import AliasedStep, Pipeline, PipelineSpec, StepInput, StepSpec

import smart_albums.core.registry as registry

from collections import defaultdict
from inspect import signature, Parameter
from typing import Any

PipelineConfigSchema = list[
    tuple[Any, dict[str, "ConfigParam | CompositeParam | ResolvedComposite"]]
]

def normalize_step(step: StepInput) -> StepSpec:

    # Unwrap AliasedStep — preserve alias on the resulting StepSpec
    step_alias: str | None = None
    if isinstance(step, AliasedStep):
        step_alias = step.alias
        step = step.step

    if isinstance(step, StepSpec):
        if step_alias and not step.alias:
            return StepSpec(
                name=step.name,
                config=step.config,
                instance=step.instance,
                alias=step_alias,
                children=step.children,
            )
        return step

    if isinstance(step, str):
        if not registry.has_stage(step):
            raise ValueError(
                f"Unknown stage: {step}"
            )

        return StepSpec(step, alias=step_alias)

    if (isinstance(step, type) and issubclass(step, PipelineNode)):
        name = step._stage_name
        if not registry.has_stage(name):
            raise ValueError(
                f"Unregistered stage class: {step.__name__}"
            )

        return StepSpec(name, alias=step_alias)

    # Pre-instantiated node object — pass through directly
    if isinstance(step, PipelineNode):
        return StepSpec(name=step.name, instance=step, alias=step_alias)

    if isinstance(step, tuple):

        name, config = step

        if not registry.has_stage(name):
            raise ValueError(
                f"Unknown stage: {name}"
            )

        return StepSpec(name, config, alias=step_alias)

    raise TypeError(
        f"Invalid pipeline step: {step!r}"
    )


def build_pipeline(pipeline: Pipeline) -> PipelineSpec:
    return [
        normalize_step(step)
        for step in pipeline
    ]

def get_pipeline_config_schema(pipeline: Pipeline | PipelineSpec) -> PipelineConfigSchema:
    specs: PipelineSpec
    if pipeline and not isinstance(pipeline[0], StepSpec):
        specs = build_pipeline(pipeline)  # type: ignore[arg-type]
    else:
        specs = pipeline  # type: ignore[assignment]


    def step_config(step: StepSpec) -> dict[str, ConfigParam | CompositeParam | ResolvedComposite]:
        params: tuple[ConfigParam | CompositeParam, ...] = registry.get_stage_config(step.name)
        config: dict[str, ConfigParam | CompositeParam | ResolvedComposite] = {}
        for param in params:
            if isinstance(param, CompositeParam) and step.instance is not None:
                # Resolve child pipelines from the instance and recurse
                children = param.resolve(step.instance)
                resolved = ResolvedComposite(
                    key=param.key,
                    description=param.description,
                    children=[
                        (name, get_pipeline_config_schema(child_pipeline))
                        for name, child_pipeline in children
                    ],
                )
                config[param.key] = resolved
            else:
                config[param.key] = param
        return config

    return [((step.name, step.alias), step_config(step)) for step in specs]


def get_init_schema(cls: type) -> dict[str, ConfigParam]:
    sig = signature(cls.__init__)  # type: ignore[misc]

    schema: dict[str, ConfigParam] = {}

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        # Skip *args and **kwargs — they cannot be mapped to config params
        if param.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD):
            continue

        schema[name] = ConfigParam(
            key=name,
            type=param.annotation if param.annotation is not Parameter.empty else Any,
            default=None if param.default is Parameter.empty else param.default,
            required=param.default is Parameter.empty,
        )

    return schema


# ---------------------------------------------------------------------------
# Alias-based config resolution
# ---------------------------------------------------------------------------


def _extract_stage_name(step: StepInput) -> str:
    """Extract the stage registry name from any step representation."""
    if isinstance(step, AliasedStep):
        return _extract_stage_name(step.step)
    if isinstance(step, str):
        return step
    if isinstance(step, type) and issubclass(step, PipelineNode):
        return step._stage_name
    if isinstance(step, PipelineNode):
        return step.name
    if isinstance(step, tuple):
        return step[0]
    if isinstance(step, StepSpec):
        return step.name
    raise TypeError(f"Cannot extract stage name from: {step!r}")


def _extract_config(step: StepInput) -> dict[str, Any]:
    """Extract existing config dict from any step representation."""
    if isinstance(step, AliasedStep):
        return _extract_config(step.step)
    if isinstance(step, tuple) and len(step) == 2:
        return step[1]
    if isinstance(step, StepSpec):
        return dict(step.config)
    return {}


def build_alias_map(pipeline: Pipeline) -> dict[str, int]:
    """Build a mapping from alias → pipeline index.

    Steps with explicit aliases use those. Steps without an alias get
    an auto-generated one from their stage name. Repeated stage names
    receive an ordinal suffix (_2, _3, …).

    Raises ValueError on alias collisions.
    """
    alias_map: dict[str, int] = {}
    auto_counts: dict[str, int] = defaultdict(int)

    for i, step in enumerate(pipeline):
        if isinstance(step, AliasedStep):
            name = step.alias
            if name in alias_map:
                raise ValueError(
                    f"Duplicate alias '{name}' at pipeline index {i} "
                    f"(first seen at index {alias_map[name]})"
                )
            alias_map[name] = i
        else:
            # Auto-alias from stage name
            stage_name = _extract_stage_name(step)
            auto_counts[stage_name] += 1
            count = auto_counts[stage_name]
            auto_alias = stage_name if count == 1 else f"{stage_name}_{count}"
            alias_map[auto_alias] = i

    return alias_map


def resolve_overrides(
    pipeline: Pipeline,
    overrides: dict[str, dict[str, Any]],
) -> Pipeline:
    """Apply config overrides keyed by alias to a pipeline definition.

    Args:
        pipeline: The pipeline definition (list of steps, possibly with aliases).
        overrides: A dict mapping alias → config dict to merge into that step.

    Returns:
        A new pipeline list with overrides applied. Original is not mutated.

    Raises:
        ValueError: If an override key doesn't match any alias.
    """
    if not overrides:
        return pipeline

    alias_map = build_alias_map(pipeline)

    # Separate top-level overrides from dot-path overrides into composites
    # e.g. "output.branches.curation.publish" targets a child pipeline
    direct_overrides: dict[str, dict[str, Any]] = {}
    composite_overrides: dict[str, dict[str, Any]] = {}

    for key, config in overrides.items():
        # Check if the full key is a known alias first
        if key in alias_map:
            direct_overrides[key] = config
        else:
            # Try splitting at the first dot to find a root alias
            parts = key.split(".", 1)
            root = parts[0]
            if root in alias_map:
                composite_overrides.setdefault(root, {})[key] = config
            else:
                # Collect available aliases for error message
                available = sorted(alias_map.keys())
                raise ValueError(
                    f"Unknown alias '{key}'. "
                    f"Available aliases: {available}"
                )

    result = list(pipeline)

    # Apply direct overrides
    for alias_key, config in direct_overrides.items():
        idx = alias_map[alias_key]
        step = result[idx]

        # Unwrap AliasedStep to get the raw step
        original_alias = None
        raw_step = step
        if isinstance(step, AliasedStep):
            original_alias = step.alias
            raw_step = step.step

        name = _extract_stage_name(raw_step)
        existing_config = _extract_config(raw_step)
        merged = {**existing_config, **config}

        # Rebuild as (name, config) tuple wrapped in AliasedStep
        new_step: StepInput = (name, merged)
        if original_alias is not None:
            new_step = AliasedStep(alias=original_alias, step=new_step)

        result[idx] = new_step

    # Apply composite overrides (dot-path into child pipelines of composite nodes)
    for root_alias, sub_overrides in composite_overrides.items():
        idx = alias_map[root_alias]
        step = result[idx]

        raw_step = step
        original_alias = None
        if isinstance(step, AliasedStep):
            original_alias = step.alias
            raw_step = step.step

        # For composite nodes (PipelineNode instances with CompositeParam),
        # we pass the sub-overrides through as nested config
        if isinstance(raw_step, PipelineNode):
            from smart_albums.core.node import CompositeParam
            composite_params = [
                p for p in raw_step._config_schema
                if isinstance(p, CompositeParam) and p.resolve is not None
            ]
            if composite_params:
                param = composite_params[0]
                children = param.resolve(raw_step)
                prefix = f"{root_alias}."
                # Route overrides to the appropriate child pipeline
                updated_children: list[tuple[str, Pipeline]] = []
                for branch_name, branch_pipeline in children:
                    branch_prefix = f"{root_alias}.{param.key}.{branch_name}."
                    branch_overrides: dict[str, dict[str, Any]] = {}
                    for full_key, cfg in sub_overrides.items():
                        if full_key.startswith(branch_prefix):
                            child_alias = full_key[len(branch_prefix):]
                            branch_overrides[child_alias] = cfg
                    if branch_overrides:
                        branch_pipeline = resolve_overrides(
                            branch_pipeline, branch_overrides
                        )
                    updated_children.append((branch_name, branch_pipeline))
                # Rebuild the composite node with updated children
                rebuilt_config = param.build(updated_children)
                new_instance = type(raw_step)(**rebuilt_config)
                new_step = new_instance
                if original_alias is not None:
                    new_step = AliasedStep(alias=original_alias, step=new_instance)
                result[idx] = new_step

    return result
