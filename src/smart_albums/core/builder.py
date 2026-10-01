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
            # Auto-alias from stage name. `auto_counts` only de-duplicates
            # against *other* auto-generated aliases of the same stage
            # name — it was never checked against aliases already present
            # in `alias_map` from an explicit AliasedStep. So an explicit
            # alias("scenes", ...) at an earlier index, followed by an
            # unaliased stage named "scenes" later in the same pipeline,
            # silently produced the same auto-alias "scenes" and
            # overwrote the explicit entry in `alias_map` — any config
            # override or composite dot-path override targeting "scenes"
            # then silently applied to the wrong step (CO-14).
            stage_name = _extract_stage_name(step)
            auto_counts[stage_name] += 1
            count = auto_counts[stage_name]
            auto_alias = stage_name if count == 1 else f"{stage_name}_{count}"
            while auto_alias in alias_map:
                count += 1
                auto_alias = f"{stage_name}_{count}"
            auto_counts[stage_name] = count
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

        if isinstance(raw_step, PipelineNode):
            # A direct (non-dot-path) override on a step that is already a
            # constructed PipelineNode *instance* — e.g. best_of_year's
            # `alias("output", ForkBySelection(Branch(...), Branch(...)))`
            # — used to be rebuilt as a plain `(name, merged_config)` tuple.
            # _extract_config() returns {} for an instance (it only knows
            # how to read config out of tuples/StepSpecs), so `merged` was
            # just the override dict, and build_node() would then call
            # `cls(config)` with that dict as the constructor's first
            # positional argument. For Fork/ForkBySelection that argument
            # is `*args: Branch`, so the override dict got treated as a
            # single Branch and wrapped as `Branch(name="branch_0",
            # pipeline=<the override dict>)` — silently discarding every
            # real branch and pipeline the instance was built with, and
            # producing a broken node that fails deep inside run_pipeline
            # with a confusing "Unknown stage" error.
            from smart_albums.core.node import CompositeNode

            if isinstance(raw_step, CompositeNode):
                # CompositeNode subclasses resolve their internal
                # sub-pipeline once, in __init__, from their config. There
                # is no supported way to apply a config change to an
                # already-built instance after the fact (updating
                # .config wouldn't touch the already-resolved
                # _resolved_pipeline), so fail loudly instead of silently
                # doing nothing or corrupting the instance.
                raise ValueError(
                    f"Cannot apply a direct config override to alias "
                    f"'{alias_key}': it is a pre-built {type(raw_step).__name__} "
                    f"instance (a CompositeNode). Pass its config as a "
                    f"(name, config) tuple or class reference instead of a "
                    f"constructed instance if you need to override it, or "
                    f"target a specific child via a dot-path override."
                )

            # For a plain PipelineNode instance (e.g. Fork/ForkBySelection,
            # which read from self.branches rather than self.config for
            # anything that matters), merge the override into a shallow
            # copy of the instance instead of discarding it. copy.copy
            # keeps the original object (and any pipeline that reuses the
            # same module-level instance, as best_of_year.py does) intact.
            import copy as _copy

            new_instance = _copy.copy(raw_step)
            new_instance.config = {**raw_step.config, **config}
            new_step: StepInput = new_instance
            if original_alias is not None:
                new_step = AliasedStep(alias=original_alias, step=new_instance)
            result[idx] = new_step
            continue

        name = _extract_stage_name(raw_step)
        existing_config = _extract_config(raw_step)
        merged = {**existing_config, **config}

        # Rebuild as (name, config) tuple wrapped in AliasedStep
        new_step = (name, merged)
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

        applied_keys: set[str] = set()

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
                # Route overrides to the appropriate child pipeline
                updated_children: list[tuple[str, Pipeline]] = []
                for branch_name, branch_pipeline in children:
                    branch_prefix = f"{root_alias}.{param.key}.{branch_name}."
                    branch_overrides: dict[str, dict[str, Any]] = {}
                    for full_key, cfg in sub_overrides.items():
                        if full_key.startswith(branch_prefix):
                            child_alias = full_key[len(branch_prefix):]
                            branch_overrides[child_alias] = cfg
                            applied_keys.add(full_key)
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

        elif isinstance(raw_step, (tuple, StepSpec)):
            # A fork (or any other CompositeParam-bearing stage) expressed
            # as a raw (name, config) tuple or StepSpec, rather than a
            # constructed instance — this is exactly how JSON pipeline
            # definitions parsed by ``cli/app.py``'s `run` command
            # represent a fork (see ``_parse_step_config``), so any
            # pipeline built that way silently ignored every
            # "<alias>.branches.<branch_name>.<child_alias>"-style
            # pipeline_settings override with no error at all — the value
            # simply never reached the branch's pipeline. This mirrors the
            # PipelineNode-instance branch above, but reads/writes the
            # config dict's "branches" key directly instead of going
            # through CompositeParam.resolve()/.build(), since there's no
            # instance yet to resolve from.
            stage_name = _extract_stage_name(raw_step)
            step_config = _extract_config(raw_step)

            from smart_albums.core.node import CompositeParam

            composite_params = [
                p for p in registry.get_stage_config(stage_name)
                if isinstance(p, CompositeParam)
            ]
            if composite_params and composite_params[0].key in step_config:
                param = composite_params[0]
                branches_raw = step_config[param.key]

                if isinstance(branches_raw, dict):
                    branch_items: list[tuple[str, Any]] = list(branches_raw.items())
                    is_dict_format = True
                else:
                    branch_items = [(item[0], item[1]) for item in branches_raw]
                    is_dict_format = False

                updated_items: list[tuple[str, Any]] = []
                for branch_name, branch_pipeline in branch_items:
                    branch_prefix = f"{root_alias}.{param.key}.{branch_name}."
                    branch_overrides = {}
                    for full_key, cfg in sub_overrides.items():
                        if full_key.startswith(branch_prefix):
                            child_alias = full_key[len(branch_prefix):]
                            branch_overrides[child_alias] = cfg
                            applied_keys.add(full_key)
                    if branch_overrides:
                        branch_pipeline = resolve_overrides(branch_pipeline, branch_overrides)
                    updated_items.append((branch_name, branch_pipeline))

                new_branches: Any = (
                    dict(updated_items) if is_dict_format
                    else [[name, p] for name, p in updated_items]
                )
                new_config = {**step_config, param.key: new_branches}
                new_step = (stage_name, new_config)
                if original_alias is not None:
                    new_step = AliasedStep(alias=original_alias, step=new_step)
                result[idx] = new_step

        # Every sub-override key must be routed to something. A key that
        # matches neither branch prefix — a typo'd branch name, a prefix
        # that doesn't match this stage's CompositeParam key, or an alias
        # that turned out not to have a CompositeParam at all — used to be
        # dropped with no error, silently doing nothing. Fail loudly
        # instead, matching the "Unknown alias" error already raised above
        # for a root alias that doesn't exist at all.
        unmatched = sorted(set(sub_overrides.keys()) - applied_keys)
        if unmatched:
            raise ValueError(
                f"Composite override(s) for alias '{root_alias}' did not match "
                f"any child pipeline: {unmatched}. Check the branch name(s) "
                f"and that '{root_alias}' has a composite/branches parameter."
            )

    return result
