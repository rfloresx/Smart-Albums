"""Pipeline runner — builds and executes a pipeline from a spec."""

from __future__ import annotations

import logging
import time

from typing import Any

from smart_albums.core.context import ContextBatch
from smart_albums.core.node import PipelineNode
from smart_albums.core.spec import Pipeline, StepSpec


import smart_albums.core.builder as builder
import smart_albums.core.registry as registry

logger = logging.getLogger(__name__)


def _coerce_and_validate(param: Any, value: Any, stage_name: str) -> Any:
    """Coerce ``value`` to ``param.type`` and enforce min/max/choices (CO-09).

    ``build_node`` previously passed config values through untouched: a
    ``ConfigParam``'s declared ``type``, ``min``, ``max``, and ``choices``
    were pure documentation that nothing enforced, so an out-of-range
    threshold, a value of the wrong type (e.g. the string ``"5"`` for an
    ``int`` param coming from a JSON/web config), or a bogus ``choices``
    value flowed straight into the stage and failed — if at all — much
    later and far less clearly. This coerces common scalar types and
    rejects out-of-range / invalid-choice values up front with a message
    naming the stage and key.
    """
    if value is None:
        return value

    expected = param.type
    # Coerce common scalar types. bool is handled explicitly because
    # ``bool("false")`` is True — a classic footgun for string configs.
    try:
        if expected is bool and not isinstance(value, bool):
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in ("true", "1", "yes", "on"):
                    value = True
                elif lowered in ("false", "0", "no", "off"):
                    value = False
                else:
                    raise ValueError(f"cannot interpret {value!r} as a boolean")
            else:
                value = bool(value)
        elif expected is int and not isinstance(value, bool) and not isinstance(value, int):
            value = int(value)
        elif expected is float and not isinstance(value, float):
            # Accept int→float silently; reject non-numeric strings.
            if isinstance(value, (int, str)):
                value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Config '{param.key}' for stage '{stage_name}' must be "
            f"{getattr(expected, '__name__', expected)!r}: {exc}"
        ) from exc

    if param.choices is not None and value not in param.choices:
        raise ValueError(
            f"Config '{param.key}' for stage '{stage_name}' must be one of "
            f"{param.choices}; got {value!r}"
        )

    if param.min is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
        if value < param.min:
            raise ValueError(
                f"Config '{param.key}' for stage '{stage_name}' must be "
                f">= {param.min}; got {value!r}"
            )
    if param.max is not None and isinstance(value, (int, float)) and not isinstance(value, bool):
        if value > param.max:
            raise ValueError(
                f"Config '{param.key}' for stage '{stage_name}' must be "
                f"<= {param.max}; got {value!r}"
            )

    return value


def build_node(spec: StepSpec) -> PipelineNode:
    """Instantiate a PipelineNode from a StepSpec, validating config."""

    from smart_albums.core.node import ConfigParam, CompositeParam

    cls = registry.get_stage(spec.name)
    if cls is None:
        raise ValueError(f"Unknown stage '{spec.name}'")

    schema = {
        p.key: p
        for p in cls._config_schema
        if isinstance(p, ConfigParam)
    }

    composite_schema = {
        p.key: p
        for p in cls._config_schema
        if isinstance(p, CompositeParam)
    }

    config = {}
    composite_kwargs: dict[str, Any] = {}

    for key, param in schema.items():

        if key in spec.config:
            # Coerce + validate caller-supplied values (CO-09). Defaults
            # declared on the ConfigParam are trusted as-is.
            config[key] = _coerce_and_validate(param, spec.config[key], spec.name)

        elif param.default is not None:
            config[key] = param.default

        elif not param.required:
            # Optional param with no default — skip
            pass

        else:
            raise ValueError(
                f"Missing config '{key}' "
                f"for stage '{spec.name}'"
            )

    # Handle CompositeParam entries
    # Config must contain the composite key directly (e.g. "branches": [...])
    for key, cparam in composite_schema.items():
        if key in spec.config and cparam.build is not None:
            built = cparam.build(spec.config[key])
            composite_kwargs.update(built)

    # Check for unrecognized keys
    for key in spec.config:
        if key not in schema and key not in composite_schema:
            raise ValueError(
                f"Unknown config '{key}' "
                f"for stage '{spec.name}'"
            )

    if composite_kwargs:
        # Composite nodes use keyword args directly (e.g. Fork(branches=[...]))
        node = cls(**composite_kwargs)
        node.config.update(config)
        return node

    return cls(config)


async def run_pipeline(
    pipeline: Pipeline,
    contexts: ContextBatch,
) -> ContextBatch:
    """Build and execute a pipeline against the given contexts."""
    global _pipeline_depth
    specs = builder.build_pipeline(pipeline)

    is_outermost = _pipeline_depth == 0
    _pipeline_depth += 1
    try:
        # Only the outermost pipeline reports pipeline-level structure;
        # nested composite/fork sub-pipelines must not reset the reporter's
        # global stage counters (CO-11).
        if is_outermost:
            _notify_pipeline_start(contexts, len(specs))
        return await _run_pipeline_body(specs, contexts, report_stages=is_outermost)
    finally:
        _pipeline_depth -= 1


async def _run_pipeline_body(
    specs: list[StepSpec],
    contexts: ContextBatch,
    *,
    report_stages: bool,
) -> ContextBatch:
    """Execute already-built pipeline specs. See ``run_pipeline``."""
    for stage_idx, spec in enumerate(specs):
        spec_name = spec.name or "<unnamed>"
        if spec.alias:
            spec_name = f"{spec.alias} ({spec_name})"
        node = spec.instance if spec.instance is not None else build_node(spec)
        assets_count = sum([len(ctx.assets) for ctx in contexts])
        logger.info("Stage %-35s  starting  (%d ctx) (%d assets)", spec_name, len(contexts), assets_count)

        # Notify progress reporter of stage-level entry (outermost only)
        if report_stages:
            _notify_stage_enter(contexts, stage_idx, spec_name)

        # Debug: log effective config for this node
        if logger.isEnabledFor(logging.DEBUG):
            effective_cfg = {
                k: v for k, v in node.config.items()
                if v is not None
            }
            if effective_cfg:
                logger.debug(
                    "Stage %-35s  config: %s",
                    spec_name,
                    effective_cfg,
                )
            # Log per-context input summary
            for i, ctx in enumerate(contexts):
                logger.debug(
                    "Stage %-35s  input ctx[%d]: %d assets, partition=%s depth=%d",
                    spec_name,
                    i,
                    len(ctx.assets),
                    ctx.partition_name or "(root)",
                    ctx.partition_depth,
                )

        t0 = time.perf_counter()

        contexts = await node.func(contexts)

        elapsed = time.perf_counter() - t0
        assets_count = sum([len(ctx.assets) for ctx in contexts])
        logger.info(
            "Stage %-35s  done      (%d ctx, %d assets, %.2fs)\n%s",
            spec_name,
            len(contexts),
            assets_count,
            elapsed,
            _summarise(contexts),
        )

        # Debug: log per-context output summary
        if logger.isEnabledFor(logging.DEBUG):
            for i, ctx in enumerate(contexts):
                logger.debug(
                    "Stage %-35s  output ctx[%d]: %d assets, stats=%s",
                    spec_name,
                    i,
                    len(ctx.assets),
                    ctx.stats or "{}",
                )

    return contexts


def _summarise(contexts: ContextBatch) -> str:
    """Return a compact multi-line summary of a ContextBatch.

    Each context is rendered as one indented line showing its asset count,
    partition path (when nested), and any stats entries that were set during
    the stage just completed.
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return ""

    lines: list[str] = []
    for i, ctx in enumerate(contexts):
        # Build a human-readable context label
        parts: list[str] = []
        if ctx.partition_name:
            depth_indent = "  " * ctx.partition_depth
            parts.append(f"{depth_indent}[{ctx.partition_name}]")
        else:
            parts.append(f"  ctx[{i}]")

        parts.append(f"{len(ctx.assets)} assets")

        # Append any stats entries (sorted for determinism)
        if ctx.stats:
            stats_str = "  stats: " + ", ".join(
                f"{k}={v}" for k, v in sorted(ctx.stats.items())
            )
            lines.append(" ".join(parts))
            lines.append(stats_str)
        else:
            lines.append(" ".join(parts))

    return "\n".join(f"    {l}" for l in lines) if lines else "    (empty batch)"


# ---------------------------------------------------------------------------
# Progress notification helpers
# ---------------------------------------------------------------------------


# Tracks pipeline nesting depth. Composite and fork nodes execute their
# own sub-pipelines by re-entering `run_pipeline`; those nested calls must
# NOT emit pipeline-level progress (start_pipeline / set_stage_index),
# because doing so reset the reporter's `_total_stages`/`_stage_index` to
# the sub-pipeline's values and the web GUI then showed the sub-pipeline's
# position as the global one (CO-11). Only the outermost call (depth 0)
# reports pipeline-level structure; nesting is gated by ``run_pipeline``.
_pipeline_depth = 0


def _notify_pipeline_start(contexts: ContextBatch, total_stages: int) -> None:
    """Notify the progress reporter that a pipeline is starting."""
    from protocols_system import ProtocolsRegistry
    from protocols_system.protocols import IProgressReporter

    progress = ProtocolsRegistry.get_instance(IProgressReporter)
    if progress and hasattr(progress, "start_pipeline"):
        progress.start_pipeline(total_stages)


def _notify_stage_enter(contexts: ContextBatch, stage_index: int, name: str) -> None:
    """Notify the progress reporter that a stage is being entered."""
    from protocols_system import ProtocolsRegistry
    from protocols_system.protocols import IProgressReporter

    progress = ProtocolsRegistry.get_instance(IProgressReporter)
    if progress and hasattr(progress, "set_stage_index"):
        progress.set_stage_index(stage_index, name)
