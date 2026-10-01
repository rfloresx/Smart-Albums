"""Shared empty-pool fallback for content-filter stages."""

from __future__ import annotations

import random
from typing import Sequence
from enum import Enum

from smart_albums.core.context import PipelineContext
from protocols_system.protocols import Asset
from smart_albums.core.node import ConfigParam, PipelineNode


class EmptyPoolError(RuntimeError):
    """Raised when a filter produces an empty pool and on_empty_pool='fail'."""


class EmptyBehavior(str, Enum):
    FAIL = "fail"
    FALLBACK_RANDOM = "fallback_random"
    SKIP = "skip"

def handle_empty_pool(
    node: PipelineNode,
    ctx: PipelineContext,
    candidates: list[Asset],
    pre_filter_assets: Sequence[Asset],
    photo_count: int,
) -> list[Asset]:
    """Apply the configured empty-pool behavior to a filter result.

    Reads from node config:
        on_empty_pool: "fail" | "fallback_random" | "skip"
                              (default: "skip")
        recipe_name: str (used in EmptyPoolError message)
        rng_seed: int (optional, for deterministic fallback)

    Side effects:
        Sets ctx.stats["filter.skipped_empty_pool"] = True when behavior
        produces an empty result (skip, or fallback_random with empty input).
    """

    # Normalize and validate the configured behavior up front. An
    # unrecognized value (a typo, or a value from a stale config) used to
    # silently fall through to the SKIP branch at the bottom — so a
    # misconfigured "fallbackrandom" quietly dropped the whole pool
    # instead of doing what the author intended (ND-15). Fail loudly
    # instead.
    raw_behavior = node.get("on_empty_pool")
    if raw_behavior is None:
        # Not set (or defaults not materialized) — use the documented
        # default rather than erroring.
        behavior = EmptyBehavior.SKIP
    else:
        try:
            behavior = EmptyBehavior(raw_behavior)
        except ValueError as exc:
            valid = ", ".join(b.value for b in EmptyBehavior)
            raise ValueError(
                f"Invalid on_empty_pool value {raw_behavior!r}; expected one of: {valid}"
            ) from exc

    if candidates:
        if len(candidates) <= photo_count:
            return candidates
        # When the filter leaves more candidates than requested, take a
        # random sample rather than ``candidates[:photo_count]`` — the
        # slice kept the first N in whatever order the server/filter
        # happened to return (typically newest-first or an arbitrary
        # scan order), which systematically biased the output toward one
        # end of the pool instead of representing it (ND-16).
        seed = node.get("rng_seed")
        rng = random.Random(seed) if seed is not None else random.Random()
        return rng.sample(candidates, photo_count)

    recipe_name = node.config.get("recipe_name", "<unnamed>")

    if behavior == EmptyBehavior.FAIL:
        raise EmptyPoolError(f"recipe '{recipe_name}' produced an empty pool")

    if behavior == EmptyBehavior.FALLBACK_RANDOM:
        seed = node.get("rng_seed")
        rng = random.Random(seed) if seed is not None else random.Random()
        if not pre_filter_assets:
            ctx.stats["filter.skipped_empty_pool"] = True
            return []
        k = min(photo_count, len(pre_filter_assets))
        return rng.sample(list(pre_filter_assets), k)

    # behavior == EmptyBehavior.SKIP
    ctx.stats["filter.skipped_empty_pool"] = True
    return []

handle_empty_pool_config: list[ConfigParam] = [
    ConfigParam("on_empty_pool", str, default=EmptyBehavior.SKIP,
                choices=[EmptyBehavior.FAIL.value, EmptyBehavior.FALLBACK_RANDOM.value, EmptyBehavior.SKIP.value]),
    ConfigParam("rng_seed", int, default=None),
]
