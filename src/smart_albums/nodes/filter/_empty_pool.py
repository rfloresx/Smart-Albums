"""Shared empty-pool fallback for content-filter stages."""

from __future__ import annotations

import random
from typing import Sequence
from enum import Enum

from smart_albums.core.context import PipelineContext
from smart_albums.core.models import Asset
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

    if candidates:
        return candidates[:photo_count]

    behavior = node.get("on_empty_pool")
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
