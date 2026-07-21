"""filter.random — randomly subsample ctx.assets to a configured count."""

from __future__ import annotations

import random as _random

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.node import Stage, ConfigParam
from smart_albums.core.registry import stage
from smart_albums.nodes.filter._empty_pool import handle_empty_pool, handle_empty_pool_config


@stage("filter.random")
class FilterRandom(Stage):
    """Randomly subsample ctx.assets to photo_count items."""

    _config_schema = tuple([
        ConfigParam("photo_count", int, default=500, min=1),]+
        handle_empty_pool_config
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        photo_count = self.get("photo_count")
        seed = self.get("rng_seed")
        rng = _random.Random(seed) if seed is not None else _random.Random()

        pre_filter = list(ctx.assets)
        k = min(photo_count, len(pre_filter))
        candidates = rng.sample(pre_filter, k) if k > 0 else []

        ctx.assets = handle_empty_pool(self, ctx, candidates, pre_filter, photo_count)
        return [ctx]
