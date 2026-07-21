"""filter.none — no-op passthrough; truncates ctx.assets to photo_count."""

from __future__ import annotations

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.node import Stage, ConfigParam
from smart_albums.core.registry import stage
from smart_albums.nodes.filter._empty_pool import handle_empty_pool, handle_empty_pool_config


@stage("filter.none")
class FilterNone(Stage):
    """No-op passthrough; truncates ctx.assets to photo_count."""

    _config_schema = tuple(
        [ConfigParam("photo_count", int, default=500, min=1)] +
         handle_empty_pool_config)

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        photo_count = self.get("photo_count")
        pre_filter = list(ctx.assets)
        candidates = pre_filter[:photo_count]
        ctx.assets = handle_empty_pool(self, ctx, candidates, pre_filter, photo_count)
        return [ctx]
