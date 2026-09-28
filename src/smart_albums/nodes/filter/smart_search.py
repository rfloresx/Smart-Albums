"""filter.smart_search — retain assets matching a smart-search natural-language query."""

from __future__ import annotations

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.node import Stage, ConfigParam
from protocols_system import ProtocolsRegistry
from protocols_system.protocols import IImageClient
from smart_albums.core.registry import stage
from smart_albums.nodes.filter._empty_pool import handle_empty_pool, handle_empty_pool_config


@stage("filter.smart_search")
class FilterSmartSearch(Stage):
    """Retain only assets matching a smart-search natural-language query."""

    _config_schema = tuple([
        ConfigParam("query", str, required=True),
        ConfigParam("max_candidates", int, default=500, min=1),
        ConfigParam("photo_count", int, default=500, min=1),]+
        handle_empty_pool_config
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        query = self.get("query")
        max_candidates = self.get("max_candidates")
        photo_count = self.get("photo_count")

        pre_filter = list(ctx.assets)
        image_client = ProtocolsRegistry.get_instance(IImageClient)
        if image_client is None:
            raise RuntimeError("image_client is required for stage 'filter.smart_search'")
        fetched = await image_client.search_smart(query, max_candidates)

        in_context = {a.id for a in pre_filter}
        candidates = [a for a in fetched if a.id in in_context]

        ctx.assets = handle_empty_pool(self, ctx, candidates, pre_filter, photo_count)
        ctx.stats["filter.smart_search.kept"] = len(ctx.assets)
        return [ctx]
