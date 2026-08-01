"""filter.people — retain only assets containing at least one recognized face."""

from __future__ import annotations

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.node import Stage, ConfigParam
from smart_albums.core.protocol_registry import ProtocolsRegistry
from smart_albums.core.protocols import IImageClient
from smart_albums.core.registry import stage
from smart_albums.nodes.filter._empty_pool import handle_empty_pool, handle_empty_pool_config


@stage("filter.people")
class FilterPeople(Stage):
    """Retain only assets containing at least one recognized face."""

    _config_schema = tuple(
        [ConfigParam("photo_count", int, default=500, min=1),] +
        handle_empty_pool_config
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        photo_count = self.get("photo_count")

        pre_filter = list(ctx.assets)
        image_client = ProtocolsRegistry.get_instance(IImageClient)
        if image_client is None:
            raise RuntimeError("image_client is required for stage 'filter.people'")
        fetched = await image_client.search_people_any()

        in_context = {a.id for a in pre_filter}
        candidates = [a for a in fetched if a.id in in_context]

        ctx.assets = handle_empty_pool(self, ctx, candidates, pre_filter, photo_count)
        ctx.stats["filter.people.kept"] = len(ctx.assets)
        return [ctx]
