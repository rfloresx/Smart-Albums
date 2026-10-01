"""filter.people — retain only assets containing at least one recognized face."""

from __future__ import annotations

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.node import Stage, ConfigParam
from protocols_system import ProtocolsRegistry
from protocols_system.protocols import IImageClient
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
        # NOTE (ND-16/ND-09): ``search_people_any`` fetches every
        # face-bearing asset in the whole library on each run and we then
        # intersect it with the current context below. This is correct but
        # not scoped — on a large library it transfers far more than the
        # context needs. Narrowing it requires a scoped people-search
        # capability on IImageClient (a protocol change), so it's left as a
        # performance follow-up rather than silently changing behavior here.
        fetched = await image_client.search_people_any()

        # Keep the context's own Asset objects, not the fresh objects the
        # search API returns — see ND-03 in the code review. Using
        # `fetched` directly would silently drop every bit of upstream
        # metadata (score, phash, embedding, ...) for matched assets.
        by_id = {a.id: a for a in pre_filter}
        candidates = [by_id[a.id] for a in fetched if a.id in by_id]

        ctx.assets = handle_empty_pool(self, ctx, candidates, pre_filter, photo_count)
        ctx.stats["filter.people.kept"] = len(ctx.assets)
        return [ctx]
