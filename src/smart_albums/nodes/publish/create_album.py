"""Publish stage that creates a new album with the current asset selection."""

from __future__ import annotations

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.node import Stage, ConfigParam
from protocols_system import ProtocolsRegistry
from protocols_system.protocols import IImageClient, IProgressReporter
from smart_albums.core.registry import stage


@stage("publish.create_album")
class PublishCreateAlbum(Stage):
    """Create a new album with the assets in ctx.assets."""

    _config_schema = (
        ConfigParam("name", str, required=True),
        ConfigParam("dry_run", bool, default=False),
    )

    async def func(self, contexts: ContextBatch) -> ContextBatch:
        """Run once against the union of all contexts' assets.

        The base ``Stage.func`` calls ``run()`` once per context. For a
        publish stage that meant, with multiple contexts (unmerged
        partitions, fork branches), only the first context to run actually
        created the album — every later context hit the "already exists"
        conflict path and its assets were silently dropped from the album.
        Combining contexts first ensures every selected asset makes it into
        the one album this stage creates.
        """
        if not contexts:
            return contexts
        if len(contexts) == 1:
            return await self.run(contexts[0])

        merged = PipelineContext(
            config=contexts[0].config,
            assets=[a for ctx in contexts for a in ctx.assets],
            stats={},
            metadata=contexts[0].metadata,
        )
        return await self.run(merged)

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        name = self.get("name")
        dry_run = self.get("dry_run")
        asset_ids = [a.id for a in ctx.assets]

        if dry_run:
            ctx.stats["publish.create_album.dry_run"] = True
            ctx.stats["publish.create_album.would_create"] = len(asset_ids)
            return [ctx]

        # Conflict detection via album listing
        image_client = ProtocolsRegistry.get_instance(IImageClient)
        if image_client is None:
            raise RuntimeError("image_client is required for stage 'publish.create_album'")

        albums = await image_client.list_albums()
        for album in albums:
            if album.name == name:
                ctx.stats["publish.create_album.conflict"] = True
                ctx.stats["publish.create_album.created"] = False
                progress = ProtocolsRegistry.get_instance(IProgressReporter)
                if progress:
                    progress.warn(f"Album {name!r} already exists.")
                return [ctx]

        result = await image_client.create_album(name, asset_ids)

        # Record publish history so select.avoid_repeat's cooldown has
        # something to read on later runs (ND-11). Done after the album is
        # successfully created, and skipped entirely on dry runs (handled
        # above). A missing cache manager makes this a no-op.
        from datetime import date
        from smart_albums.utils.history import record_shown
        record_shown(asset_ids, date.today())

        ctx.stats["publish.create_album.created"] = True
        ctx.stats["publish.create_album.name"] = name
        ctx.stats["publish.create_album.url"] = result.url
        ctx.stats["publish.create_album.asset_count"] = len(asset_ids)
        return [ctx]
