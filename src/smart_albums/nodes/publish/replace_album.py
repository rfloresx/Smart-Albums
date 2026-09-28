"""Publish stage that replaces a fixed album's contents via add/remove diff."""

from __future__ import annotations

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.node import Stage, ConfigParam
from protocols_system import ProtocolsRegistry
from protocols_system.protocols import IImageClient
from smart_albums.core.registry import stage


@stage("publish.replace_album")
class PublishReplaceAlbum(Stage):
    """Replace a fixed album's contents via add/remove diff."""

    _config_schema = (
        ConfigParam(
            "name",
            str,
            required=True,
            description="Name of the album to create or synchronise.",
        ),
        ConfigParam(
            "dry_run",
            bool,
            default=False,
            description="If true, compute diff stats without writing.",
        ),
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        name = self.get("name")
        dry_run = self.get("dry_run")
        desired = {a.id for a in ctx.assets}

        image_client = ProtocolsRegistry.get_instance(IImageClient)
        if image_client is None:
            raise RuntimeError("image_client is required for stage 'publish.replace_album'")
        album = await image_client.get_album_by_name(name)
        if album is None:
            current: set[str] = set()
            created = True
        else:
            current = {a.id for a in await image_client.list_album_assets(album.id)}
            created = False

        to_add = desired - current
        to_remove = current - desired

        if dry_run:
            ctx.stats["publish.replace_album.dry_run"] = True
            ctx.stats["publish.replace_album.would_add"] = len(to_add)
            ctx.stats["publish.replace_album.would_remove"] = len(to_remove)
            return [ctx]

        if created:
            result = await image_client.create_album(name, list(desired))
            album_id = result.id
        else:
            # album is guaranteed non-None when created is False
            album_id = album.id  # type: ignore[union-attr]

        if to_add:
            await image_client.add_assets_to_album(album_id, list(to_add))
        if to_remove:
            await image_client.remove_assets_from_album(album_id, list(to_remove))

        ctx.stats["publish.replace_album.created"] = created
        ctx.stats["publish.replace_album.added"] = len(to_add)
        ctx.stats["publish.replace_album.removed"] = len(to_remove)
        ctx.stats["publish.replace_album.final_count"] = len(desired)
        return [ctx]
