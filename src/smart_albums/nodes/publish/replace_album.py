"""Publish stage that replaces a fixed album's contents via add/remove diff."""

from __future__ import annotations

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.node import Stage, ConfigParam
from protocols_system import ProtocolsRegistry
from protocols_system.protocols import IImageClient, IProgressReporter
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
        ConfigParam(
            "allow_empty",
            bool,
            default=False,
            description=(
                "If false (default), refuse to replace the album when the "
                "desired asset set is empty. Without this guard, an empty "
                "pool caused by an upstream failure or an empty-pool filter "
                "would silently wipe the live album."
            ),
        ),
    )

    async def func(self, contexts: ContextBatch) -> ContextBatch:
        """Run once against the union of all contexts' assets.

        The base ``Stage.func`` calls ``run()`` once per context in the
        batch. For a publish stage that would mean each context — e.g. one
        per unmerged partition, or one per fork branch — independently
        "replaces" the same album in turn, with the last one to run
        winning and every earlier context's assets being removed again.
        Publishing is meant to happen once per pipeline run against the
        full selection, so contexts are combined before the album is
        touched.
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
        result = await self.run(merged)
        # Report the single merged result back once per input context so
        # callers that iterate the returned batch see consistent length
        # expectations; the stats/dry-run outcome is identical for both.
        return result

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        name = self.get("name")
        dry_run = self.get("dry_run")
        allow_empty = self.get("allow_empty")
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

        # Refuse to empty out an existing album. An empty `desired` set
        # here usually means something went wrong upstream (an
        # empty-pool filter outcome, a failed retrieve/select stage, or
        # ND-01-style data loss) rather than a deliberate "clear the
        # album" request. Without this guard, `to_remove = current` and
        # every asset in the live album gets removed.
        if not desired and current and not allow_empty:
            ctx.stats["publish.replace_album.dry_run"] = False
            ctx.stats["publish.replace_album.refused_empty"] = True
            ctx.stats["publish.replace_album.created"] = False
            ctx.stats["publish.replace_album.added"] = 0
            ctx.stats["publish.replace_album.removed"] = 0
            ctx.stats["publish.replace_album.final_count"] = len(current)

            progress = ProtocolsRegistry.get_instance(IProgressReporter)
            if progress:
                progress.warn(
                    f"publish.replace_album: refusing to empty album {name!r} "
                    f"(desired selection is empty, album currently has "
                    f"{len(current)} asset(s)). Set allow_empty=true to override."
                )
            return [ctx]

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
            # The album was just created with the full desired set, so
            # there is nothing left to add — the previous code re-issued
            # add_assets_to_album with the same ids create_album() was
            # just given, which some servers reject as a duplicate add.
            to_add = set()
        else:
            # album is guaranteed non-None when created is False
            album_id = album.id  # type: ignore[union-attr]

        if to_add:
            await image_client.add_assets_to_album(album_id, list(to_add))
        if to_remove:
            await image_client.remove_assets_from_album(album_id, list(to_remove))

        # Record publish history for everything now in the album (not just
        # the newly-added ids) so an asset that stays in the album keeps a
        # fresh last-shown date, giving select.avoid_repeat's cooldown
        # something to read on later runs (ND-11). Skipped on dry runs
        # (handled above); a missing cache manager makes this a no-op.
        from datetime import date
        from smart_albums.utils.history import record_shown
        record_shown(desired, date.today())

        ctx.stats["publish.replace_album.created"] = created
        ctx.stats["publish.replace_album.added"] = len(to_add)
        ctx.stats["publish.replace_album.removed"] = len(to_remove)
        ctx.stats["publish.replace_album.final_count"] = len(desired)
        return [ctx]
