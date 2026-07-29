"""Source stage — retrieve candidates using a configured Source provider.

Constructs a SourceProvider via build_source() from the recipe's source
specification and fetches the scoped candidate pool onto ctx.assets.
"""

from __future__ import annotations

from datetime import date

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.node import Stage, ConfigParam
from smart_albums.core.protocols import IImageClient, ProtocolsRegistry
from smart_albums.core.registry import stage
from smart_albums.nodes.retrieve._sources import build_source


@stage("source")
class Source(Stage):
    """Retrieve candidates using the configured Source provider."""

    _config_schema = (
        ConfigParam(
            "type",
            str,
            required=True,
            choices=["library", "year", "date_range", "recent", "album"],
        ),
        ConfigParam("year", int | None, description="Year for type=year."),
        ConfigParam("from", date | None),
        ConfigParam("to", date | None),
        ConfigParam("days", int | None),
        ConfigParam("weeks", int | None),
        ConfigParam("months", int | None),
        ConfigParam("album_id", str | None),
        ConfigParam("run_date", date, required=True),
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        spec = {
            k: v
            for k, v in self.config.items()
            if k not in ("run_date",) and v is not None
        }
        run_date = self.get("run_date")

        provider = build_source(spec)
        image_client = ProtocolsRegistry.get_instance(IImageClient)
        if image_client is None:
            raise RuntimeError("image_client is required for stage 'source'")
        fetched = await provider.fetch(image_client, run_date)

        ctx.assets.extend(fetched)
        ctx.stats["source.total_retrieved"] = len(fetched)
        return [ctx]
