"""retrieve.by_year — fetch all photo assets for a configured calendar year."""

from __future__ import annotations

from datetime import datetime

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.node import Stage, ConfigParam
from smart_albums.core.protocol_registry import ProtocolsRegistry
from smart_albums.core.protocols import IImageClient, IProgressReporter
from smart_albums.core.registry import stage


@stage("retrieve.by_year")
class RetrieveByYear(Stage):
    """Fetch all photo assets for the configured calendar year."""

    _config_schema = (
        ConfigParam("year", int, required=True, description="Calendar year."),
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        year = self.get("year")
        after = datetime(year, 1, 1, 0, 0, 0)
        before = datetime(year, 12, 31, 23, 59, 59)

        progress = ProtocolsRegistry.get_instance(IProgressReporter)
        image_client = ProtocolsRegistry.get_instance(IImageClient)

        if progress:
            progress.start_stage(f"Retrieving assets for {year}")

        if image_client is None:
            raise RuntimeError("image_client is required for stage 'retrieve.by_year'")
        fetched = await image_client.search_assets(after, before)

        for _ in fetched:
            if progress:
                progress.advance()

        ctx.assets.extend(fetched)
        ctx.stats["retrieve.by_year.total_retrieved"] = len(fetched)

        if progress:
            progress.finish_stage()
            progress.log(f"Retrieved {len(fetched)} assets for {year}.")

        return [ctx]
