"""Filter stage that removes screenshots from the pipeline."""

from __future__ import annotations

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.node import Stage
from smart_albums.core.registry import stage


@stage("filter.screenshots")
class FilterScreenshots(Stage):
    """Remove assets flagged as screenshots."""

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        before = len(ctx.assets)
        ctx.assets = [a for a in ctx.assets if not a.metadata.get("is_screenshot")]
        excluded = before - len(ctx.assets)
        ctx.stats["filter.screenshots.excluded"] = excluded
        if ctx.progress and excluded:
            ctx.progress.log(f"Excluded {excluded} screenshot(s).")
        return [ctx]
