"""Filter stage that removes non-image assets (e.g. videos) from the pipeline."""

from __future__ import annotations

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.node import Stage
from smart_albums.core.registry import stage


@stage("filter.non_images")
class RetainImages(Stage):
    """Remove non-image MIME types."""

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        before = len(ctx.assets)
        ctx.assets = [a for a in ctx.assets if a.mime_type.lower().startswith("image/")]
        excluded = before - len(ctx.assets)
        ctx.stats["filter.non_images.excluded"] = excluded
        if ctx.progress and excluded:
            ctx.progress.log(f"Excluded {excluded} non-image asset(s).")
        return [ctx]
