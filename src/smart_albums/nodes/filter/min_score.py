"""filter.min_score — exclude assets scoring below a configurable threshold."""

from __future__ import annotations

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.node import Stage, ConfigParam
from smart_albums.core.protocols import IProgressReporter, ProtocolsRegistry
from smart_albums.core.registry import stage


@stage("filter.min_score")
class FilterMinScore(Stage):
    """Exclude assets scoring below threshold."""

    _config_schema = (
        ConfigParam(
            "threshold",
            float,
            default=0.0,
            min=0.0,
            max=1.0,
            description="Minimum score to retain an asset.",
        ),
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        threshold = self.get("threshold")
        if threshold <= 0.0:
            ctx.stats["filter.min_score.excluded"] = 0
            return [ctx]
        before = len(ctx.assets)
        ctx.assets = [
            a for a in ctx.assets if a.metadata.get("score", 0.0) >= threshold
        ]
        excluded = before - len(ctx.assets)
        ctx.stats["filter.min_score.excluded"] = excluded
        if excluded:
            progress = ProtocolsRegistry.get_instance(IProgressReporter)
            if progress:
                progress.log(
                    f"Excluded {excluded} asset(s) below score {threshold:.2f}."
                )
        return [ctx]
