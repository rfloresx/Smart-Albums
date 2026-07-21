"""partition.evenly — split assets into N roughly equal-sized partitions.

Distributes assets across a fixed number of partitions in round-robin order,
ensuring each partition has approximately the same number of assets
(differing by at most one).
"""

from __future__ import annotations

import logging

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.models import Asset
from smart_albums.core.node import Stage, ConfigParam
from smart_albums.core.registry import stage
from smart_albums.utils.split import split_contexts

logger = logging.getLogger(__name__)


@stage("partition.evenly")
class PartitionEvenly(Stage):
    """Split assets into N partitions with approximately equal counts.

    Assets are distributed sequentially: the first partition gets indices
    0, N, 2N, …; the second gets 1, N+1, 2N+1, …; and so on. This produces
    partitions that differ in size by at most one asset.
    """

    _config_schema = (
        ConfigParam(
            key="number_of_partitions",
            type=int,
            default=4,
            min=1,
            description="Number of partitions to split the assets into.",
        ),
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        if not ctx.assets:
            return [ctx]

        n = self.get("number_of_partitions")

        # Clamp to actual asset count — can't have more partitions than assets
        n = min(n, len(ctx.assets))

        # Distribute assets into N buckets
        buckets: list[list[Asset]] = [[] for _ in range(n)]
        for i, asset in enumerate(ctx.assets):
            buckets[i % n].append(asset)

        results = split_contexts(ctx, buckets, "evenly")

        if results:
            results[0].stats["partition.evenly.total_assets"] = len(ctx.assets)
            results[0].stats["partition.evenly.partitions_created"] = len(results)

        return results
