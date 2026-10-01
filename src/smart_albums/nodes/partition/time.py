"""partition.time — partition assets by temporal proximity.

Sorts assets by captured_at (ascending, tie-break by id) and creates a new
partition whenever the gap between consecutive assets exceeds the configured
time window.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.node import Stage, ConfigParam
from smart_albums.core.registry import stage
from smart_albums.utils.datetime_utils import to_aware_utc
from smart_albums.utils.split import split_contexts


@stage("partition.time")
class PartitionTime(Stage):
    """Partition assets by temporal proximity.

    Assets are sorted by captured_at ascending (tie-break by id). A new
    partition is created whenever the time gap between consecutive sorted
    assets exceeds the configured time_window_minutes.
    """

    _config_schema = (
        ConfigParam(
            key="time_window_minutes",
            type=float,
            default=5.0,
            min=0.1,
            description="Maximum gap in minutes between consecutive assets within a partition.",
        ),
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        if not ctx.assets:
            return [ctx]

        window = timedelta(
            minutes=self.get("time_window_minutes")
        )

        # Separate assets without a timestamp — they cannot be time-partitioned
        timed_assets = [a for a in ctx.assets if a.captured_at is not None]
        untimed_assets = [a for a in ctx.assets if a.captured_at is None]

        if not timed_assets:
            # All assets lack timestamps — return as a single partition
            return [ctx]

        # Normalize to aware-UTC before any comparison/subtraction so a mix
        # of naive and aware captured_at values (different source clients
        # disagree — see CL-12) doesn't raise TypeError (ND-14).
        sorted_assets = sorted(timed_assets, key=lambda a: (to_aware_utc(a.captured_at), a.id))

        partitions: list[list[Any]] = [[sorted_assets[0]]]
        for asset in sorted_assets[1:]:
            gap = to_aware_utc(asset.captured_at) - to_aware_utc(partitions[-1][-1].captured_at)
            if gap > window:
                partitions.append([asset])
            else:
                partitions[-1].append(asset)

        # Add untimed assets as their own partition if any exist
        if untimed_assets:
            partitions.append(untimed_assets)

        results = split_contexts(ctx, partitions, "time")
        if results:
            results[0].stats["partition.time.total_assets"] = len(ctx.assets)
            results[0].stats["partition.time.partitions_created"] = len(partitions)
            results[0].stats["partition.time.untimed_assets"] = len(untimed_assets)
        return results
