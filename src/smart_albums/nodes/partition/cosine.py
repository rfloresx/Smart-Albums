"""partition.cosine — partition assets by cosine similarity (scene clustering).

Groups assets whose embedding cosine similarity exceeds the configured
threshold using a FAISS IndexFlatIP index on L2-normalized vectors and
Union-Find for transitive grouping.  Optionally pre-partitions by temporal
proximity before clustering within each temporal group.

Also registered as ``partition.scene`` (alias).
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.node import Stage, ConfigParam
from smart_albums.core.registry import stage, STAGE_REGISTRY
from smart_albums.utils.faiss_grouping import group_by_embedding_similarity
from smart_albums.utils.split import split_contexts


def _temporal_partitions(assets: list[Any], window: timedelta) -> list[list[Any]]:
    """Split assets into temporal groups using a time-window gap criterion.

    Sorts assets by captured_at (tie-break by id) and starts a new group
    whenever the gap between consecutive assets exceeds *window*.
    Assets without a timestamp are grouped together separately.
    """
    timed = [a for a in assets if a.captured_at is not None]
    untimed = [a for a in assets if a.captured_at is None]

    if not timed:
        return [assets] if assets else []

    sorted_assets = sorted(timed, key=lambda a: (a.captured_at, a.id))
    partitions: list[list[Any]] = [[sorted_assets[0]]]
    for asset in sorted_assets[1:]:
        if asset.captured_at - partitions[-1][-1].captured_at > window:
            partitions.append([asset])
        else:
            partitions[-1].append(asset)

    if untimed:
        partitions.append(untimed)

    return partitions


@stage("partition.cosine")
class PartitionCosine(Stage):
    """Partition assets by cosine similarity (scene clustering).

    If the context has no existing partitions (depth 0), assets are first
    split into temporal sub-groups using the configured time window. Within
    each temporal sub-group, a FAISS IndexFlatIP index with L2-normalized
    embeddings and Union-Find at the cosine threshold clusters assets into
    scene groups.
    """

    _config_schema = (
        ConfigParam(
            key="threshold",
            type=float,
            default=0.85,
            description="Minimum cosine similarity to group assets as scene members.",
        ),
        ConfigParam(
            key="time_window_minutes",
            type=float,
            default=30.0,
            description="Time window in minutes for optional temporal pre-partitioning.",
        ),
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        if not ctx.assets:
            return [ctx]

        threshold = self.get("threshold")
        time_window_minutes = self.get("time_window_minutes")

        with_emb = [a for a in ctx.assets if a.metadata.get("embedding")]
        without_emb = [a for a in ctx.assets if not a.metadata.get("embedding")]

        if len(with_emb) < 2:
            return [ctx]

        # If context is at depth 0 (not already partitioned), pre-partition by time
        if ctx.partition_depth == 0:
            window = timedelta(minutes=time_window_minutes)
            temporal_groups = _temporal_partitions(with_emb, window)
        else:
            # Already within a partition — treat all assets as one group
            temporal_groups = [with_emb]

        # Cluster within each temporal group using FAISS + Union-Find
        all_partitions: list[list[Any]] = []
        total_groups_formed = 0

        for temporal_group in temporal_groups:
            if len(temporal_group) < 2:
                all_partitions.append(temporal_group)
                total_groups_formed += 1
                continue

            groups = group_by_embedding_similarity(temporal_group, threshold)
            for group_assets in groups.values():
                all_partitions.append(group_assets)
            total_groups_formed += len(groups)

        # Add assets without embeddings as individual partitions
        for asset in without_emb:
            all_partitions.append([asset])

        results = split_contexts(ctx, all_partitions, "cosine")
        if results:
            results[0].stats["partition.cosine.total_assets"] = len(ctx.assets)
            results[0].stats["partition.cosine.groups_formed"] = total_groups_formed
            results[0].stats["partition.cosine.partitions"] = len(all_partitions)
        return results


# Register alias: partition.scene → same class
STAGE_REGISTRY["partition.scene"] = PartitionCosine
