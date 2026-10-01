"""partition.cosine — partition assets by cosine similarity (scene clustering).

Groups assets whose embedding cosine similarity exceeds the configured
threshold using a FAISS IndexFlatIP index on L2-normalized vectors and
Union-Find for transitive grouping.

Also registered as ``partition.scene`` (alias).
"""

from __future__ import annotations

from typing import Any

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.node import Stage, ConfigParam
from smart_albums.core.registry import stage, STAGE_REGISTRY
from smart_albums.utils.faiss_grouping import group_by_embedding_similarity
from smart_albums.utils.split import split_contexts


@stage("partition.cosine")
class PartitionCosine(Stage):
    """Partition assets by cosine similarity (scene clustering).

    Builds a FAISS IndexFlatIP index from L2-normalized embedding vectors
    and uses Union-Find to form transitive groups of assets whose cosine
    similarity exceeds the configured threshold.
    """

    _config_schema = (
        ConfigParam(
            key="threshold",
            type=float,
            default=0.85,
            min=-1.0,
            max=1.0,
            description="Minimum cosine similarity to group assets as scene members. "
            "Cosine similarity is defined on [-1.0, 1.0] (ND-15).",
        ),
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        if not ctx.assets:
            return [ctx]

        threshold = self.get("threshold")

        with_emb = [a for a in ctx.assets if a.metadata.get("embedding")]
        without_emb = [a for a in ctx.assets if not a.metadata.get("embedding")]

        # Only cluster when there are at least 2 embedded assets to compare.
        # With fewer than 2, fall back to singleton partitions rather than
        # returning the whole (unsplit) context — see partition/faiss.py
        # for the failure mode this avoids.
        if len(with_emb) >= 2:
            groups = group_by_embedding_similarity(with_emb, threshold)
            all_partitions: list[list[Any]] = list(groups.values())
        else:
            groups = {}
            all_partitions = [[a] for a in with_emb]

        # Add assets without embeddings as individual partitions
        for asset in without_emb:
            all_partitions.append([asset])

        results = split_contexts(ctx, all_partitions, "cosine")
        if results:
            results[0].stats["partition.cosine.total_assets"] = len(ctx.assets)
            results[0].stats["partition.cosine.groups_formed"] = len(groups)
            results[0].stats["partition.cosine.partitions"] = len(all_partitions)
        return results


# Register alias: partition.scene → same class
STAGE_REGISTRY["partition.scene"] = PartitionCosine
