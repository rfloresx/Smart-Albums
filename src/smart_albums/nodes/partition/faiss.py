"""partition.faiss — partition assets by FAISS near-duplicate detection.

Groups assets whose embedding cosine similarity exceeds the configured
threshold using a FAISS IndexFlatIP index on L2-normalized vectors and
Union-Find for transitive grouping.
"""

from __future__ import annotations

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.node import Stage, ConfigParam
from smart_albums.core.registry import stage
from smart_albums.utils.faiss_grouping import group_by_embedding_similarity
from smart_albums.utils.split import split_contexts


@stage("partition.faiss")
class PartitionFaiss(Stage):
    """Partition assets by FAISS near-duplicate detection.

    Builds a FAISS IndexFlatIP from L2-normalized embedding vectors and uses
    Union-Find to form transitive groups of assets whose cosine similarity
    exceeds the configured threshold.
    """

    _config_schema = (
        ConfigParam(
            key="threshold",
            type=float,
            default=0.95,
            description="Minimum cosine similarity to group assets as near-duplicates.",
        ),
        ConfigParam(
            key="mode",
            type=str,
            default="global",
            choices=["global", "per_partition"],
            description="Grouping mode: 'global' groups all assets together, "
            "'per_partition' groups within existing partition contexts.",
        ),
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        if not ctx.assets:
            return [ctx]

        threshold = self.get("threshold")

        with_emb = [a for a in ctx.assets if a.metadata.get("embedding")]
        without_emb = [a for a in ctx.assets if not a.metadata.get("embedding")]

        if len(with_emb) < 2:
            return [ctx]

        # Build FAISS index and form transitive groups
        groups = group_by_embedding_similarity(with_emb, threshold)

        partitions = list(groups.values())
        for asset in without_emb:
            partitions.append([asset])

        results = split_contexts(ctx, partitions, "faiss")
        if results:
            results[0].stats["partition.faiss.total_assets"] = len(ctx.assets)
            results[0].stats["partition.faiss.groups_formed"] = len(groups)
        return results
