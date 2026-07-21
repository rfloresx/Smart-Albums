"""partition.phash — partition assets by pHash Hamming distance.

Uses Union-Find to transitively group assets whose perceptual hashes are
within a configurable Hamming distance threshold.  Each group becomes a
separate partition context.  Assets without a pHash value receive their own
individual contexts.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.models import Asset
from smart_albums.core.node import Stage, ConfigParam
from smart_albums.utils.phash_utils import hamming_distance as _hamming_distance
from smart_albums.core.registry import stage
from smart_albums.utils.split import split_contexts
from smart_albums.utils.union_find import UnionFind

logger = logging.getLogger(__name__)


def _union_find_phash(
    assets: list[Asset], threshold: float
) -> dict[int, list[Asset]]:
    """Group assets by pHash Hamming distance using Union-Find.

    Compares all pairs (O(N^2)) — acceptable for the expected small group sizes
    within a partition context.

    Args:
        assets: Assets with valid phash metadata.
        threshold: Similarity threshold in [0.0, 1.0]. Two assets are
            considered duplicates when their pHash Hamming distance is
            ≤ int((1.0 - threshold) * 64).

    Returns:
        A dict mapping root index → list of assets in that group.
    """
    max_distance = int((1.0 - threshold) * 64)
    n = len(assets)
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            h1 = assets[i].metadata["phash"]
            h2 = assets[j].metadata["phash"]
            if _hamming_distance(h1, h2) <= max_distance:
                uf.union(i, j)

    groups: dict[int, list[Asset]] = defaultdict(list)
    for i in range(n):
        root = uf.find(i)
        groups[root].append(assets[i])

    return groups


# ---------------------------------------------------------------------------
# Stage
# ---------------------------------------------------------------------------


@stage("partition.phash")
class PartitionPHash(Stage):
    """Partition assets by pHash Hamming distance (exact/near-exact duplicates).

    Assets with a pHash are grouped transitively via Union-Find: if A is within
    threshold of B, and B is within threshold of C, then {A, B, C} form one
    group.  Each group becomes a partition context.

    Assets without a pHash value are placed in their own individual contexts.
    """

    _config_schema = (
        ConfigParam(
            key="threshold",
            type=float,
            default=0.90,
            min=0.0,
            max=1.0,
            description="Similarity threshold in [0.0, 1.0]. Two assets are duplicates "
            "when their pHash Hamming distance is ≤ int((1.0 - threshold) * 64).",
        ),
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        if not ctx.assets:
            return [ctx]

        threshold: float = self.get("threshold")
        with_phash = [a for a in ctx.assets if a.metadata.get("phash")]
        without_phash = [a for a in ctx.assets if not a.metadata.get("phash")]

        if len(with_phash) < 2:
            return [ctx]

        # Union-Find grouping
        groups = _union_find_phash(with_phash, threshold)

        # Each group + individual contexts for assets without phash
        partitions: list[list[Asset]] = list(groups.values())
        for asset in without_phash:
            partitions.append([asset])

        results = split_contexts(ctx, partitions, "phash")
        if results:
            results[0].stats["partition.phash.total_assets"] = len(ctx.assets)
            results[0].stats["partition.phash.groups_formed"] = len(partitions)
        return results
