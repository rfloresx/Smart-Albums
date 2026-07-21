"""Dedup stage that collapses near-duplicate groups by perceptual hash similarity."""

from __future__ import annotations

from collections import defaultdict

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.models import Asset
from smart_albums.core.node import Stage, ConfigParam
from smart_albums.utils.phash_utils import hamming_distance as _hamming_distance
from smart_albums.core.registry import stage
from smart_albums.utils.union_find import UnionFind


def _find_duplicate_groups(
    assets: list[Asset], max_distance: int
) -> list[list[Asset]]:
    """Find groups of near-duplicate assets using Union-Find.

    Groups assets by calendar day, then clusters within each day by pHash
    Hamming distance. Only assets with a non-empty ``phash`` in metadata are
    eligible for grouping.

    Returns only groups with 2+ members (singletons are not duplicates).
    """
    # Filter to assets that have a usable phash
    eligible_indices: list[int] = []
    for i, asset in enumerate(assets):
        phash = asset.metadata.get("phash")
        if phash:
            eligible_indices.append(i)

    if not eligible_indices:
        return []

    # Group eligible asset indices by calendar day
    day_buckets: dict[str, list[int]] = defaultdict(list)
    for i in eligible_indices:
        if assets[i].captured_at is not None:
            day_key = assets[i].captured_at.strftime("%Y-%m-%d")
        else:
            day_key = "_untimed"
        day_buckets[day_key].append(i)

    # Union-Find over asset indices
    # Map eligible indices to a contiguous 0..N-1 range for UnionFind
    idx_to_pos: dict[int, int] = {idx: pos for pos, idx in enumerate(eligible_indices)}
    uf = UnionFind(len(eligible_indices))

    # Compare pairs within the same day
    for indices in day_buckets.values():
        for idx_a in range(len(indices)):
            i = indices[idx_a]
            phash_i = assets[i].metadata["phash"]
            for idx_b in range(idx_a + 1, len(indices)):
                j = indices[idx_b]
                phash_j = assets[j].metadata["phash"]
                if _hamming_distance(phash_i, phash_j) <= max_distance:
                    uf.union(idx_to_pos[i], idx_to_pos[j])

    # Collect non-singleton clusters
    clusters: dict[int, list[int]] = defaultdict(list)
    for i in eligible_indices:
        clusters[uf.find(idx_to_pos[i])].append(i)

    groups: list[list[Asset]] = []
    for members in clusters.values():
        if len(members) >= 2:
            groups.append([assets[i] for i in members])

    return groups


@stage("dedup.phash")
class DedupPHash(Stage):
    """Collapse near-duplicate groups, keeping the best member."""

    _config_schema = (
        ConfigParam(
            "threshold",
            float,
            default=0.90,
            min=0.0,
            max=1.0,
            description="Hamming similarity. Higher = stricter.",
        ),
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        threshold = self.get("threshold")
        max_distance = int((1.0 - threshold) * 64)

        groups = _find_duplicate_groups(ctx.assets, max_distance)
        excluded_ids: set[str] = set()

        for gid, group in enumerate(groups):
            # Tag every member with group_id
            for asset in group:
                asset.metadata["group_id"] = f"phash:{gid}"

            # Pick the best: highest score, tiebreak by lex-smallest id
            best = max(group, key=lambda a: a.metadata.get("score", 0.0))
            best_score = best.metadata.get("score", 0.0)
            candidates = [
                a for a in group if a.metadata.get("score", 0.0) == best_score
            ]
            if len(candidates) > 1:
                best = min(candidates, key=lambda a: a.id)

            for asset in group:
                if asset.id != best.id:
                    excluded_ids.add(asset.id)

        before = len(ctx.assets)
        ctx.assets = [a for a in ctx.assets if a.id not in excluded_ids]
        ctx.stats["dedup.phash.excluded"] = before - len(ctx.assets)
        return [ctx]
