"""Dedup stage that collapses near-duplicate groups by perceptual hash similarity."""

from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta

from smart_albums.core.context import PipelineContext, ContextBatch
from protocols_system.protocols import Asset
from smart_albums.core.node import Stage, ConfigParam
from smart_albums.utils.phash_utils import hamming_distance as _hamming_distance
from smart_albums.core.registry import stage
from smart_albums.utils.union_find import AnchoredUnionFind


def _find_duplicate_groups(
    assets: list[Asset], max_distance: int
) -> list[list[Asset]]:
    """Find groups of near-duplicate assets using Union-Find.

    Buckets assets by calendar day (a cheap way to avoid an all-pairs O(n²)
    comparison across an entire year), then clusters by pHash Hamming
    distance — comparing each day's bucket against both itself and the
    *next* calendar day. Only assets with a non-empty ``phash`` in metadata
    are eligible for grouping.

    Comparing adjacent days (not just within a single day) fixes the
    midnight-boundary bug (ND-14): a burst of shots taken around 23:59 and
    00:01 landed in two different calendar-day buckets and so were never
    compared to each other, leaving an obvious duplicate pair un-deduped
    purely because of where the clock happened to tick over.

    Uses an anchor-bounded union-find (ND-10) so a chain of gradually
    drifting near-duplicates (A~B, B~C, C~D, ...) can't collapse into one
    unbounded group — each candidate must also match the group's original
    anchor, not just its immediate neighbor.

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

    # Group eligible asset indices by calendar day. Keep the untimed
    # bucket separate (it has no neighbor day to compare against).
    _UNTIMED = "_untimed"
    day_buckets: dict[str, list[int]] = defaultdict(list)
    for i in eligible_indices:
        captured_at = assets[i].captured_at
        if captured_at is not None:
            day_key = captured_at.date().isoformat()
        else:
            day_key = _UNTIMED
        day_buckets[day_key].append(i)

    # Union-Find over asset indices
    # Map eligible indices to a contiguous 0..N-1 range for UnionFind
    idx_to_pos: dict[int, int] = {idx: pos for pos, idx in enumerate(eligible_indices)}
    hashes = {idx: assets[idx].metadata["phash"] for idx in eligible_indices}
    uf = AnchoredUnionFind(len(eligible_indices))

    def _similar(pos_a: int, pos_b: int) -> bool:
        idx_a = eligible_indices[pos_a]
        idx_b = eligible_indices[pos_b]
        return _hamming_distance(hashes[idx_a], hashes[idx_b]) <= max_distance

    def _compare_within(indices: list[int]) -> None:
        for a in range(len(indices)):
            for b in range(a + 1, len(indices)):
                pos_i, pos_j = idx_to_pos[indices[a]], idx_to_pos[indices[b]]
                if _similar(pos_i, pos_j):
                    uf.try_union(pos_i, pos_j, _similar)

    # Only bridge the midnight boundary for assets actually close in time
    # to it — a shot at 23:58 and one at 00:02 belong to the same burst,
    # but two shots a full 24h apart (both at midnight) do not. Without
    # this window, comparing whole adjacent-day buckets would group photos
    # that merely happen to fall on consecutive days.
    _BOUNDARY_WINDOW = timedelta(hours=1)

    def _near_boundary_end(idx: int) -> bool:
        """True if the asset is within the window *before* its day's end."""
        captured = assets[idx].captured_at
        if captured is None:
            return False
        day_end = captured.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        return (day_end - captured) <= _BOUNDARY_WINDOW

    def _near_boundary_start(idx: int) -> bool:
        """True if the asset is within the window *after* its day's start."""
        captured = assets[idx].captured_at
        if captured is None:
            return False
        day_start = captured.replace(hour=0, minute=0, second=0, microsecond=0)
        return (captured - day_start) <= _BOUNDARY_WINDOW

    def _compare_across(indices_a: list[int], indices_b: list[int]) -> None:
        # indices_a are from the earlier day, indices_b from the next day.
        late = [i for i in indices_a if _near_boundary_end(i)]
        early = [j for j in indices_b if _near_boundary_start(j)]
        for i in late:
            for j in early:
                pos_i, pos_j = idx_to_pos[i], idx_to_pos[j]
                if _similar(pos_i, pos_j):
                    uf.try_union(pos_i, pos_j, _similar)

    for day_key, indices in day_buckets.items():
        _compare_within(indices)
        if day_key == _UNTIMED:
            continue
        # Also compare against the following calendar day's bucket so a
        # burst straddling midnight is still grouped (ND-14), but only for
        # assets within _BOUNDARY_WINDOW of the shared midnight.
        try:
            next_day = (date.fromisoformat(day_key) + timedelta(days=1)).isoformat()
        except ValueError:  # pragma: no cover - day_key is always ISO here
            continue
        next_bucket = day_buckets.get(next_day)
        if next_bucket:
            _compare_across(indices, next_bucket)

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
