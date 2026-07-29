"""Merge utilities — combine partitioned contexts back into parent contexts.

Provides merge_concat (the default merge strategy) which concatenates
assets from branch contexts and flattens stats via summation/concatenation.
"""

from __future__ import annotations

from typing import Any

from smart_albums.core.context import ContextBatch, PipelineContext
from smart_albums.core.node import PipelineNode
from smart_albums.core.registry import stage


def _flatten_stats(stats_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge stats dicts from multiple branch contexts into one.

    Aggregation rules:
    - Numeric (int/float) values: summed across branches.
    - List values: concatenated in order.
    - Other types: first occurrence wins.

    Args:
        stats_list: List of stats dicts, one per branch context.

    Returns:
        A single merged stats dict.
    """
    if not stats_list:
        return {}

    result: dict[str, Any] = {}

    # Collect all keys in order of first appearance
    all_keys: list[str] = []
    seen: set[str] = set()
    for stats in stats_list:
        for key in stats:
            if key not in seen:
                all_keys.append(key)
                seen.add(key)

    for key in all_keys:
        values = [s[key] for s in stats_list if key in s]
        if not values:
            continue

        if all(isinstance(v, (int, float)) for v in values):
            result[key] = sum(values)
        elif all(isinstance(v, list) for v in values):
            merged: list[Any] = []
            for v in values:
                merged.extend(v)
            result[key] = merged
        else:
            # Non-mergeable: first occurrence wins
            result[key] = values[0]

    return result


def merge_concat(contexts: ContextBatch) -> PipelineContext:
    """Merge multiple branch contexts into a single context.

    Assets are concatenated in order. Stats are flattened via _flatten_stats.
    The first context's config, clients, and progress reporter are preserved.
    The resulting context's partition depth is reduced by one level.

    Args:
        contexts: Branch contexts to merge (must be non-empty).

    Returns:
        A single merged PipelineContext.
    """
    if not contexts:
        raise ValueError("Cannot merge an empty context batch")

    first = contexts[0]

    if len(contexts) == 1:
        # Single context — still reduce depth by one level
        return PipelineContext(
            config=first.config,
            assets=first.assets,
            stats=first.stats,
            metadata=first.metadata,
            partition_name=first.partition_name,
            partition_id=first.parent_partition_id or "",
            parent_partition_id=None,
            partition_depth=max(0, first.partition_depth - 1),
        )

    # Concatenate all assets in partition order
    merged_assets = []
    for ctx in contexts:
        merged_assets.extend(ctx.assets)

    # Flatten stats from all branches
    merged_stats = _flatten_stats([ctx.stats for ctx in contexts])

    # The merged context inherits the parent's partition tracking
    # (one level up from the branches)
    return PipelineContext(
        config=first.config,
        assets=merged_assets,
        stats=merged_stats,
        metadata=first.metadata,
        partition_name=first.partition_name,
        partition_id=first.parent_partition_id or "",
        parent_partition_id=None,
        partition_depth=max(0, first.partition_depth - 1),
    )


@stage("merge.concat")
class MergeConcat(PipelineNode):
    """Merge partitioned contexts by collapsing one partition level.

    Groups contexts by parent_partition_id, merges each group into a
    single context via asset concatenation and stats flattening, then
    returns the reduced batch.

    If all contexts are at depth 0 (no partitioning), they are all
    merged into a single context.
    """

    _config_schema = ()

    async def func(self, contexts: ContextBatch) -> ContextBatch:
        if not contexts:
            return contexts

        # If no partitioning depth, merge everything into one
        max_depth = max(ctx.partition_depth for ctx in contexts)
        if max_depth == 0:
            return [merge_concat(contexts)]

        # Group by parent_partition_id and merge each group
        groups: dict[str | None, list[PipelineContext]] = {}
        for ctx in contexts:
            key = ctx.parent_partition_id
            groups.setdefault(key, []).append(ctx)

        results: ContextBatch = ContextBatch()
        for group in groups.values():
            results.append(merge_concat(group))

        return results
