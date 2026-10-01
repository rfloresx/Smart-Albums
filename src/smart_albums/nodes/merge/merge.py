"""Merge utilities — combine partitioned contexts back into parent contexts.

Provides merge_concat (the default merge strategy) which concatenates
assets from branch contexts and flattens stats via summation/concatenation.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from smart_albums.core.context import ContextBatch, PipelineContext
from smart_albums.core.node import PipelineNode
from smart_albums.core.registry import stage

logger = logging.getLogger(__name__)

# split_contexts() (smart_albums.utils.split) builds each child's
# partition_id as f"{parent_id}__{partition_name}_{i}", chaining parent ids
# together with this separator. Partition/stage names in this codebase use
# single underscores (e.g. "time_gps", "time_gps_anchor"), never "__", so
# splitting on the *last* occurrence of "__" reliably recovers the
# grandparent id from a child id.
_ID_SEPARATOR = "__"


def _grandparent_id(partition_id: str) -> Optional[str]:
    """Derive the grandparent partition id from a (possibly nested) id.

    E.g. "evenly_0__time_0" -> "evenly_0" (one level up from "time_0",
    which is itself one level up from the assets). A depth-1 id like
    "evenly_0" has no encoded parent, so this returns None.
    """
    if not partition_id or _ID_SEPARATOR not in partition_id:
        return None
    parent, _, _ = partition_id.rpartition(_ID_SEPARATOR)
    return parent or None


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

        # bool is a subclass of int, so a naive `isinstance(v, (int, float))`
        # + sum() turned a flag like {"skipped_empty_pool": True} appearing
        # in two branches into the integer 2 (ND-12). Treat all-bool values
        # as a logical OR instead — "did this happen in any branch" is the
        # meaningful aggregate for a boolean flag, and it preserves the bool
        # type rather than silently promoting it to an int.
        if all(isinstance(v, bool) for v in values):
            result[key] = any(values)
        elif all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            result[key] = sum(values)
        elif all(isinstance(v, list) for v in values):
            merged: list[Any] = []
            for v in values:
                merged.extend(v)
            result[key] = merged
        else:
            # Mixed/unmergeable types (including a bool mixed with ints).
            # First occurrence wins, but log it: silently keeping the first
            # value of a key whose type varies across branches hides a real
            # inconsistency (ND-12).
            if len({type(v) for v in values}) > 1:
                logger.debug(
                    "merge: stats key %r has mixed types across branches %r; "
                    "keeping first value %r",
                    key, [type(v).__name__ for v in values], values[0],
                )
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

    # The merged context's own id becomes the parent id the branches shared
    # (one level up), and *its* parent is derived from that id rather than
    # hardcoded to None. Previously this always set parent_partition_id to
    # None, which is only correct when merging the outermost partition
    # level. With nested partitioning (e.g. an outer partition.evenly
    # wrapping an inner dedup.scenes/dedup.similar composite), that made
    # every intermediate merge collapse straight to "no parent" — the next
    # merge level then grouped by parent_partition_id=None and silently
    # fused contexts from *different* outer partitions into one, destroying
    # the outer partitioning. See utils/split.py, which encodes the full
    # lineage into each child's partition_id as "<parent_id>__<name>_<i>",
    # making it possible to recover the grandparent id here.
    merged_partition_id = first.parent_partition_id or ""
    merged_parent_id = _grandparent_id(merged_partition_id)

    if len(contexts) == 1:
        # Single context — still reduce depth by one level
        return PipelineContext(
            config=first.config,
            assets=first.assets,
            stats=first.stats,
            metadata=first.metadata,
            partition_name=first.partition_name,
            partition_id=merged_partition_id,
            parent_partition_id=merged_parent_id,
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
        partition_id=merged_partition_id,
        parent_partition_id=merged_parent_id,
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
