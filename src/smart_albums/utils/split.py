"""Helper for splitting a PipelineContext into multiple child contexts.

Used by all partition.* stages to create isolated child contexts — one per
partition — while sharing configuration.
"""

from __future__ import annotations

import copy
from typing import Optional

from smart_albums.core.context import PipelineContext
from protocols_system.protocols import Asset


def split_contexts(
    ctx: PipelineContext,
    partitions: list[list[Asset]],
    partition_name: str,
) -> list[PipelineContext]:
    """Split a PipelineContext into multiple contexts, one per partition.

    Creates a new child PipelineContext for each partition. Only the assets list is deep-copied
    to ensure mutation isolation. Stats are reset to an empty dict on each child.

    Args:
        ctx: The parent context to split.
        partitions: List of asset groups; each group becomes a child context.
        partition_name: Identifier for this partition level (e.g. "time", "faiss").

    Returns:
        A list of child PipelineContext instances with partition tracking fields
        set. Returns an empty list if partitions is empty.
    """
    if not partitions:
        return []

    new_depth = ctx.partition_depth + 1
    parent_id: Optional[str] = ctx.partition_id or None
    results: list[PipelineContext] = []

    for i, assets in enumerate(partitions):
        # Build a globally unique partition_id by prefixing with the parent's
        # partition_id.  Without this, sibling contexts from different parents
        # would share the same id (e.g. "phash_0") causing debug output and
        # any partition_id-keyed logic to incorrectly conflate them.
        if parent_id:
            child_id = f"{parent_id}__{partition_name}_{i}"
        else:
            child_id = f"{partition_name}_{i}"

        child = PipelineContext(
            config=ctx.config,
            assets=copy.deepcopy(assets),
            stats={},
            partition_name=partition_name,
            partition_id=child_id,
            parent_partition_id=parent_id,
            partition_depth=new_depth,
        )
        results.append(child)

    return results
