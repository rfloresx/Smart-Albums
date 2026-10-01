"""Fork utilities — execute multiple branches from the same input.

Provides two fork strategies:

- Fork: sends a full copy of the input to every branch (parallel fan-out).
- ForkBySelection: runs N branches sequentially. Each branch receives
  the assets not selected by any prior branch (cascading remainder).
"""

from __future__ import annotations

import asyncio
import copy
import logging
from dataclasses import dataclass
from typing import Any

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.node import PipelineNode, CompositeParam
from smart_albums.core.spec import Pipeline
from smart_albums.core.registry import stage

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Branch:
    name: str
    pipeline: Pipeline


def _clone_contexts(contexts: ContextBatch) -> ContextBatch:
    """Create isolated copies of contexts for branch execution.

    Deep-copies assets, stats, and metadata for mutation isolation.
    """
    clones: ContextBatch = []
    for ctx in contexts:
        clones.append(PipelineContext(
            config=ctx.config,
            assets=copy.deepcopy(ctx.assets),
            stats=dict(ctx.stats),
            metadata=dict(ctx.metadata),
            partition_name=ctx.partition_name,
            partition_id=ctx.partition_id,
            parent_partition_id=ctx.parent_partition_id,
            partition_depth=ctx.partition_depth,
        ))
    return clones


def _resolve_branches(node: "Fork") -> list[tuple[str, Pipeline]]:
    """Extract (name, pipeline) pairs from a Fork instance."""
    return [(b.name, b.pipeline) for b in node.branches]


def _build_branches(config: dict[str, list[Any]] | list[Any]) -> dict[str, Any]:
    """Build Fork constructor kwargs from a JSON config value.

    Accepts two formats:
    - Dict: {"curation": [...pipeline...], "remainder": [...pipeline...]}
    - List of pairs: [["curation", [...pipeline...]], ["remainder", [...pipeline...]]]

    Returns kwargs to pass to Fork.__init__.
    """
    if isinstance(config, list):
        # List of [name, pipeline] pairs
        branches = [
            Branch(name=item[0], pipeline=item[1])
            for item in config
        ]
    else:
        # Dict of {name: pipeline}
        branches = [
            Branch(name=name, pipeline=pipeline)
            for name, pipeline in config.items()
        ]
    return {"branches": branches}


@stage("fork")
class Fork(PipelineNode):
    """Execute multiple pipelines against the same input contexts.

    Each branch receives a copy of the incoming contexts.

    Outputs from all branches are flattened into a single
    ContextBatch collection.
    """

    _config_schema = (
        CompositeParam(
            key="branches",
            description="Child pipelines to execute.",
            resolve=_resolve_branches,
            build=_build_branches,
        ),
    )

    def __init__(
        self,
        *args: Branch,
        branches: list[Branch | Pipeline] | None = None,
    ):
        super().__init__()
        if branches is None:
            branches = list(args)
        # Accept raw pipeline lists as shorthand for Branch objects
        self.branches: list[Branch] = [
            b if isinstance(b, Branch) else Branch(name=f"branch_{i}", pipeline=b)
            for i, b in enumerate(branches)
        ]

    async def func(
        self,
        contexts: ContextBatch,
    ) -> ContextBatch:

        # Avoid circular import
        from smart_albums.core.runner import run_pipeline

        total_assets = sum(len(ctx.assets) for ctx in contexts)
        logger.debug(
            "Fork starting %d branches (%d ctx, %d assets)",
            len(self.branches),
            len(contexts),
            total_assets,
        )

        tasks = []

        for branch in self.branches:

            branch_contexts = _clone_contexts(contexts)

            for ctx in branch_contexts:
                ctx.metadata["branch"] = branch.name

            logger.debug(
                "Fork branch '%s': %d steps",
                branch.name,
                len(branch.pipeline),
            )

            tasks.append(
                run_pipeline(
                    branch.pipeline,
                    branch_contexts,
                )
            )

        results = await asyncio.gather(*tasks)

        merged: ContextBatch = []

        for i, result in enumerate(results):
            branch_assets = sum(len(ctx.assets) for ctx in result)
            logger.debug(
                "Fork branch '%s' produced %d ctx, %d assets",
                self.branches[i].name,
                len(result),
                branch_assets,
            )
            merged.extend(result)

        return merged


@stage("fork.by_selection")
class ForkBySelection(Fork):
    """Fork where each branch receives assets NOT selected by prior branches.

    Execution flow (sequential, N branches):
    1. Run branch 1 on the full input. Collect the asset IDs it outputs.
    2. Remove those IDs from the remaining pool.
    3. Run branch 2 on the remainder. Collect its output IDs.
    4. Remove those IDs from the remaining pool.
    5. … repeat for all N branches.
    6. Return the merged results from all branches.

    This enables patterns like "run the main curation pipeline, then do
    something different with the leftovers" — generalized to N tiers.
    """

    async def func(
        self,
        contexts: ContextBatch,
    ) -> ContextBatch:
        from smart_albums.core.runner import run_pipeline

        total_assets = sum(len(ctx.assets) for ctx in contexts)
        logger.debug(
            "ForkBySelection starting %d branches (%d ctx, %d assets)",
            len(self.branches),
            len(contexts),
            total_assets,
        )

        merged: ContextBatch = []
        remaining_contexts = _clone_contexts(contexts)

        for branch in self.branches:
            remaining_assets = sum(len(ctx.assets) for ctx in remaining_contexts)
            logger.debug(
                "ForkBySelection branch '%s': %d assets remaining",
                branch.name,
                remaining_assets,
            )

            # Prepare input for this branch
            branch_contexts = _clone_contexts(remaining_contexts)
            for ctx in branch_contexts:
                ctx.metadata["branch"] = branch.name

            # Run the branch pipeline
            branch_results = await run_pipeline(
                branch.pipeline,
                branch_contexts,
            )
            merged.extend(branch_results)

            # Compute which asset IDs were produced by this branch
            selected_ids: set[str] = set()
            for ctx in branch_results:
                for asset in ctx.assets:
                    selected_ids.add(asset.id)

            logger.debug(
                "ForkBySelection branch '%s' selected %d assets",
                branch.name,
                len(selected_ids),
            )

            # Remove selected assets from the remaining pool
            for ctx in remaining_contexts:
                ctx.assets = [a for a in ctx.assets if a.id not in selected_ids]

        return merged
