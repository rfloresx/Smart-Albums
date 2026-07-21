"""select.top_n — select the top N assets from the context by quality score.

A simple selection stage that keeps the N highest-scoring assets.
Useful as a quick way to cap output size without temporal balancing
or diversity logic.

Sorting is by ``metadata["score"]`` descending, with file size as
tie-breaker. Assets without a score are ranked last.
"""

from __future__ import annotations

import logging
from typing import Any

from smart_albums.core.context import ContextBatch, PipelineContext
from smart_albums.core.models import Asset
from smart_albums.core.node import ConfigParam, Stage
from smart_albums.core.registry import stage

logger = logging.getLogger(__name__)


def _sort_key(asset: Asset) -> tuple[float, int]:
    """Return (score desc, file_size desc) for sorting."""
    score = asset.metadata.get("score", 0.0)
    exif = asset.metadata.get("exif")
    file_size = 0
    if isinstance(exif, dict):
        size = exif.get("file_size_in_byte")
        if size is not None:
            file_size = int(size)
    return (-score, -file_size)


@stage("select.top_n")
class SelectTopN(Stage):
    """Select the top N assets from the context ranked by quality score.

    Keeps up to ``count`` assets, sorted by descending score. When two
    assets share the same score, the one with the larger file size wins.

    Config:
        count: Maximum number of assets to keep. Defaults to 10.
    """

    _description = "Select top N images by quality score"

    _config_schema = (
        ConfigParam(
            "count",
            int,
            default=10,
            min=1,
            description="Maximum number of assets to select.",
        ),
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        count: int = self.get("count")

        if not ctx.assets:
            ctx.stats["select.top_n.input"] = 0
            ctx.stats["select.top_n.selected"] = 0
            return [ctx]

        total = len(ctx.assets)
        sorted_assets = sorted(ctx.assets, key=_sort_key)
        ctx.assets = sorted_assets[:count]

        ctx.stats["select.top_n.input"] = total
        ctx.stats["select.top_n.selected"] = len(ctx.assets)

        logger.info(
            "select.top_n: kept %d of %d assets (count=%d)",
            len(ctx.assets),
            total,
            count,
        )

        return [ctx]
