"""select.best — select best representative from a partition context.

Operates on a single partition context (after partition stages have split
into groups). Computes a composite quality score (sharpness + analysis score)
for each asset and picks the best. Falls back to score-only ranking when
thumbnails are unavailable.

The composite scoring replicates the original BestShotScorer logic:
- 50% weight: normalized image sharpness (Laplacian variance)
- 50% weight: LLM analysis score
- Tie-break: largest file size (inferred from filename or metadata)
"""

from __future__ import annotations

import io
import logging
from typing import Optional

import numpy as np
from PIL import Image, ImageFilter

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.models import Asset
from smart_albums.core.node import Stage
from smart_albums.core.protocol_registry import ProtocolsRegistry
from smart_albums.core.protocols import IImageClient
from smart_albums.core.registry import stage

logger = logging.getLogger(__name__)

# Maximum expected Laplacian variance for normalization.
# Values above this are clamped to 1.0.
_SHARPNESS_MAX = 1000.0

# Weight for sharpness and analysis score in the composite.
_SHARPNESS_WEIGHT = 0.5
_ANALYSIS_WEIGHT = 1.0 - _SHARPNESS_WEIGHT


def _compute_sharpness(image_bytes: bytes) -> float:
    """Compute normalized sharpness via Laplacian variance.

    Opens the image, converts to grayscale, applies a Laplacian filter,
    and computes the variance of the response. Normalized to [0, 1].

    Args:
        image_bytes: Raw image bytes.

    Returns:
        Normalized sharpness in [0.0, 1.0]. Returns 0.0 on failure.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("L")
        laplacian = img.filter(
            ImageFilter.Kernel(
                (3, 3),
                [0, 1, 0, 1, -4, 1, 0, 1, 0],
                scale=1,
                offset=128,
            )
        )
        arr = np.array(laplacian, dtype=np.float64) - 128.0
        variance = float(np.var(arr))
        return min(variance / _SHARPNESS_MAX, 1.0)
    except Exception:
        return 0.0


def _get_file_size(asset: Asset) -> int:
    """Extract file size from asset metadata (EXIF or other sources).

    Returns 0 if file size is not available.
    """
    exif = asset.metadata.get("exif")
    if isinstance(exif, dict):
        size = exif.get("file_size_in_byte")
        if size is not None:
            return int(size)
    return 0


@stage("select.best")
class SelectBest(Stage):
    """Select the best representative from the current context.

    Uses a composite quality score: 50% normalized sharpness (Laplacian
    variance of the thumbnail) + 50% LLM analysis score. Tie-break by
    largest file size.

    When the image client is unavailable or thumbnail fetches fail, falls
    back to score-only ranking with file size tie-break.

    Sets ``metadata["group_best"] = True`` on the selected asset and narrows
    the context to contain only that asset.
    """

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        if not ctx.assets:
            ctx.stats["select.best.groups_processed"] = 0
            ctx.stats["select.best.representatives_selected"] = 0
            return [ctx]

        if len(ctx.assets) == 1:
            ctx.assets[0].metadata["group_best"] = True
            ctx.stats["select.best.groups_processed"] = 1
            ctx.stats["select.best.representatives_selected"] = 1
            return [ctx]

        # Try composite scoring with sharpness
        best = await self._select_composite(ctx)
        best.metadata["group_best"] = True

        ctx.assets = [best]
        ctx.stats["select.best.groups_processed"] = 1
        ctx.stats["select.best.representatives_selected"] = 1

        return [ctx]

    async def _select_composite(self, ctx: PipelineContext) -> Asset:
        """Compute composite scores and return the best asset.

        Falls back to score-only if thumbnails cannot be fetched.
        """
        image_client = ProtocolsRegistry.get_instance(IImageClient)
        scored: list[tuple[Asset, float, int]] = []

        for asset in ctx.assets:
            analysis_score = asset.metadata.get("score", 0.0)
            sharpness = 0.0

            # Try to compute sharpness from thumbnail
            if image_client is not None:
                try:
                    thumbnail = await image_client.get_asset_thumbnail(asset.id)
                    sharpness = _compute_sharpness(thumbnail)
                except Exception:
                    logger.debug(
                        "Could not fetch thumbnail for sharpness of asset %s",
                        asset.id,
                    )

            composite = _SHARPNESS_WEIGHT * sharpness + _ANALYSIS_WEIGHT * analysis_score
            file_size = _get_file_size(asset)
            scored.append((asset, composite, file_size))

        # Sort: highest composite, then largest file size, then smallest id
        scored.sort(key=lambda t: (-t[1], -t[2], t[0].id))
        return scored[0][0]
