"""analyze.phash — compute perceptual hash for each asset.

Fetches the thumbnail for each asset and computes a 64-bit DCT-based
perceptual hash using the imagehash library. The pHash is stored in
``asset.metadata["phash"]`` as a hex string.

Assets that already have a phash (e.g. from a cache hit in analyze.score)
are skipped. Assets whose thumbnails cannot be fetched get an empty string.
"""

from __future__ import annotations

import asyncio
import io
import logging
from typing import Any

import imagehash
from PIL import Image

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.node import Stage, ConfigParam
from smart_albums.core.protocols import ICacheManager, IImageClient, IProgressReporter, ProtocolsRegistry
from smart_albums.core.registry import stage

logger = logging.getLogger(__name__)


def _compute_phash(image_bytes: bytes) -> str:
    """Compute a 64-bit perceptual hash (hex string) from image bytes.

    Uses the imagehash library's pHash algorithm (DCT-based) which produces
    hashes that are resilient to resizing, compression, and minor edits.

    Args:
        image_bytes: Raw image bytes (JPEG, PNG, etc.)

    Returns:
        Hex-encoded 64-bit perceptual hash string.
        Returns empty string if computation fails.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        return str(imagehash.phash(img))
    except Exception as exc:
        logger.warning("Failed to compute pHash: %s", exc)
        return ""


@stage("analyze.phash")
class AnalyzePHash(Stage):
    """Compute perceptual hash for each asset's thumbnail.

    Fetches thumbnails concurrently and computes a 64-bit DCT-based pHash.
    Stores the result in ``asset.metadata["phash"]``. Skips assets that
    already have a phash value set.
    """

    _config_schema = (
        ConfigParam(
            "concurrency",
            int,
            default=4,
            min=1,
            description="Maximum concurrent thumbnail fetches.",
        ),
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        if not ctx.assets:
            return [ctx]

        concurrency: int = self.get("concurrency")

        cache_manager = ProtocolsRegistry.get_instance(ICacheManager)
        cache = None
        if cache_manager:
            cache = cache_manager.get_cache(self.name)

        progress = ProtocolsRegistry.get_instance(IProgressReporter)
        if progress:
            progress.start_stage("Computing pHash", total=len(ctx.assets))

        image_client = ProtocolsRegistry.get_instance(IImageClient)

        semaphore = asyncio.Semaphore(concurrency)
        computed = 0
        cached_count = 0
        skipped = 0
        failures = 0
        lock = asyncio.Lock()

        async def _process_one(asset: Any) -> None:
            nonlocal computed, cached_count, skipped, failures

            # Skip if phash already present (from upstream cache)
            if asset.metadata.get("phash"):
                async with lock:
                    skipped += 1
                if progress:
                    progress.advance()
                return

            async with semaphore:
                # Check cache
                cache_key = asset.id
                if cache is not None:
                    hit = cache.get(cache_key)
                    if hit is not None:
                        asset.metadata["phash"] = hit
                        async with lock:
                            cached_count += 1
                        if progress:
                            progress.advance()
                        return

                try:
                    if image_client is None:
                        raise RuntimeError("image_client is required for stage 'analyze.phash'")
                    thumbnail = await image_client.get_asset_thumbnail(asset.id)
                    phash = _compute_phash(thumbnail)
                    asset.metadata["phash"] = phash

                    if cache is not None and phash:
                        cache.put(cache_key, phash)

                    async with lock:
                        computed += 1

                except Exception as exc:
                    logger.warning(
                        "analyze.phash: failed for asset %s: %s", asset.id, exc
                    )
                    asset.metadata["phash"] = ""
                    async with lock:
                        failures += 1

            if progress:
                progress.advance()

        await asyncio.gather(*[_process_one(a) for a in ctx.assets])

        if progress:
            progress.finish_stage()

        ctx.stats["analyze.phash.total_assets"] = len(ctx.assets)
        ctx.stats["analyze.phash.computed"] = computed
        ctx.stats["analyze.phash.cached"] = cached_count
        ctx.stats["analyze.phash.skipped"] = skipped
        ctx.stats["analyze.phash.failures"] = failures

        return [ctx]
