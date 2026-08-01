"""analyze.embedding — compute and cache vector embeddings for each asset.

Two execution paths are supported depending on the embedding client:

* **Batch path** (preferred): used when the client exposes ``embed_batch()``.
  Uncached assets are fetched concurrently, then handed to the client as a
  single list so the model can process them in batched forward passes. This
  is significantly faster on GPU than the per-asset path.

* **Per-asset path** (fallback): used when the client only implements
  ``embed()``. Assets are processed concurrently up to ``concurrency``
  at a time, each triggering an individual forward pass.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.node import Stage, ConfigParam
from smart_albums.core.protocol_registry import ProtocolsRegistry
from smart_albums.core.protocols import (
    ICacheManager,
    IEmbeddingClient,
    IImageClient,
)
from smart_albums.core.registry import stage

logger = logging.getLogger(__name__)


@stage("analyze.embedding")
class AnalyzeEmbedding(Stage):
    """Compute and cache embeddings for each asset in the context.

    Automatically selects the batch or per-asset execution path based on
    whether the configured embedding client implements ``embed_batch()``.
    """

    _config_schema = (
        ConfigParam(
            "concurrency",
            int,
            default=4,
            min=1,
            description=(
                "Per-asset path: max concurrent embed() calls. "
                "Batch path: max concurrent thumbnail fetches."
            ),
        ),
        ConfigParam(
            "batch_size",
            int,
            default=16,
            min=1,
            description=(
                "Batch path only: number of images per forward pass. "
                "Ignored when the client does not support embed_batch()."
            ),
        ),
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        embedding_client = ProtocolsRegistry.get_instance(IEmbeddingClient)
        if embedding_client is None:
            logger.warning("No embedding_client configured, skipping embedding stage")
            return [ctx]

        if not ctx.assets:
            logger.debug("No assets in context — nothing to embed")
            _zero_stats(ctx)
            return [ctx]

        model: str = embedding_client.embed_model
        concurrency: int = self.get("concurrency")
        batch_size: int = self.get("batch_size")

        logger.debug(
            "analyze.embedding starting: assets=%d, model=%r, concurrency=%d, batch_size=%d",
            len(ctx.assets), model, concurrency, batch_size,
        )

        cache_manager = ProtocolsRegistry.get_instance(ICacheManager)
        cache = None
        if cache_manager:
            cache = cache_manager.get_cache(self.name)
            logger.debug("Cache enabled for stage %r", self.name)
        else:
            logger.debug("No cache_manager on context — cache disabled")

        # embed_batch() is a required method on IEmbeddingClient — always use batch path
        logger.debug("Using batch embedding path")
        await self._run_batch(ctx, model, concurrency, batch_size, cache)

        return [ctx]

    # ------------------------------------------------------------------
    # Batch path
    # ------------------------------------------------------------------

    async def _run_batch(
        self,
        ctx: PipelineContext,
        model: str,
        concurrency: int,
        batch_size: int,
        cache: Any,
    ) -> None:
        """Batch execution path: fetch thumbnails concurrently, embed in bulk.

        Steps:
        1. Serve cache hits immediately.
        2. Fetch thumbnails for all cache-miss assets concurrently (bounded
           by ``concurrency``).
        3. Call ``embed_batch()`` in chunks of ``batch_size``, mapping
           results back to asset IDs.
        4. Store new embeddings in cache.

        Args:
            ctx: The pipeline context whose assets will be mutated.
            model: Cache key string.
            concurrency: Max concurrent thumbnail fetches.
            batch_size: Images per forward pass.
            cache: Optional cache instance.
        """
        image_client = ProtocolsRegistry.get_instance(IImageClient)
        embedding_client = ProtocolsRegistry.get_instance(IEmbeddingClient)

        computed = 0
        cached_count = 0
        failures = 0

        # --- Phase 1: separate cached from uncached ---
        uncached_assets = []
        for asset in ctx.assets:
            cache_key = (asset.id, model)
            if cache is not None:
                cached_emb = cache.get(cache_key)
                if cached_emb is not None:
                    asset.metadata["embedding"] = cached_emb
                    cached_count += 1
                    continue
            uncached_assets.append(asset)

        logger.debug(
            "Batch path phase 1: %d cache hits, %d uncached assets to process",
            cached_count, len(uncached_assets),
        )

        if not uncached_assets:
            logger.debug("All assets served from cache — nothing to compute")
            _write_stats(ctx, computed, cached_count, failures)
            return

        logger.info(
            "Batch embedding: %d assets to compute (%d cache hits).",
            len(uncached_assets),
            cached_count,
        )

        # --- Phase 2: fetch thumbnails concurrently ---
        semaphore = asyncio.Semaphore(concurrency)

        async def _fetch(asset: Any) -> tuple[str, bytes | None]:
            async with semaphore:
                try:
                    if image_client is None:
                        raise RuntimeError("image_client is required for stage 'analyze.embedding'")
                    data = await image_client.get_asset_thumbnail(asset.id)
                    logger.debug(
                        "Thumbnail fetched for asset %s: %d bytes", asset.id, len(data)
                    )
                    return asset.id, data
                except FileNotFoundError:
                    logger.warning(
                        "Skipping asset %s — media file not found on server.",
                        asset.id,
                    )
                    return asset.id, None
                except Exception as exc:
                    logger.warning(
                        "Thumbnail fetch failed for asset %s: %s", asset.id, exc
                    )
                    return asset.id, None

        logger.debug("Batch path phase 2: fetching %d thumbnails (concurrency=%d)", len(uncached_assets), concurrency)
        fetch_results: list[tuple[str, bytes | None]] = await asyncio.gather(
            *[_fetch(a) for a in uncached_assets]
        )

        # Keep only successful fetches, preserving order for result mapping
        valid_ids: list[str] = []
        valid_bytes: list[bytes] = []
        failed_fetch: set[str] = set()

        for asset_id, image_bytes in fetch_results:
            if image_bytes is not None:
                valid_ids.append(asset_id)
                valid_bytes.append(image_bytes)
            else:
                failed_fetch.add(asset_id)
                failures += 1

        # Mark fetch failures on assets
        id_to_asset = {a.id: a for a in uncached_assets}
        for asset_id in failed_fetch:
            id_to_asset[asset_id].metadata["error"] = "thumbnail fetch failed"

        logger.debug(
            "Batch path phase 2 done: %d valid thumbnails, %d fetch failures",
            len(valid_ids), len(failed_fetch),
        )

        if not valid_ids:
            logger.debug("No valid thumbnails — skipping embed phase")
            _write_stats(ctx, computed, cached_count, failures)
            return

        # --- Phase 3: embed in batch_size chunks ---
        logger.debug(
            "Batch path phase 3: embedding %d images in chunks of %d",
            len(valid_ids), batch_size,
        )
        for chunk_start in range(0, len(valid_ids), batch_size):
            chunk_ids = valid_ids[chunk_start : chunk_start + batch_size]
            chunk_bytes = valid_bytes[chunk_start : chunk_start + batch_size]

            logger.debug(
                "Processing chunk [%d:%d] (%d images)",
                chunk_start, chunk_start + len(chunk_ids), len(chunk_ids),
            )

            try:
                if embedding_client is None:
                    raise RuntimeError("embedding_client is required for stage 'analyze.embedding'")
                embeddings: list[list[float] | None] = (
                    await embedding_client.embed_batch(chunk_bytes)
                )
            except Exception as exc:
                logger.warning(
                    "embed_batch failed for chunk at index %d: %s",
                    chunk_start,
                    exc,
                )
                for asset_id in chunk_ids:
                    id_to_asset[asset_id].metadata["error"] = str(exc)
                    failures += 1
                continue

            for asset_id, embedding in zip(chunk_ids, embeddings):
                if embedding is None:
                    logger.warning(
                        "embed_batch returned None for asset %s", asset_id
                    )
                    id_to_asset[asset_id].metadata["error"] = "embedding decode failed"
                    failures += 1
                    continue

                id_to_asset[asset_id].metadata["embedding"] = embedding
                computed += 1
                logger.debug(
                    "Embedded asset %s: vector dim=%d", asset_id, len(embedding)
                )

                if cache is not None:
                    cache.put((asset_id, model), embedding)

        logger.debug(
            "Batch path complete: computed=%d, cached=%d, failures=%d",
            computed, cached_count, failures,
        )
        _write_stats(ctx, computed, cached_count, failures)

    # ------------------------------------------------------------------
    # Per-asset fallback path
    # ------------------------------------------------------------------

    async def _run_sequential(
        self,
        ctx: PipelineContext,
        model: str,
        concurrency: int,
        cache: Any,
    ) -> None:
        """Per-asset execution path using embed() with a concurrency semaphore.

        Args:
            ctx: The pipeline context whose assets will be mutated.
            model: Cache key string.
            concurrency: Max concurrent embed() calls.
            cache: Optional cache instance.
        """
        image_client = ProtocolsRegistry.get_instance(IImageClient)
        embedding_client = ProtocolsRegistry.get_instance(IEmbeddingClient)

        semaphore = asyncio.Semaphore(concurrency)
        computed = 0
        cached_count = 0
        failures = 0
        lock = asyncio.Lock()

        async def _process(asset: Any) -> None:
            nonlocal computed, cached_count, failures
            cache_key = (asset.id, model)
            async with semaphore:
                if cache is not None:
                    cached_emb = cache.get(cache_key)
                    if cached_emb is not None:
                        logger.debug("Sequential path cache HIT for asset %s", asset.id)
                        asset.metadata["embedding"] = cached_emb
                        async with lock:
                            cached_count += 1
                        return

                try:
                    logger.debug("Sequential path: fetching thumbnail for asset %s", asset.id)
                    if image_client is None:
                        raise RuntimeError("image_client is required for stage 'analyze.embedding'")
                    image_bytes = await image_client.get_asset_thumbnail(asset.id)
                    logger.debug(
                        "Sequential path: embedding asset %s (%d bytes)",
                        asset.id, len(image_bytes),
                    )
                    if embedding_client is None:
                        raise RuntimeError("embedding_client is required for stage 'analyze.embedding'")
                    embedding = await embedding_client.embed(image_bytes)
                    asset.metadata["embedding"] = embedding
                    logger.debug(
                        "Embedded asset %s: vector dim=%d", asset.id, len(embedding)
                    )
                    async with lock:
                        computed += 1
                    if cache is not None:
                        cache.put(cache_key, embedding)
                except FileNotFoundError:
                    logger.warning(
                        "Skipping asset %s — media file not found on server.",
                        asset.id,
                    )
                    asset.metadata["error"] = "media not found"
                    async with lock:
                        failures += 1
                except Exception as exc:
                    logger.warning(
                        "Embedding failed for asset %s: %s", asset.id, exc
                    )
                    asset.metadata["error"] = str(exc)
                    async with lock:
                        failures += 1

        await asyncio.gather(*[_process(a) for a in ctx.assets])

        logger.debug(
            "Sequential path complete: computed=%d, cached=%d, failures=%d",
            computed, cached_count, failures,
        )
        _write_stats(ctx, computed, cached_count, failures)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _zero_stats(ctx: PipelineContext) -> None:
    ctx.stats["analyze.embedding.total_assets"] = 0
    ctx.stats["analyze.embedding.computed"] = 0
    ctx.stats["analyze.embedding.cached"] = 0
    ctx.stats["analyze.embedding.failures"] = 0


def _write_stats(
    ctx: PipelineContext,
    computed: int,
    cached: int,
    failures: int,
) -> None:
    ctx.stats["analyze.embedding.total_assets"] = len(ctx.assets)
    ctx.stats["analyze.embedding.computed"] = computed
    ctx.stats["analyze.embedding.cached"] = cached
    ctx.stats["analyze.embedding.failures"] = failures
