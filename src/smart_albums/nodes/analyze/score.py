"""analyze.score — score every asset using a vision LLM."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
from typing import Any

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.node import Stage, ConfigParam
from protocols_system import ProtocolsRegistry
from protocols_system.protocols import (
    ICacheManager,
    IImageClient,
    ILLMClient,
    IProgressReporter,
)
from smart_albums.core.registry import stage

logger = logging.getLogger(__name__)


@stage("analyze.score")
class AnalyzeScore(Stage):
    """Score every asset using a vision LLM."""

    _config_schema = (
        ConfigParam("concurrency", int, default=4, min=1),
        ConfigParam(
            "prompt_file",
            str | None,
            default=None,
            description="Path to the scoring prompt file.",
            widget={"type": "file", "glob": "*.md", "source": "prompts"},
        ),
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        concurrency: int = self.get("concurrency")
        prompt_file: str = self.get("prompt_file") or ""

        if prompt_file:
            from pathlib import Path

            prompt_path = Path(prompt_file)
            # Security: reject paths that contain traversal sequences
            if ".." in prompt_path.parts:
                raise ValueError(
                    f"prompt_file must not contain directory traversal '..'. "
                    f"Got: {prompt_file!r}"
                )

            try:
                with open(prompt_file, "r") as f:
                    prompt = f.read()
            except FileNotFoundError:
                raise ValueError(
                    f"Prompt file not found: {prompt_file!r}. "
                    "Check the prompt_file path in your config."
                )
            except OSError as exc:
                raise ValueError(
                    f"Cannot read prompt file {prompt_file!r}: {exc}"
                )
        else:
            raise ValueError("No prompt provided")

        # Hash prompt *content* so the cache key is stable across renames/moves
        prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()[:8]

        logger.debug(
            "analyze.score starting: assets=%d, concurrency=%d, prompt_file=%r, prompt_hash=%s",
            len(ctx.assets), concurrency, prompt_file, prompt_hash,
        )
        logger.debug(
            "Loaded prompt from %s (%s), length=%d chars",
            prompt_file, prompt_hash, len(prompt),
        )

        cache_manager = ProtocolsRegistry.get_instance(ICacheManager)
        cache = None
        if cache_manager:
            cache = cache_manager.get_cache(self.name)
            logger.debug("Cache enabled for stage %r", self.name)
        else:
            logger.debug("No cache_manager on context — cache disabled")

        progress = ProtocolsRegistry.get_instance(IProgressReporter)
        image_client = ProtocolsRegistry.get_instance(IImageClient)
        llm_client = ProtocolsRegistry.get_instance(ILLMClient)

        if progress:
            progress.start_stage("Scoring assets", total=len(ctx.assets))

        semaphore = asyncio.Semaphore(concurrency)
        computed = 0
        cached = 0
        failures = 0
        lock = asyncio.Lock()

        async def _analyze_one(asset: Any) -> None:
            nonlocal computed, cached, failures
            cache_key = f"{asset.id}:{prompt_hash}"

            # Check cache before acquiring semaphore — cache hits are instant
            # and should not consume a concurrency slot.
            if cache is not None:
                hit = cache.get(cache_key)
                if hit is not None:
                    logger.debug("Cache HIT for asset %s — skipping LLM call", asset.id)
                    asset.metadata.update(hit)
                    # A cache hit always represents a *successful* prior
                    # analysis (error results are never cached, see below),
                    # so clear out any stale error/vision_error left over
                    # from a previous failed run of this stage.
                    asset.metadata.pop("error", None)
                    asset.metadata.pop("vision_error", None)
                    async with lock:
                        cached += 1
                    if progress:
                        progress.advance()
                    return

            async with semaphore:
                try:
                    logger.debug("Fetching thumbnail for asset %s", asset.id)
                    if image_client is None:
                        raise RuntimeError("image_client is required for stage 'analyze.score'")
                    thumbnail = await image_client.get_asset_thumbnail(asset.id)
                    logger.debug(
                        "Thumbnail fetched for asset %s: %d bytes", asset.id, len(thumbnail)
                    )

                    schema = {"score": "float", "is_screenshot": "bool"}

                    logger.debug("Sending asset %s to LLM for analysis", asset.id)
                    if llm_client is None:
                        raise RuntimeError("llm_client is required for stage 'analyze.score'")
                    result = await llm_client.analyze_image(
                        thumbnail, prompt, schema
                    )
                    logger.debug("LLM result for asset %s: %s", asset.id, result)

                    if not isinstance(result, dict):
                        # A client that doesn't honor the documented
                        # dict-or-{"error": ...} contract (e.g. returns a
                        # list or None) would otherwise raise AttributeError
                        # from result.get() below, escape the try/except
                        # entirely (none of the caught exception types match
                        # AttributeError), and abort the whole
                        # asyncio.gather() for every other in-flight asset.
                        raise TypeError(
                            f"LLM client returned {type(result).__name__}, expected dict"
                        )

                    # Backend error: result contains an "error" key. This
                    # counts as a failure (not "computed"), is not cached,
                    # and is tagged with vision_error so filter.sensitive's
                    # fail_closed option can distinguish "vision analysis
                    # failed" from unrelated errors set by other stages.
                    if "error" in result:
                        logger.debug(
                            "LLM returned error for asset %s: %s", asset.id, result["error"]
                        )
                        asset.metadata["score"] = 0.0
                        asset.metadata["error"] = result["error"]
                        asset.metadata["vision_error"] = result["error"]
                        async with lock:
                            failures += 1
                    else:
                        raw_score = result.get("score", 0.0)
                        try:
                            score = float(raw_score)
                        except (TypeError, ValueError) as exc:
                            raise ValueError(
                                f"LLM returned a non-numeric score: {raw_score!r}"
                            ) from exc
                        if not math.isfinite(score):
                            raise ValueError(f"LLM returned a non-finite score: {raw_score!r}")
                        # Scores are documented/consumed downstream (e.g.
                        # min_score's threshold, select.best's weighting) as
                        # being in [0.0, 1.0]. A model answering on a
                        # different scale (0-10, 0-100) would otherwise
                        # silently break every stage that assumes [0, 1]
                        # rather than surfacing as an error.
                        score = max(0.0, min(1.0, score))
                        is_screenshot = bool(result.get("is_screenshot", False))

                        asset.metadata["score"] = score
                        asset.metadata["is_screenshot"] = is_screenshot
                        # Clear any previous error on successful re-analysis
                        asset.metadata.pop("error", None)
                        asset.metadata.pop("vision_error", None)

                        logger.debug(
                            "Scored asset %s: score=%.3f, is_screenshot=%s",
                            asset.id, score, is_screenshot,
                        )

                        # Store in cache
                        if cache is not None:
                            cache.put(cache_key, {
                                "score": score,
                                "is_screenshot": is_screenshot,
                            })
                            logger.debug("Cached result for asset %s", asset.id)

                        async with lock:
                            computed += 1

                except Exception as exc:
                    # Catch every exception, not just a curated subset.
                    # The previous version only caught (ConnectionError,
                    # OSError, TimeoutError, ValueError, TypeError, KeyError,
                    # RuntimeError). Anything else — an AttributeError from
                    # a malformed client response, a client-specific
                    # exception that isn't an OSError subclass (e.g. some
                    # httpx/aiohttp errors), asyncio.CancelledError from an
                    # internal timeout, etc. — would propagate out of this
                    # task, fail the whole asyncio.gather() below, and lose
                    # every other asset's result along with it, including
                    # skipping progress.finish_stage() and the stats update.
                    asset.metadata["error"] = str(exc)
                    asset.metadata["vision_error"] = str(exc)
                    logger.warning(
                        "analyze.score: error for asset %s: %s",
                        asset.id,
                        exc,
                    )
                    async with lock:
                        failures += 1

            if progress:
                progress.advance()

        tasks = [_analyze_one(asset) for asset in ctx.assets]
        await asyncio.gather(*tasks)

        if progress:
            progress.finish_stage()

        ctx.stats["analyze.score.total_assets"] = len(ctx.assets)
        ctx.stats["analyze.score.computed"] = computed
        ctx.stats["analyze.score.cached"] = cached
        ctx.stats["analyze.score.failures"] = failures

        logger.debug(
            "analyze.score complete: total=%d, computed=%d, cached=%d, failures=%d",
            len(ctx.assets), computed, cached, failures,
        )

        return [ctx]
