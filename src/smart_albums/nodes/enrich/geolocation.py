"""enrich.geolocation — resolve GPS coordinates to place name candidates.

Populates ``asset.metadata["geo"]`` with a ranked list of nearby place
candidates, enabling human review and selection of the best match.

Uses the ``IGeoClient`` protocol for the actual reverse geocoding and the
``ICacheManager`` for persistent caching to limit API usage. Cache keys are
based on rounded coordinates so nearby photos share a single API call.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Any

from smart_albums.core.context import ContextBatch, PipelineContext
from smart_albums.core.models import Asset
from smart_albums.core.node import ConfigParam, Stage
from smart_albums.core.models import PlaceCandidate
from smart_albums.core.protocol_registry import ProtocolsRegistry
from smart_albums.core.protocols import (
    ICacheManager,
    IGeoClient,
)
from smart_albums.core.registry import stage

logger = logging.getLogger(__name__)


def _cache_key(lat: float, lng: float, radius: int, precision: int) -> str:
    """Generate a stable cache key by rounding coordinates.

    precision=4 → ~11m resolution (venue-level)
    precision=3 → ~111m resolution (neighborhood-level, default)
    precision=2 → ~1.1km resolution (city-level)
    """
    rlat = round(lat, precision)
    rlng = round(lng, precision)
    return f"{rlat}:{rlng}:{radius}"


def _build_label(candidate: PlaceCandidate) -> str:
    """Build a human-readable display label from a place candidate."""
    parts: list[str] = []
    if candidate.name:
        parts.append(candidate.name)
    if candidate.city:
        parts.append(candidate.city)
    elif candidate.state:
        parts.append(candidate.state)
    if candidate.country and not candidate.city:
        parts.append(candidate.country)
    return ", ".join(parts) if parts else ""


def _serialize_candidate(candidate: PlaceCandidate) -> dict[str, Any]:
    """Serialize a PlaceCandidate to a JSON-compatible dict, excluding raw."""
    data = asdict(candidate)
    # Remove raw API response to keep cache/metadata lightweight
    data.pop("raw", None)
    return data


@stage("enrich.geolocation")
class EnrichGeolocation(Stage):
    """Resolve GPS coordinates to place names with candidate ranking.

    For each asset with latitude/longitude, queries the configured IGeoClient
    to retrieve nearby place candidates. Results are cached by rounded
    coordinates to minimize API usage.

    Writes to ``asset.metadata["geo"]``:
        - candidates: list of place dicts (name, city, state, country, distance)
        - selected_index: default selection (0 = nearest)
        - label: pre-built display label from the top candidate
        - coordinates: original GPS coords
    """

    _config_schema = (
        ConfigParam(
            key="radius_meters",
            type=int,
            default=1000,
            min=50,
            max=50000,
            description="Search radius in meters for the nearby places API.",
        ),
        ConfigParam(
            key="max_candidates",
            type=int,
            default=5,
            min=1,
            max=20,
            description="Maximum place candidates to store per asset location.",
        ),
        ConfigParam(
            key="cache_precision",
            type=int,
            default=3,
            min=2,
            max=5,
            description=(
                "Coordinate rounding digits for cache keys. "
                "3 ≈ 111m (recommended), 4 ≈ 11m, 2 ≈ 1.1km."
            ),
        ),
        ConfigParam(
            key="concurrency",
            type=int,
            default=2,
            min=1,
            max=10,
            description="Max parallel geocoding API calls (keep low for rate limits).",
        ),
        ConfigParam(
            key="skip_if_exists",
            type=bool,
            default=True,
            description="Skip geocoding if asset.metadata['geo'] is already populated.",
        ),
        ConfigParam(
            key="place_type",
            type=str,
            default=None,
            description="Optional Google Places type filter (e.g. 'point_of_interest').",
        ),
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        geo_client = ProtocolsRegistry.get_instance(IGeoClient)
        if geo_client is None:
            logger.warning(
                "No geo_client configured — skipping geolocation enrichment. "
                "Register an IGeoClient provider (e.g. 'google_places') to enable."
            )
            return [ctx]

        if not ctx.assets:
            logger.debug("No assets in context — nothing to geocode")
            return [ctx]

        cache_manager = ProtocolsRegistry.get_instance(ICacheManager)
        cache = cache_manager.get_cache("geo_places") if cache_manager else None

        radius: int = self.get("radius_meters")
        max_candidates: int = self.get("max_candidates")
        precision: int = self.get("cache_precision")
        skip_existing: bool = self.get("skip_if_exists")
        concurrency: int = self.get("concurrency")

        if cache:
            logger.debug("Geo cache enabled (precision=%d)", precision)
        else:
            logger.debug("Geo cache disabled — all lookups will hit the API")

        resolved = 0
        cache_hits = 0
        skipped = 0
        api_calls = 0

        sem = asyncio.Semaphore(concurrency)

        async def process_asset(asset: Asset) -> None:
            nonlocal resolved, cache_hits, skipped, api_calls

            # Skip assets without GPS coordinates
            if asset.latitude is None or asset.longitude is None:
                skipped += 1
                return

            # Skip if already enriched
            if skip_existing and "geo" in asset.metadata:
                skipped += 1
                return

            key = _cache_key(asset.latitude, asset.longitude, radius, precision)

            # Check cache first
            if cache and key in cache:
                cached_data = cache.get(key)
                candidates = [
                    PlaceCandidate(**entry) for entry in cached_data
                ]
                cache_hits += 1
            else:
                # Hit the API with concurrency control
                async with sem:
                    candidates = await geo_client.reverse_geocode(
                        latitude=asset.latitude,
                        longitude=asset.longitude,
                        radius_meters=radius,
                        max_results=max_candidates,
                    )
                api_calls += 1

                # Persist to cache
                if cache:
                    cache.put(
                        key,
                        [_serialize_candidate(c) for c in candidates],
                    )

            # Trim to configured max
            candidates = candidates[:max_candidates]

            # Write geo metadata to asset
            asset.metadata["geo"] = {
                "candidates": [_serialize_candidate(c) for c in candidates],
                "selected_index": 0,
                "label": _build_label(candidates[0]) if candidates else "",
                "coordinates": {
                    "lat": asset.latitude,
                    "lng": asset.longitude,
                },
            }
            resolved += 1

        # Process assets sequentially for better cache locality.
        # Photos are typically time-sorted, so nearby photos are adjacent
        # and benefit from cache hits after the first API call in an area.
        for asset in ctx.assets:
            await process_asset(asset)

        ctx.stats["enrich.geolocation.resolved"] = resolved
        ctx.stats["enrich.geolocation.cache_hits"] = cache_hits
        ctx.stats["enrich.geolocation.skipped"] = skipped
        ctx.stats["enrich.geolocation.api_calls"] = api_calls

        logger.info(
            "enrich.geolocation: resolved=%d, cache_hits=%d, skipped=%d, api_calls=%d",
            resolved,
            cache_hits,
            skipped,
            api_calls,
        )

        return [ctx]
