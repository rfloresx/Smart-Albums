"""GooglePlacesGeoClient — reverse geocoding via Google Maps Places API.

Provides a concrete implementation of the IGeoClient protocol using the
googlemaps SDK to resolve GPS coordinates into ranked lists of nearby
place candidates.

Reference implementation: calendar_maker/lib/gui/geoutil.py

The GOOGLE_API_KEY environment variable (or explicit api_key config) must
be set for the googlemaps client to authenticate requests.

This module requires the optional ``geo`` dependency group:
    pip install immich-smart-albums[geo]
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
import time
from typing import Any

from smart_albums.core.protocols import (
    IGeoClient,
    IHealthCheck,
    PlaceCandidate,
    ProtocolsRegistry,
)

logger = logging.getLogger(__name__)


def _check_dependencies() -> None:
    """Verify that the optional googlemaps package is installed."""
    try:
        import googlemaps  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "GooglePlacesGeoClient requires the 'googlemaps' package. "
            "Install it with: pip install immich-smart-albums[geo]"
        ) from exc


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great-circle distance between two points using Haversine.

    Returns distance in meters.
    """
    R = 6_371_000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# Pattern to parse plus_code compound_code: "CODE+CODE City, STATE, COUNTRY"
_COMPOUND_CODE_PATTERN = re.compile(
    r"[23456789CFGHJMPQRVWX+]+\s+(.+?),\s*([A-Z]{2,}),\s*([A-Z]{2,})"
)


@ProtocolsRegistry.register("google_places", IGeoClient)
@ProtocolsRegistry.register("google_places", IHealthCheck)
class GooglePlacesGeoClient:
    """Reverse geocoding via Google Maps Places API.

    Wraps googlemaps.Client.places_nearby() with pagination support,
    parses plus_code compound_code for city/state/country extraction.
    """

    def __init__(
        self,
        api_key: str,
        default_radius: int = 1000,
        max_results: int = 10,
    ) -> None:
        _check_dependencies()
        import googlemaps

        self._client = googlemaps.Client(key=api_key)
        self._api_key = api_key
        self._default_radius = default_radius
        self._max_results = max_results

    @property
    def health_check_url(self) -> str:
        """Display URL for health check status."""
        return "https://maps.googleapis.com/maps/api/place/nearbysearch"

    async def health_check(self) -> bool:
        """Check connectivity by performing a minimal places_nearby request."""
        try:
            # Use a known location (0,0) with tiny radius — should return quickly
            await asyncio.to_thread(
                self._client.places_nearby,
                location=(0.0, 0.0),
                radius=1,
            )
            return True
        except Exception:
            return False

    async def reverse_geocode(
        self,
        latitude: float,
        longitude: float,
        radius_meters: int | None = None,
        max_results: int | None = None,
    ) -> list[PlaceCandidate]:
        """Resolve GPS coordinates into ranked nearby place candidates.

        Args:
            latitude: Latitude in degrees.
            longitude: Longitude in degrees.
            radius_meters: Search radius (default from constructor).
            max_results: Maximum candidates to return (default from constructor).

        Returns:
            List of PlaceCandidate sorted by distance from the query point.
        """
        radius = radius_meters if radius_meters is not None else self._default_radius
        limit = max_results if max_results is not None else self._max_results

        raw_results = await asyncio.to_thread(
            self._fetch_nearby, latitude, longitude, radius, limit
        )

        candidates: list[PlaceCandidate] = []
        for place in raw_results:
            loc = place.get("geometry", {}).get("location", {})
            plat = float(loc.get("lat", 0.0))
            plng = float(loc.get("lng", 0.0))
            distance = _haversine_meters(latitude, longitude, plat, plng)

            name = place.get("name", "")
            city, state, country = self._parse_location(place)

            candidates.append(
                PlaceCandidate(
                    name=name,
                    city=city,
                    state=state,
                    country=country,
                    latitude=plat,
                    longitude=plng,
                    distance_meters=round(distance, 1),
                    place_type=",".join(place.get("types", [])),
                    raw=place,
                )
            )

        candidates.sort(key=lambda c: c.distance_meters)
        return candidates[:limit]

    def _fetch_nearby(
        self, lat: float, lng: float, radius: int, max_results: int
    ) -> list[dict[str, Any]]:
        """Synchronous Places API call with pagination.

        Mirrors the calendar_maker geoutil._get_nearby_places pattern.
        """
        results: list[dict[str, Any]] = []
        response: dict[str, Any] = self._client.places_nearby(
            location=(lat, lng), radius=radius, open_now=False
        )
        results.extend(response.get("results", []))

        # Follow next_page_token pagination
        while "next_page_token" in response and len(results) < max_results:
            time.sleep(2)  # Token needs a short delay to become valid
            response = self._client.places_nearby(
                page_token=response["next_page_token"]
            )
            results.extend(response.get("results", []))

        return results[:max_results]

    def _parse_location(self, place: dict[str, Any]) -> tuple[str, str, str]:
        """Extract city/state/country from plus_code compound_code or vicinity.

        Returns:
            Tuple of (city, state, country). Empty strings for unavailable fields.
        """
        plus_code = place.get("plus_code", {})
        compound = plus_code.get("compound_code", "")

        if compound:
            match = _COMPOUND_CODE_PATTERN.match(compound)
            if match:
                return (
                    match.group(1).strip(),
                    match.group(2).strip(),
                    match.group(3).strip(),
                )

        # Fallback: parse vicinity string
        vicinity = place.get("vicinity", "")
        if vicinity:
            parts = [p.strip() for p in vicinity.split(",")]
            city = parts[-1] if parts else ""
            return city, "", ""

        return "", "", ""
