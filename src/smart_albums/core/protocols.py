"""Protocol interfaces for image clients, LLM backends, and progress reporting."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Protocol, runtime_checkable

from smart_albums.core.models import AlbumResult, AlbumSummary, Asset, PlaceCandidate
from smart_albums.core.protocol_registry import ProtocolsRegistry

__all__ = [
    "AlbumResult",
    "AlbumSummary",
    "PlaceCandidate",
    "ProtocolsRegistry",
    "ICache",
    "ICacheManager",
    "IEmbeddingClient",
    "IGeoClient",
    "IHealthCheck",
    "IImageClient",
    "ILLMClient",
    "IProgressReporter",
]


@runtime_checkable
class IImageClient(Protocol):
    """Image Library:Photo library provider — the source of assets and the target for album publishing."""

    async def search_assets(
        self,
        taken_after: datetime,
        taken_before: datetime,
        **kwargs: Any,
    ) -> list[Asset]: ...

    async def get_asset_thumbnail(self, asset_id: str) -> bytes: ...

    async def list_albums(self) -> list[AlbumSummary]: ...

    async def create_album(self, name: str, asset_ids: list[str]) -> AlbumResult: ...

    async def search_smart(self, query: str, limit: int) -> list[Asset]: ...

    async def search_people_any(self) -> list[Asset]: ...

    async def search_on_this_day(self, run_date: date) -> list[Asset]: ...

    async def get_album_by_name(self, name: str) -> AlbumSummary | None: ...

    async def list_album_assets(self, album_id: str) -> list[Asset]: ...

    async def add_assets_to_album(self, album_id: str, asset_ids: list[str]) -> None: ...

    async def remove_assets_from_album(self, album_id: str, asset_ids: list[str]) -> None: ...

    async def get_asset_full(self, asset_id: str) -> bytes: ...


@runtime_checkable
class ILLMClient(Protocol):
    """Vision / LLM:Vision model backend used for aesthetic quality scoring."""

    async def analyze_image(self, image_bytes: bytes, prompt: str, return_schema: dict[str, Any]) -> dict[str, Any]: ...


@runtime_checkable
class IEmbeddingClient(Protocol):
    """Embedding:Backend that computes image embeddings for scene clustering and near-duplicate detection."""

    @property
    def embed_model(self) -> str: ...

    async def embed(self, image_bytes: bytes) -> list[float]: ...

    async def embed_batch(
        self,
        image_bytes_list: list[bytes],
    ) -> list[list[float] | None]: ...

    async def close(self) -> None: ...


@runtime_checkable
class IProgressReporter(Protocol):
    """Lifecycle hooks for stage progress reporting."""

    def start_stage(self, description: str, total: int | None = None) -> None: ...

    def advance(self, n: int = 1) -> None: ...

    def finish_stage(self) -> None: ...

    def log(self, message: str) -> None: ...

    def warn(self, message: str) -> None: ...


@runtime_checkable
class IHealthCheck(Protocol):
    """Protocol for clients that can report their server connectivity status.

    Clients implementing this protocol expose a lightweight health check
    that verifies the backing service is reachable without performing any
    heavy computation. Used by the webgui to display dynamic service status.
    """

    async def health_check(self) -> bool:
        """Return True if the backing service is reachable and healthy.

        Should complete quickly (timeout ~5-10s). Must not raise — returns
        False on any failure.
        """
        ...

    @property
    def health_check_url(self) -> str:
        """The base URL being checked, for display purposes."""
        ...


@runtime_checkable
class IGeoClient(Protocol):
    """Geolocation:Reverse geocoding provider that resolves GPS coordinates into place name candidates."""

    async def reverse_geocode(
        self,
        latitude: float,
        longitude: float,
        radius_meters: int = 1000,
        max_results: int = 10,
    ) -> list[PlaceCandidate]: ...


@runtime_checkable
class ICache(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...

    def put(self, key: str, value: Any) -> None: ...

    def __contains__(self, key: object) -> bool: ...

    def __len__(self) -> int: ...


@runtime_checkable
class ICacheManager(Protocol):
    """Cache:Local cache for scores and embeddings so re-runs skip already-computed assets."""

    def get_cache(self, cache_name: str) -> ICache | None: ...
