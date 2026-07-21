"""Protocol interfaces for image clients, LLM backends, and progress reporting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Protocol, runtime_checkable

from smart_albums.core.models import Asset


@dataclass(frozen=True)
class AlbumSummary:
    """Summary of an existing album."""

    id: str
    name: str


@dataclass(frozen=True)
class AlbumResult:
    """Result of creating or updating an album."""

    id: str
    name: str
    url: str


@runtime_checkable
class IImageClient(Protocol):
    """Abstraction over any photo library provider."""

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


@runtime_checkable
class ILLMClient(Protocol):
    """Abstraction over vision/LLM backends."""

    async def analyze_image(self, image_bytes: bytes, prompt: str, return_schema: dict[str, Any]) -> dict[str, Any]: ...

    async def embed(self, image_bytes: bytes) -> list[float]: ...


@runtime_checkable
class IEmbeddingClient(Protocol):
    """Abstraction over embedding-only backends (e.g. Ollama, HuggingFace).

    Unlike ILLMClient which combines vision analysis with embedding,
    this protocol is purpose-specific for computing vector embeddings
    from raw image bytes. The caller is responsible for lifecycle
    management via the optional close() method.

    Batch API
    ---------
    Clients that can process multiple images in a single forward pass
    (e.g. GPU-backed HuggingFace models) should also implement
    ``embed_batch()``. The node layer detects support via
    ``hasattr(client, 'embed_batch')`` and prefers the batch path when
    available, falling back to sequential ``embed()`` calls otherwise.
    """
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
class ICache(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...

    def put(self, key: str, value: Any) -> None: ...

    def __contains__(self, key: object) -> bool: ...

    def __len__(self) -> int: ...

@runtime_checkable
class ICacheManager(Protocol):
    def get_cache(self, cache_name: str) -> ICache | None: ...


class ProtocolsRegistry:
    _registry: dict[type, dict[str, type]] = {}

    @staticmethod
    def register(name: str, protocol: type) -> Any:
        def decorator(cls: type) -> type:
            if protocol not in ProtocolsRegistry._registry:
                ProtocolsRegistry._registry[protocol] = {}
            ProtocolsRegistry._registry[protocol][name] = cls
            return cls
        return decorator

    @staticmethod
    def get_registry() -> dict[type, dict[str, type]]:
        return ProtocolsRegistry._registry

    @staticmethod
    def get_protocols() -> list[type]:
        return list(ProtocolsRegistry._registry.keys())

    @staticmethod
    def get_providers(protocol: type) -> dict[str, type]:
        return ProtocolsRegistry._registry.get(protocol, {})

    @staticmethod
    def create(protocol: type, name: str, config: dict[str, Any]) -> Any:
        if protocol not in ProtocolsRegistry._registry:
            raise ValueError(f"No clients registered for protocol {protocol}")
        if name not in ProtocolsRegistry._registry[protocol]:
            raise ValueError(f"No client named {name} registered for protocol {protocol}")
        return ProtocolsRegistry._registry[protocol][name](**config)

@ProtocolsRegistry.register("disabled", ICacheManager)
class NoCacheManager:
    """No-op cache manager that always returns None from the factory.

    Registered as the "disabled" provider so that pipeline execution can
    proceed without caching when no cache_dir is configured.
    """

    def get_cache(self, cache_name: str) -> None:
        """Return None — signals to stages that caching is disabled."""
        return None
