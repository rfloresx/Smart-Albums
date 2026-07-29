"""Shared fixtures and helpers for smart_albums tests."""

from __future__ import annotations

from datetime import datetime, date
from typing import Any

import pytest

from smart_albums.core.models import Asset
from smart_albums.core.context import PipelineContext
from smart_albums.core.protocols import (
    AlbumResult,
    AlbumSummary,
    ICacheManager,
    IEmbeddingClient,
    IImageClient,
    ILLMClient,
    IProgressReporter,
    ProtocolsRegistry,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_protocol_instances():
    """Ensure ProtocolsRegistry instances are cleared between every test."""
    ProtocolsRegistry.clear_instances()
    yield
    ProtocolsRegistry.clear_instances()


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def make_asset(
    id: str = "asset-1",
    filename: str = "IMG_001.jpg",
    captured_at: datetime | None = None,
    mime_type: str = "image/jpeg",
    metadata: dict[str, Any] | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> Asset:
    """Create an Asset with sensible defaults for testing."""
    return Asset(
        id=id,
        filename=filename,
        captured_at=captured_at or datetime(2024, 6, 15, 12, 0, 0),
        mime_type=mime_type,
        metadata=metadata if metadata is not None else {},
        latitude=latitude,
        longitude=longitude,
    )


def make_scored_asset(
    id: str = "asset-1",
    score: float = 0.5,
    captured_at: datetime | None = None,
    **kwargs: Any,
) -> Asset:
    """Create an Asset with a score in metadata."""
    metadata = kwargs.pop("metadata", {})
    metadata["score"] = score
    return make_asset(id=id, captured_at=captured_at, metadata=metadata, **kwargs)


def make_context(
    assets: list[Asset] | None = None,
    config: dict[str, Any] | None = None,
    image_client: IImageClient | None = None,
    llm_client: ILLMClient | None = None,
    embedding_client: IEmbeddingClient | None = None,
    cache_manager: Any = None,
    progress: IProgressReporter | None = None,
    stats: dict[str, Any] | None = None,
) -> PipelineContext:
    """Create a PipelineContext with sensible defaults for testing.

    Also registers provided protocol instances in ProtocolsRegistry so that
    nodes using the registry-based lookup can find them.
    """
    if image_client is not None:
        ProtocolsRegistry.set_instance(IImageClient, image_client)
    if llm_client is not None:
        ProtocolsRegistry.set_instance(ILLMClient, llm_client)
    if embedding_client is not None:
        ProtocolsRegistry.set_instance(IEmbeddingClient, embedding_client)
    if cache_manager is not None:
        ProtocolsRegistry.set_instance(ICacheManager, cache_manager)
    if progress is not None:
        ProtocolsRegistry.set_instance(IProgressReporter, progress)

    return PipelineContext(
        config=config or {},
        assets=assets or [],
        stats=stats or {},
    )


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class FakeImageClient:
    """Fake image client for testing stages that need IImageClient."""

    def __init__(
        self,
        assets: list[Asset] | None = None,
        thumbnail: bytes = b"\x00" * 100,
        albums: list[AlbumSummary] | None = None,
    ) -> None:
        self._assets = assets or []
        self._thumbnail = thumbnail
        self._albums = albums or []
        self._created_albums: list[tuple[str, list[str]]] = []
        self._album_assets: dict[str, list[Asset]] = {}

    async def search_assets(self, taken_after: datetime, taken_before: datetime, **kwargs: Any) -> list[Asset]:
        return [a for a in self._assets if taken_after <= a.captured_at <= taken_before]

    async def get_asset_thumbnail(self, asset_id: str) -> bytes:
        return self._thumbnail

    async def list_albums(self) -> list[AlbumSummary]:
        return self._albums

    async def create_album(self, name: str, asset_ids: list[str]) -> AlbumResult:
        self._created_albums.append((name, asset_ids))
        return AlbumResult(id="album-1", name=name, url=f"http://test/albums/album-1")

    async def search_smart(self, query: str, limit: int) -> list[Asset]:
        return self._assets[:limit]

    async def search_people_any(self) -> list[Asset]:
        return self._assets

    async def search_on_this_day(self, run_date: date) -> list[Asset]:
        return [a for a in self._assets if a.captured_at.month == run_date.month and a.captured_at.day == run_date.day]

    async def get_album_by_name(self, name: str) -> AlbumSummary | None:
        for album in self._albums:
            if album.name == name:
                return album
        return None

    async def list_album_assets(self, album_id: str) -> list[Asset]:
        return self._album_assets.get(album_id, [])

    async def add_assets_to_album(self, album_id: str, asset_ids: list[str]) -> None:
        pass

    async def remove_assets_from_album(self, album_id: str, asset_ids: list[str]) -> None:
        pass


class FakeLLMClient:
    """Fake LLM client for testing analyze stages."""

    def __init__(self, response: dict[str, Any] | None = None) -> None:
        self._response = response or {"score": 0.8, "is_screenshot": False}

    async def analyze_image(self, image_bytes: bytes, prompt: str, return_schema: dict[str, Any]) -> dict[str, Any]:
        return self._response

    async def embed(self, image_bytes: bytes) -> list[float]:
        return [0.1] * 128


class FakeEmbeddingClient:
    """Fake embedding client for testing embedding stages."""

    def __init__(self, embedding: list[float] | None = None) -> None:
        self._embedding = embedding or [0.1] * 128

    @property
    def embed_model(self) -> str:
        return "fake-model"

    async def embed(self, image_bytes: bytes) -> list[float]:
        return self._embedding

    async def embed_batch(self, image_bytes_list: list[bytes]) -> list[list[float] | None]:
        return [self._embedding for _ in image_bytes_list]

    async def close(self) -> None:
        pass


class FakeProgress:
    """Fake progress reporter that records calls."""

    def __init__(self) -> None:
        self.stages: list[str] = []
        self.advances: int = 0
        self.logs: list[str] = []
        self.warnings: list[str] = []

    def start_stage(self, description: str, total: int | None = None) -> None:
        self.stages.append(description)

    def advance(self, n: int = 1) -> None:
        self.advances += n

    def finish_stage(self) -> None:
        pass

    def log(self, message: str) -> None:
        self.logs.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)
