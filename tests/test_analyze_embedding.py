"""Tests for smart_albums.nodes.analyze.embedding."""

from __future__ import annotations

import pytest

from smart_albums.nodes.analyze.embedding import AnalyzeEmbedding

from conftest import make_asset, make_context, FakeImageClient, FakeEmbeddingClient, FakeProgress


class _FakeCache:
    """In-memory cache for testing."""

    def __init__(self, data: dict | None = None) -> None:
        self._data = data or {}

    def get(self, key, default=None):
        return self._data.get(str(key), default)

    def put(self, key, value):
        self._data[str(key)] = value


class _FakeCacheManager:
    """Cache manager that returns an in-memory cache."""

    def __init__(self, data: dict | None = None) -> None:
        self._cache = _FakeCache(data)

    def get_cache(self, name: str):
        return self._cache


class _BatchEmbeddingClient:
    """Embedding client that supports embed_batch()."""

    def __init__(self, embedding: list[float] | None = None) -> None:
        self._embedding = embedding or [0.5] * 64
        self.embed_calls = 0
        self.batch_calls = 0

    @property
    def embed_model(self) -> str:
        return "test-model"

    async def embed(self, image_bytes: bytes) -> list[float]:
        self.embed_calls += 1
        return self._embedding

    async def embed_batch(self, image_bytes_list: list[bytes]) -> list[list[float] | None]:
        self.batch_calls += 1
        return [self._embedding for _ in image_bytes_list]

    async def close(self) -> None:
        pass


class _SequentialEmbeddingClient:
    """Embedding client that implements embed_batch() via sequential calls."""

    def __init__(self, embedding: list[float] | None = None) -> None:
        self._embedding = embedding or [0.3] * 64
        self.embed_calls = 0
        self.batch_calls = 0

    @property
    def embed_model(self) -> str:
        return "seq-model"

    async def embed(self, image_bytes: bytes) -> list[float]:
        self.embed_calls += 1
        return self._embedding

    async def embed_batch(self, image_bytes_list: list[bytes]) -> list[list[float] | None]:
        self.batch_calls += 1
        results = []
        for img in image_bytes_list:
            results.append(await self.embed(img))
        return results

    async def close(self) -> None:
        pass


class TestAnalyzeEmbedding:
    """Tests for the AnalyzeEmbedding stage."""

    @pytest.mark.asyncio
    async def test_no_embedding_client_skips(self):
        assets = [make_asset(id="a1")]
        ctx = make_context(assets=assets)
        node = AnalyzeEmbedding({})
        result = await node.run(ctx)
        assert "embedding" not in result[0].assets[0].metadata

    @pytest.mark.asyncio
    async def test_empty_assets_zero_stats(self):
        ctx = make_context(assets=[], embedding_client=_BatchEmbeddingClient())
        node = AnalyzeEmbedding({})
        result = await node.run(ctx)
        assert result[0].stats["analyze.embedding.total_assets"] == 0
        assert result[0].stats["analyze.embedding.computed"] == 0

    @pytest.mark.asyncio
    async def test_batch_path_used_when_available(self):
        client = FakeImageClient()
        embed_client = _BatchEmbeddingClient()
        assets = [make_asset(id="a1"), make_asset(id="a2")]
        ctx = make_context(assets=assets, image_client=client, embedding_client=embed_client)
        node = AnalyzeEmbedding({})
        result = await node.run(ctx)
        # batch path used
        assert embed_client.batch_calls >= 1
        assert embed_client.embed_calls == 0
        # embeddings set
        assert result[0].assets[0].metadata["embedding"] == [0.5] * 64
        assert result[0].assets[1].metadata["embedding"] == [0.5] * 64
        assert result[0].stats["analyze.embedding.computed"] == 2

    @pytest.mark.asyncio
    async def test_sequential_client_uses_batch_path(self):
        client = FakeImageClient()
        embed_client = _SequentialEmbeddingClient()
        assets = [make_asset(id="a1"), make_asset(id="a2")]
        ctx = make_context(assets=assets, image_client=client, embedding_client=embed_client)
        node = AnalyzeEmbedding({})
        result = await node.run(ctx)
        # batch path used (embed_batch delegates to embed internally)
        assert embed_client.batch_calls >= 1
        assert result[0].assets[0].metadata["embedding"] == [0.3] * 64
        assert result[0].stats["analyze.embedding.computed"] == 2

    @pytest.mark.asyncio
    async def test_cache_hit_skips_computation(self):
        embed_client = _BatchEmbeddingClient()
        cached_embedding = [0.9] * 64
        cache_key = str(("a1", "test-model"))
        cache_manager = _FakeCacheManager({cache_key: cached_embedding})

        client = FakeImageClient()
        assets = [make_asset(id="a1")]
        ctx = make_context(assets=assets, image_client=client, embedding_client=embed_client, cache_manager=cache_manager)
        node = AnalyzeEmbedding({})
        result = await node.run(ctx)
        assert result[0].assets[0].metadata["embedding"] == cached_embedding
        assert result[0].stats["analyze.embedding.cached"] == 1
        assert result[0].stats["analyze.embedding.computed"] == 0

    @pytest.mark.asyncio
    async def test_thumbnail_failure_records_error(self):
        class FailingImageClient(FakeImageClient):
            async def get_asset_thumbnail(self, asset_id: str) -> bytes:
                raise ConnectionError("timeout")

        embed_client = _SequentialEmbeddingClient()
        client = FailingImageClient()
        assets = [make_asset(id="a1")]
        ctx = make_context(assets=assets, image_client=client, embedding_client=embed_client)
        node = AnalyzeEmbedding({})
        result = await node.run(ctx)
        assert "error" in result[0].assets[0].metadata
        assert result[0].stats["analyze.embedding.failures"] == 1
        assert result[0].stats["analyze.embedding.computed"] == 0

    @pytest.mark.asyncio
    async def test_file_not_found_records_error(self):
        class NotFoundImageClient(FakeImageClient):
            async def get_asset_thumbnail(self, asset_id: str) -> bytes:
                raise FileNotFoundError("media gone")

        embed_client = _SequentialEmbeddingClient()
        client = NotFoundImageClient()
        assets = [make_asset(id="a1")]
        ctx = make_context(assets=assets, image_client=client, embedding_client=embed_client)
        node = AnalyzeEmbedding({})
        result = await node.run(ctx)
        assert "error" in result[0].assets[0].metadata
        assert result[0].stats["analyze.embedding.failures"] == 1

    @pytest.mark.asyncio
    async def test_mixed_cached_and_uncached(self):
        embed_client = _BatchEmbeddingClient(embedding=[0.5] * 64)
        cached_embedding = [0.9] * 64
        cache_key = str(("a1", "test-model"))
        cache_manager = _FakeCacheManager({cache_key: cached_embedding})

        client = FakeImageClient()
        assets = [make_asset(id="a1"), make_asset(id="a2")]
        ctx = make_context(assets=assets, image_client=client, embedding_client=embed_client, cache_manager=cache_manager)
        node = AnalyzeEmbedding({})
        result = await node.run(ctx)
        # a1 from cache, a2 computed
        assert result[0].assets[0].metadata["embedding"] == cached_embedding
        assert result[0].assets[1].metadata["embedding"] == [0.5] * 64
        assert result[0].stats["analyze.embedding.cached"] == 1
        assert result[0].stats["analyze.embedding.computed"] == 1
