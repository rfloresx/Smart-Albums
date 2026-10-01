"""Tests for smart_albums.nodes.analyze.score."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from smart_albums.nodes.analyze.score import AnalyzeScore

from conftest import make_asset, make_context, FakeImageClient, FakeLLMClient, FakeProgress


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


class TestAnalyzeScore:
    """Tests for the AnalyzeScore stage."""

    @pytest.fixture
    def prompt_file(self, tmp_path: Path) -> str:
        """Create a temporary prompt file."""
        p = tmp_path / "prompt.md"
        p.write_text("Rate this photo from 0 to 1.")
        return str(p)

    @pytest.mark.asyncio
    async def test_no_prompt_file_raises(self):
        ctx = make_context(assets=[make_asset()])
        node = AnalyzeScore({"prompt_file": None})
        with pytest.raises(ValueError, match="No prompt provided"):
            await node.run(ctx)

    @pytest.mark.asyncio
    async def test_missing_prompt_file_raises(self):
        ctx = make_context(assets=[make_asset()])
        node = AnalyzeScore({"prompt_file": "nonexistent/prompt.md"})
        with pytest.raises(ValueError, match="Prompt file not found"):
            await node.run(ctx)

    @pytest.mark.asyncio
    async def test_scores_single_asset(self, prompt_file: str):
        client = FakeImageClient()
        llm = FakeLLMClient(response={"score": 0.85, "is_screenshot": False})
        assets = [make_asset(id="a1")]
        ctx = make_context(assets=assets, image_client=client, llm_client=llm)
        node = AnalyzeScore({"prompt_file": prompt_file})
        result = await node.run(ctx)
        assert result[0].assets[0].metadata["score"] == 0.85
        assert result[0].assets[0].metadata["is_screenshot"] is False
        assert result[0].stats["analyze.score.computed"] == 1
        assert result[0].stats["analyze.score.cached"] == 0

    @pytest.mark.asyncio
    async def test_scores_multiple_assets(self, prompt_file: str):
        client = FakeImageClient()
        llm = FakeLLMClient(response={"score": 0.7, "is_screenshot": True})
        assets = [make_asset(id="a1"), make_asset(id="a2"), make_asset(id="a3")]
        ctx = make_context(assets=assets, image_client=client, llm_client=llm)
        node = AnalyzeScore({"prompt_file": prompt_file})
        result = await node.run(ctx)
        assert result[0].stats["analyze.score.total_assets"] == 3
        assert result[0].stats["analyze.score.computed"] == 3
        for asset in result[0].assets:
            assert asset.metadata["score"] == 0.7
            assert asset.metadata["is_screenshot"] is True

    @pytest.mark.asyncio
    async def test_llm_error_response_sets_zero_score(self, prompt_file: str):
        client = FakeImageClient()
        llm = FakeLLMClient(response={"error": "model timeout"})
        assets = [make_asset(id="a1")]
        ctx = make_context(assets=assets, image_client=client, llm_client=llm)
        node = AnalyzeScore({"prompt_file": prompt_file})
        result = await node.run(ctx)
        assert result[0].assets[0].metadata["score"] == 0.0
        assert result[0].assets[0].metadata["error"] == "model timeout"
        assert result[0].assets[0].metadata["vision_error"] == "model timeout"
        # An LLM-reported error is a failure, not a successful computation:
        # it must not be counted as "computed" (which downstream reporting
        # treats as a real score), and it must not be silently swallowed by
        # a later min_score filter without ever showing up in failure stats.
        assert result[0].stats["analyze.score.computed"] == 0
        assert result[0].stats["analyze.score.failures"] == 1

    @pytest.mark.asyncio
    async def test_score_out_of_range_is_clamped(self, prompt_file: str):
        client = FakeImageClient()
        llm = FakeLLMClient(response={"score": 8.5, "is_screenshot": False})
        assets = [make_asset(id="a1")]
        ctx = make_context(assets=assets, image_client=client, llm_client=llm)
        node = AnalyzeScore({"prompt_file": prompt_file})
        result = await node.run(ctx)
        # A model answering on a 0-10 scale must not silently break every
        # downstream stage that assumes scores are in [0, 1] (min_score's
        # threshold, select.best's weighting, ...).
        assert result[0].assets[0].metadata["score"] == 1.0
        assert result[0].stats["analyze.score.computed"] == 1

    @pytest.mark.asyncio
    async def test_non_numeric_score_is_a_failure(self, prompt_file: str):
        client = FakeImageClient()
        llm = FakeLLMClient(response={"score": "not-a-number", "is_screenshot": False})
        assets = [make_asset(id="a1")]
        ctx = make_context(assets=assets, image_client=client, llm_client=llm)
        node = AnalyzeScore({"prompt_file": prompt_file})
        result = await node.run(ctx)
        assert result[0].stats["analyze.score.failures"] == 1
        assert result[0].stats["analyze.score.computed"] == 0
        assert "error" in result[0].assets[0].metadata

    @pytest.mark.asyncio
    async def test_non_dict_llm_result_is_a_failure_not_a_crash(self, prompt_file: str):
        client = FakeImageClient()
        llm = FakeLLMClient()
        llm._response = ["unexpected", "list"]  # type: ignore[assignment]
        assets = [make_asset(id="a1"), make_asset(id="a2")]
        ctx = make_context(assets=assets, image_client=client, llm_client=llm)
        node = AnalyzeScore({"prompt_file": prompt_file})
        # A malformed (non-dict) LLM response must be recorded as a
        # per-asset failure, not propagate out of asyncio.gather() and take
        # down the whole stage (which would also lose the second asset's
        # result and skip progress.finish_stage()/the stats update).
        result = await node.run(ctx)
        assert result[0].stats["analyze.score.failures"] == 2
        assert result[0].stats["analyze.score.computed"] == 0

    @pytest.mark.asyncio
    async def test_cache_hit_clears_stale_error(self, prompt_file: str):
        import hashlib

        with open(prompt_file, "r") as f:
            prompt_content = f.read()
        prompt_hash = hashlib.sha256(prompt_content.encode()).hexdigest()[:8]
        cache_key = f"a1:{prompt_hash}"
        cache_data = {cache_key: {"score": 0.9, "is_screenshot": False}}
        cache_manager = _FakeCacheManager(cache_data)

        client = FakeImageClient()
        llm = FakeLLMClient(response={"score": 0.1, "is_screenshot": True})
        # Simulate a stale error left over from a previous failed run.
        assets = [make_asset(id="a1", metadata={"error": "old failure", "vision_error": "old failure"})]
        ctx = make_context(assets=assets, image_client=client, llm_client=llm, cache_manager=cache_manager)
        node = AnalyzeScore({"prompt_file": prompt_file})
        result = await node.run(ctx)
        assert result[0].assets[0].metadata["score"] == 0.9
        assert "error" not in result[0].assets[0].metadata
        assert "vision_error" not in result[0].assets[0].metadata

    @pytest.mark.asyncio
    async def test_thumbnail_exception_records_failure(self, prompt_file: str):
        class FailingImageClient(FakeImageClient):
            async def get_asset_thumbnail(self, asset_id: str) -> bytes:
                raise ConnectionError("network down")

        client = FailingImageClient()
        llm = FakeLLMClient()
        assets = [make_asset(id="a1")]
        ctx = make_context(assets=assets, image_client=client, llm_client=llm)
        node = AnalyzeScore({"prompt_file": prompt_file})
        result = await node.run(ctx)
        assert result[0].assets[0].metadata["error"] == "network down"
        assert result[0].stats["analyze.score.failures"] == 1
        assert result[0].stats["analyze.score.computed"] == 0

    @pytest.mark.asyncio
    async def test_cache_hit_skips_llm(self, prompt_file: str):
        import hashlib

        # Read prompt content and hash it (mirrors the updated score.py logic)
        with open(prompt_file, "r") as f:
            prompt_content = f.read()
        prompt_hash = hashlib.sha256(prompt_content.encode()).hexdigest()[:8]
        cache_key = f"a1:{prompt_hash}"
        cache_data = {cache_key: {"score": 0.99, "is_screenshot": False}}
        cache_manager = _FakeCacheManager(cache_data)

        client = FakeImageClient()
        llm = FakeLLMClient(response={"score": 0.1, "is_screenshot": True})
        assets = [make_asset(id="a1")]
        ctx = make_context(assets=assets, image_client=client, llm_client=llm, cache_manager=cache_manager)
        node = AnalyzeScore({"prompt_file": prompt_file})
        result = await node.run(ctx)
        # Should get cached value, not LLM value
        assert result[0].assets[0].metadata["score"] == 0.99
        assert result[0].stats["analyze.score.cached"] == 1
        assert result[0].stats["analyze.score.computed"] == 0

    @pytest.mark.asyncio
    async def test_progress_reported(self, prompt_file: str):
        client = FakeImageClient()
        llm = FakeLLMClient()
        progress = FakeProgress()
        assets = [make_asset(id="a1"), make_asset(id="a2")]
        ctx = make_context(assets=assets, image_client=client, llm_client=llm, progress=progress)
        node = AnalyzeScore({"prompt_file": prompt_file})
        await node.run(ctx)
        assert "Scoring assets" in progress.stages
        assert progress.advances == 2
