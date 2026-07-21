"""Tests for smart_albums.nodes.filter.min_score."""

from __future__ import annotations

import pytest

from smart_albums.nodes.filter.min_score import FilterMinScore

from conftest import make_scored_asset, make_asset, make_context


class TestFilterMinScore:
    """Tests for the FilterMinScore stage."""

    @pytest.mark.asyncio
    async def test_zero_threshold_keeps_all(self):
        assets = [make_scored_asset(id="a", score=0.1), make_scored_asset(id="b", score=0.9)]
        ctx = make_context(assets=assets)
        node = FilterMinScore({"threshold": 0.0})
        result = await node.run(ctx)
        assert len(result[0].assets) == 2
        assert result[0].stats["filter.min_score.excluded"] == 0

    @pytest.mark.asyncio
    async def test_threshold_excludes_below(self):
        assets = [
            make_scored_asset(id="low", score=0.2),
            make_scored_asset(id="high", score=0.8),
        ]
        ctx = make_context(assets=assets)
        node = FilterMinScore({"threshold": 0.5})
        result = await node.run(ctx)
        assert len(result[0].assets) == 1
        assert result[0].assets[0].id == "high"
        assert result[0].stats["filter.min_score.excluded"] == 1

    @pytest.mark.asyncio
    async def test_threshold_at_boundary_keeps_equal(self):
        assets = [make_scored_asset(id="exact", score=0.5)]
        ctx = make_context(assets=assets)
        node = FilterMinScore({"threshold": 0.5})
        result = await node.run(ctx)
        assert len(result[0].assets) == 1

    @pytest.mark.asyncio
    async def test_asset_without_score_excluded(self):
        assets = [make_asset(id="no-score"), make_scored_asset(id="scored", score=0.6)]
        ctx = make_context(assets=assets)
        node = FilterMinScore({"threshold": 0.3})
        result = await node.run(ctx)
        assert len(result[0].assets) == 1
        assert result[0].assets[0].id == "scored"

    @pytest.mark.asyncio
    async def test_all_excluded(self):
        assets = [make_scored_asset(id="a", score=0.1), make_scored_asset(id="b", score=0.2)]
        ctx = make_context(assets=assets)
        node = FilterMinScore({"threshold": 0.9})
        result = await node.run(ctx)
        assert len(result[0].assets) == 0
        assert result[0].stats["filter.min_score.excluded"] == 2
