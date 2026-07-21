"""Tests for smart_albums.nodes.select.best."""

from __future__ import annotations

import pytest

from smart_albums.nodes.select.best import SelectBest

from conftest import make_asset, make_scored_asset, make_context


class TestSelectBest:
    """Tests for the SelectBest stage."""

    @pytest.mark.asyncio
    async def test_empty_assets(self):
        ctx = make_context(assets=[])
        node = SelectBest({})
        result = await node.run(ctx)
        assert len(result) == 1
        assert result[0].assets == []
        assert result[0].stats["select.best.groups_processed"] == 0

    @pytest.mark.asyncio
    async def test_single_asset(self):
        asset = make_scored_asset(id="only", score=0.7)
        ctx = make_context(assets=[asset])
        node = SelectBest({})
        result = await node.run(ctx)
        assert len(result[0].assets) == 1
        assert result[0].assets[0].id == "only"
        assert result[0].assets[0].metadata["group_best"] is True

    @pytest.mark.asyncio
    async def test_picks_highest_score(self):
        assets = [
            make_scored_asset(id="low", score=0.3),
            make_scored_asset(id="high", score=0.95),
            make_scored_asset(id="mid", score=0.6),
        ]
        ctx = make_context(assets=assets)
        node = SelectBest({})
        result = await node.run(ctx)
        assert result[0].assets[0].id == "high"

    @pytest.mark.asyncio
    async def test_tie_break_by_smallest_id(self):
        assets = [
            make_scored_asset(id="zzz", score=0.8),
            make_scored_asset(id="aaa", score=0.8),
            make_scored_asset(id="mmm", score=0.8),
        ]
        ctx = make_context(assets=assets)
        node = SelectBest({})
        result = await node.run(ctx)
        assert result[0].assets[0].id == "aaa"

    @pytest.mark.asyncio
    async def test_assets_without_score_default_to_zero(self):
        assets = [
            make_asset(id="no-score"),
            make_scored_asset(id="scored", score=0.1),
        ]
        ctx = make_context(assets=assets)
        node = SelectBest({})
        result = await node.run(ctx)
        assert result[0].assets[0].id == "scored"

    @pytest.mark.asyncio
    async def test_stats_populated(self):
        assets = [make_scored_asset(id="a"), make_scored_asset(id="b")]
        ctx = make_context(assets=assets)
        node = SelectBest({})
        result = await node.run(ctx)
        assert result[0].stats["select.best.groups_processed"] == 1
        assert result[0].stats["select.best.representatives_selected"] == 1
