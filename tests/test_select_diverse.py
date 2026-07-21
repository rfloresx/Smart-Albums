"""Tests for smart_albums.nodes.select.diverse."""

from __future__ import annotations

import pytest

from smart_albums.nodes.select.diverse import SelectDiversePick, _mmr_select

from conftest import make_asset, make_scored_asset, make_context


def _make_embedded_asset(id: str, score: float, embedding: list[float]) -> object:
    """Create an asset with score and embedding."""
    return make_asset(id=id, metadata={"score": score, "embedding": embedding})


class TestMmrSelect:
    """Tests for _mmr_select algorithm."""

    def test_first_pick_is_highest_quality(self):
        assets = [
            _make_embedded_asset("low", 0.3, [1.0, 0.0]),
            _make_embedded_asset("high", 0.9, [0.0, 1.0]),
        ]
        picks = _mmr_select(assets, pick_limit=1, quality_weight=0.5, diversity_weight=0.5)
        assert len(picks) == 1
        assert picks[0].id == "high"

    def test_respects_pick_limit(self):
        assets = [
            _make_embedded_asset("a", 0.9, [1.0, 0.0, 0.0]),
            _make_embedded_asset("b", 0.8, [0.0, 1.0, 0.0]),
            _make_embedded_asset("c", 0.7, [0.0, 0.0, 1.0]),
        ]
        picks = _mmr_select(assets, pick_limit=2, quality_weight=0.3, diversity_weight=0.7)
        assert len(picks) == 2

    def test_diversity_promotes_different_embeddings(self):
        # Three assets: two are similar, one is different
        assets = [
            _make_embedded_asset("sim1", 0.8, [1.0, 0.0, 0.0]),
            _make_embedded_asset("sim2", 0.79, [0.99, 0.01, 0.0]),
            _make_embedded_asset("diff", 0.7, [0.0, 0.0, 1.0]),
        ]
        # With high diversity weight, "diff" should be picked over "sim2"
        picks = _mmr_select(assets, pick_limit=2, quality_weight=0.1, diversity_weight=0.9)
        assert picks[0].id == "sim1"
        assert picks[1].id == "diff"

    def test_no_embeddings_falls_back_to_quality(self):
        assets = [
            make_scored_asset(id="a", score=0.9),
            make_scored_asset(id="b", score=0.7),
            make_scored_asset(id="c", score=0.8),
        ]
        picks = _mmr_select(assets, pick_limit=2, quality_weight=0.5, diversity_weight=0.5)
        assert picks[0].id == "a"
        assert picks[1].id == "c"  # second highest

    def test_tie_break_by_smallest_id(self):
        assets = [
            _make_embedded_asset("zzz", 0.8, [1.0, 0.0]),
            _make_embedded_asset("aaa", 0.8, [1.0, 0.0]),
        ]
        picks = _mmr_select(assets, pick_limit=1, quality_weight=0.5, diversity_weight=0.5)
        assert picks[0].id == "aaa"


class TestSelectDiversePickStage:
    """Tests for the SelectDiversePick stage."""

    @pytest.mark.asyncio
    async def test_empty_assets(self):
        ctx = make_context(
            assets=[],
            config={
                "select.diverse_pick.max_picks": 3,
                "select.diverse_pick.pick_percentage": 0.33,
                "select.diverse_pick.quality_weight": 0.3,
                "select.diverse_pick.diversity_weight": 0.7,
            },
        )
        node = SelectDiversePick({})
        result = await node.run(ctx)
        assert result[0].stats["picks_selected"] == 0

    @pytest.mark.asyncio
    async def test_single_asset(self):
        ctx = make_context(
            assets=[make_scored_asset(id="solo", score=0.8)],
            config={
                "select.diverse_pick.max_picks": 3,
                "select.diverse_pick.pick_percentage": 0.33,
                "select.diverse_pick.quality_weight": 0.3,
                "select.diverse_pick.diversity_weight": 0.7,
            },
        )
        node = SelectDiversePick({})
        result = await node.run(ctx)
        assert len(result[0].assets) == 1
        assert result[0].assets[0].metadata["group_pick"] is True
        assert result[0].assets[0].metadata["group_best"] is True

    @pytest.mark.asyncio
    async def test_multiple_assets_picks_limited(self):
        assets = [
            _make_embedded_asset("a", 0.9, [1.0, 0.0, 0.0]),
            _make_embedded_asset("b", 0.8, [0.0, 1.0, 0.0]),
            _make_embedded_asset("c", 0.7, [0.0, 0.0, 1.0]),
            _make_embedded_asset("d", 0.6, [0.5, 0.5, 0.0]),
            _make_embedded_asset("e", 0.5, [0.0, 0.5, 0.5]),
        ]
        ctx = make_context(
            assets=assets,
            config={
                "select.diverse_pick.max_picks": 2,
                "select.diverse_pick.pick_percentage": 0.5,
                "select.diverse_pick.quality_weight": 0.3,
                "select.diverse_pick.diversity_weight": 0.7,
            },
        )
        node = SelectDiversePick({})
        result = await node.run(ctx)
        # max_picks=2 caps the selection
        assert len(result[0].assets) == 2
        assert result[0].assets[0].metadata["group_best"] is True
