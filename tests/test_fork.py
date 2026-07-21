"""Tests for smart_albums.nodes.fork."""

from __future__ import annotations

import pytest

from smart_albums.nodes.fork import Fork, Branch
from smart_albums.nodes.select.best import SelectBest
from smart_albums.nodes.filter.videos import RetainImages

from conftest import make_asset, make_scored_asset, make_context


class TestFork:
    """Tests for the Fork composite stage."""

    @pytest.mark.asyncio
    async def test_single_branch(self):
        assets = [
            make_scored_asset(id="a", score=0.8),
            make_scored_asset(id="b", score=0.3),
        ]
        ctx = make_context(assets=assets)

        fork = Fork(branches=[Branch(name="best", pipeline=[SelectBest])])
        result = await fork.func([ctx])
        assert len(result) == 1
        assert result[0].assets[0].id == "a"

    @pytest.mark.asyncio
    async def test_multiple_branches(self):
        assets = [
            make_scored_asset(id="img1", score=0.9, mime_type="image/jpeg"),
            make_scored_asset(id="vid1", score=0.95, mime_type="video/mp4"),
        ]
        ctx = make_context(assets=assets)

        fork = Fork(branches=[
            Branch(name="images_only", pipeline=[RetainImages, SelectBest]),
            Branch(name="all_best", pipeline=[SelectBest]),
        ])
        result = await fork.func([ctx])
        # Two branches produce two contexts
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_branches_get_independent_copies(self):
        assets = [make_scored_asset(id="a", score=0.5)]
        ctx = make_context(assets=assets)

        fork = Fork(branches=[
            Branch(name="b1", pipeline=[SelectBest]),
            Branch(name="b2", pipeline=[SelectBest]),
        ])
        result = await fork.func([ctx])
        # Both branches should have their own context, not interfere
        assert len(result) == 2
        assert result[0].metadata["branch"] == "b1"
        assert result[1].metadata["branch"] == "b2"

    @pytest.mark.asyncio
    async def test_raw_pipeline_shorthand(self):
        assets = [make_scored_asset(id="x", score=0.7)]
        ctx = make_context(assets=assets)

        fork = Fork(branches=[[SelectBest]])
        result = await fork.func([ctx])
        assert len(result) == 1
        assert result[0].assets[0].id == "x"
