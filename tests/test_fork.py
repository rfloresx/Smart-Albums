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
        # Two assets: one image, one video. The "images_only" branch filters
        # the video out before SelectBest, so it must pick the image; the
        # "all_best" branch sees both and picks the higher-scored video.
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

        # Two branches produce two contexts, tagged with their branch names.
        assert len(result) == 2
        by_branch = {c.metadata["branch"]: c for c in result}
        assert set(by_branch) == {"images_only", "all_best"}

        # Assert the actual *content* each branch produced, not just the
        # context count (TS-03) — the old test only checked len(result)==2,
        # which passed even if both branches had emitted the wrong asset or
        # nothing at all.
        images_only = by_branch["images_only"]
        assert [a.id for a in images_only.assets] == ["img1"], (
            "images_only branch should drop the video and keep the image"
        )

        all_best = by_branch["all_best"]
        assert [a.id for a in all_best.assets] == ["vid1"], (
            "all_best branch should pick the highest-scored asset (the video)"
        )

    @pytest.mark.asyncio
    async def test_branches_get_independent_copies(self):
        # Give each branch a no-op pipeline so the input assets pass through
        # unchanged, then mutate one branch's result and prove the other
        # branch's asset — and the original input — are unaffected (TS-03).
        # The old test only checked the branch tags, never proving the
        # per-branch asset objects were actually independent copies.
        from smart_albums.nodes.filter.none import FilterNone

        assets = [make_scored_asset(id="a", score=0.5)]
        ctx = make_context(assets=assets)
        original_asset = assets[0]

        fork = Fork(branches=[
            Branch(name="b1", pipeline=[FilterNone]),
            Branch(name="b2", pipeline=[FilterNone]),
        ])
        result = await fork.func([ctx])

        assert len(result) == 2
        assert result[0].metadata["branch"] == "b1"
        assert result[1].metadata["branch"] == "b2"

        b1_asset = result[0].assets[0]
        b2_asset = result[1].assets[0]

        # Each branch must receive a distinct Asset object (deep-copied),
        # not the same shared instance.
        assert b1_asset is not b2_asset
        assert b1_asset is not original_asset
        assert b2_asset is not original_asset

        # Mutating b1's asset metadata must not bleed into b2's copy or the
        # original input asset.
        b1_asset.metadata["mutated_by_b1"] = True
        assert "mutated_by_b1" not in b2_asset.metadata
        assert "mutated_by_b1" not in original_asset.metadata

    @pytest.mark.asyncio
    async def test_raw_pipeline_shorthand(self):
        assets = [make_scored_asset(id="x", score=0.7)]
        ctx = make_context(assets=assets)

        fork = Fork(branches=[[SelectBest]])
        result = await fork.func([ctx])
        assert len(result) == 1
        assert result[0].assets[0].id == "x"
