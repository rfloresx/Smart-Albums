"""Tests for smart_albums.core.split."""

from __future__ import annotations

from datetime import datetime

from smart_albums.utils.split import split_contexts
from smart_albums.core.models import Asset

from conftest import make_asset, make_context


class TestSplitContexts:
    """Tests for the split_contexts helper."""

    def test_empty_partitions_returns_empty(self):
        ctx = make_context(assets=[make_asset()])
        result = split_contexts(ctx, [], "test")
        assert result == []

    def test_single_partition(self):
        assets = [make_asset(id="a1"), make_asset(id="a2")]
        ctx = make_context(assets=assets)
        result = split_contexts(ctx, [assets], "time")
        assert len(result) == 1
        assert len(result[0].assets) == 2
        assert result[0].partition_name == "time"
        assert result[0].partition_id == "time_0"
        assert result[0].partition_depth == 1

    def test_multiple_partitions(self):
        a1, a2, a3 = make_asset(id="a1"), make_asset(id="a2"), make_asset(id="a3")
        ctx = make_context(assets=[a1, a2, a3])
        partitions = [[a1], [a2, a3]]
        result = split_contexts(ctx, partitions, "phash")
        assert len(result) == 2
        assert len(result[0].assets) == 1
        assert len(result[1].assets) == 2
        assert result[0].partition_id == "phash_0"
        assert result[1].partition_id == "phash_1"

    def test_children_share_config(self):
        ctx = make_context(config={"year": 2024})
        assets = [make_asset(id="a1"), make_asset(id="a2")]
        result = split_contexts(ctx, [[assets[0]], [assets[1]]], "time")
        assert result[0].config is result[1].config
        assert result[0].config["year"] == 2024

    def test_children_have_independent_stats(self):
        ctx = make_context(stats={"original": True})
        assets = [make_asset(id="a1"), make_asset(id="a2")]
        result = split_contexts(ctx, [[assets[0]], [assets[1]]], "time")
        result[0].stats["child"] = "first"
        assert "child" not in result[1].stats

    def test_depth_increments(self):
        ctx = make_context()
        ctx.partition_depth = 2
        assets = [make_asset()]
        result = split_contexts(ctx, [assets], "nested")
        assert result[0].partition_depth == 3

    def test_parent_partition_id_set(self):
        ctx = make_context()
        ctx.partition_id = "time_0"
        assets = [make_asset()]
        result = split_contexts(ctx, [assets], "phash")
        assert result[0].parent_partition_id == "time_0"

    def test_assets_are_deep_copied(self):
        a = make_asset(id="a1", metadata={"score": 0.5})
        ctx = make_context(assets=[a])
        result = split_contexts(ctx, [[a]], "test")
        # Modifying the child's asset should not affect the original
        result[0].assets[0].metadata["score"] = 0.9
        assert a.metadata["score"] == 0.5
