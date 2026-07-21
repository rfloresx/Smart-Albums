"""Tests for smart_albums.nodes.merge."""

from __future__ import annotations

import pytest

from smart_albums.nodes.merge import merge_concat, _flatten_stats, MergeConcat
from smart_albums.core.context import PipelineContext

from conftest import make_asset, make_context


class TestFlattenStats:
    """Tests for _flatten_stats aggregation logic."""

    def test_empty_list(self):
        assert _flatten_stats([]) == {}

    def test_single_stats(self):
        result = _flatten_stats([{"count": 5, "name": "foo"}])
        assert result == {"count": 5, "name": "foo"}

    def test_numeric_values_summed(self):
        result = _flatten_stats([{"count": 3}, {"count": 7}])
        assert result["count"] == 10

    def test_float_values_summed(self):
        result = _flatten_stats([{"score": 0.5}, {"score": 0.3}])
        assert abs(result["score"] - 0.8) < 1e-9

    def test_list_values_concatenated(self):
        result = _flatten_stats([{"items": [1, 2]}, {"items": [3, 4]}])
        assert result["items"] == [1, 2, 3, 4]

    def test_other_types_first_wins(self):
        result = _flatten_stats([{"label": "a"}, {"label": "b"}])
        assert result["label"] == "a"

    def test_mixed_keys(self):
        result = _flatten_stats([
            {"count": 2, "items": ["x"]},
            {"count": 3, "items": ["y"], "extra": "yes"},
        ])
        assert result["count"] == 5
        assert result["items"] == ["x", "y"]
        assert result["extra"] == "yes"


class TestMergeConcat:
    """Tests for the merge_concat function."""

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="empty"):
            merge_concat([])

    def test_single_context(self):
        ctx = make_context(
            assets=[make_asset(id="a1")],
            stats={"count": 1},
        )
        ctx.partition_depth = 1
        result = merge_concat([ctx])
        assert len(result.assets) == 1
        assert result.partition_depth == 0

    def test_multiple_contexts_concatenates_assets(self):
        ctx1 = make_context(assets=[make_asset(id="a1")])
        ctx1.partition_depth = 1
        ctx2 = make_context(assets=[make_asset(id="a2"), make_asset(id="a3")])
        ctx2.partition_depth = 1
        result = merge_concat([ctx1, ctx2])
        assert len(result.assets) == 3
        assert [a.id for a in result.assets] == ["a1", "a2", "a3"]

    def test_depth_reduced_by_one(self):
        ctx1 = make_context(assets=[make_asset()])
        ctx1.partition_depth = 2
        ctx2 = make_context(assets=[make_asset(id="b")])
        ctx2.partition_depth = 2
        result = merge_concat([ctx1, ctx2])
        assert result.partition_depth == 1

    def test_depth_does_not_go_negative(self):
        ctx = make_context(assets=[make_asset()])
        ctx.partition_depth = 0
        result = merge_concat([ctx])
        assert result.partition_depth == 0

    def test_stats_merged(self):
        ctx1 = make_context(assets=[make_asset(id="a")], stats={"x": 3})
        ctx1.partition_depth = 1
        ctx2 = make_context(assets=[make_asset(id="b")], stats={"x": 7})
        ctx2.partition_depth = 1
        result = merge_concat([ctx1, ctx2])
        assert result.stats["x"] == 10


class TestMergeConcatStage:
    """Tests for the MergeConcat stage node."""

    @pytest.mark.asyncio
    async def test_empty_batch(self):
        node = MergeConcat({})
        result = await node.func([])
        assert result == []

    @pytest.mark.asyncio
    async def test_all_depth_zero_merged(self):
        ctx1 = make_context(assets=[make_asset(id="a")])
        ctx1.partition_depth = 0
        ctx2 = make_context(assets=[make_asset(id="b")])
        ctx2.partition_depth = 0
        node = MergeConcat({})
        result = await node.func([ctx1, ctx2])
        assert len(result) == 1
        assert len(result[0].assets) == 2

    @pytest.mark.asyncio
    async def test_groups_by_parent_partition_id(self):
        ctx1 = make_context(assets=[make_asset(id="a")])
        ctx1.partition_depth = 1
        ctx1.parent_partition_id = "parent_0"
        ctx2 = make_context(assets=[make_asset(id="b")])
        ctx2.partition_depth = 1
        ctx2.parent_partition_id = "parent_0"
        ctx3 = make_context(assets=[make_asset(id="c")])
        ctx3.partition_depth = 1
        ctx3.parent_partition_id = "parent_1"
        node = MergeConcat({})
        result = await node.func([ctx1, ctx2, ctx3])
        assert len(result) == 2
        # One group has 2 assets, the other has 1
        sizes = sorted(len(r.assets) for r in result)
        assert sizes == [1, 2]
