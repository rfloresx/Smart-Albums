"""Tests for smart_albums.nodes.partition.time."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from smart_albums.nodes.partition.time import PartitionTime

from conftest import make_asset, make_context


class TestPartitionTime:
    """Tests for the PartitionTime stage."""

    @pytest.mark.asyncio
    async def test_empty_assets(self):
        ctx = make_context(assets=[], config={"partition.time.time_window_minutes": 5.0})
        node = PartitionTime({"time_window_minutes": 5.0})
        result = await node.run(ctx)
        assert len(result) == 1
        assert result[0].assets == []

    @pytest.mark.asyncio
    async def test_single_asset_one_partition(self):
        assets = [make_asset(id="a1", captured_at=datetime(2024, 6, 1, 12, 0))]
        ctx = make_context(assets=assets, config={"partition.time.time_window_minutes": 5.0})
        node = PartitionTime({"time_window_minutes": 5.0})
        result = await node.run(ctx)
        assert len(result) == 1
        assert len(result[0].assets) == 1

    @pytest.mark.asyncio
    async def test_close_assets_same_partition(self):
        t = datetime(2024, 6, 1, 12, 0)
        assets = [
            make_asset(id="a1", captured_at=t),
            make_asset(id="a2", captured_at=t + timedelta(minutes=2)),
            make_asset(id="a3", captured_at=t + timedelta(minutes=4)),
        ]
        ctx = make_context(assets=assets, config={"partition.time.time_window_minutes": 5.0})
        node = PartitionTime({"time_window_minutes": 5.0})
        result = await node.run(ctx)
        assert len(result) == 1
        assert len(result[0].assets) == 3

    @pytest.mark.asyncio
    async def test_gap_creates_new_partition(self):
        t = datetime(2024, 6, 1, 12, 0)
        assets = [
            make_asset(id="a1", captured_at=t),
            make_asset(id="a2", captured_at=t + timedelta(minutes=2)),
            make_asset(id="a3", captured_at=t + timedelta(minutes=10)),
        ]
        ctx = make_context(assets=assets, config={"partition.time.time_window_minutes": 5.0})
        node = PartitionTime({"time_window_minutes": 5.0})
        result = await node.run(ctx)
        assert len(result) == 2
        assert len(result[0].assets) == 2
        assert len(result[1].assets) == 1

    @pytest.mark.asyncio
    async def test_unsorted_input_gets_sorted(self):
        t = datetime(2024, 6, 1, 12, 0)
        assets = [
            make_asset(id="a3", captured_at=t + timedelta(minutes=20)),
            make_asset(id="a1", captured_at=t),
            make_asset(id="a2", captured_at=t + timedelta(minutes=2)),
        ]
        ctx = make_context(assets=assets, config={"partition.time.time_window_minutes": 5.0})
        node = PartitionTime({"time_window_minutes": 5.0})
        result = await node.run(ctx)
        # a1 and a2 are within 5min, a3 is 20min later
        assert len(result) == 2
        assert result[0].assets[0].id == "a1"
        assert result[0].assets[1].id == "a2"
        assert result[1].assets[0].id == "a3"

    @pytest.mark.asyncio
    async def test_stats_populated(self):
        t = datetime(2024, 6, 1, 12, 0)
        assets = [
            make_asset(id="a1", captured_at=t),
            make_asset(id="a2", captured_at=t + timedelta(hours=1)),
        ]
        ctx = make_context(assets=assets, config={"partition.time.time_window_minutes": 5.0})
        node = PartitionTime({"time_window_minutes": 5.0})
        result = await node.run(ctx)
        assert result[0].stats["partition.time.total_assets"] == 2
        assert result[0].stats["partition.time.partitions_created"] == 2

    @pytest.mark.asyncio
    async def test_partition_naming(self):
        t = datetime(2024, 6, 1, 12, 0)
        assets = [
            make_asset(id="a1", captured_at=t),
            make_asset(id="a2", captured_at=t + timedelta(hours=1)),
        ]
        ctx = make_context(assets=assets, config={"partition.time.time_window_minutes": 5.0})
        node = PartitionTime({"time_window_minutes": 5.0})
        result = await node.run(ctx)
        assert result[0].partition_name == "time"
        assert result[0].partition_id == "time_0"
        assert result[1].partition_id == "time_1"
