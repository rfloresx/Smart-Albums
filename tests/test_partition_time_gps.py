"""Tests for smart_albums.nodes.partition.time_gps."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from smart_albums.nodes.partition.time_gps import (
    PartitionTimeGps,
    _haversine_meters,
    _gps_within,
)

from conftest import make_asset, make_context


class TestHaversineMeters:
    """Tests for _haversine_meters helper."""

    def test_same_point_is_zero(self):
        assert _haversine_meters(48.8566, 2.3522, 48.8566, 2.3522) == 0.0

    def test_known_distance(self):
        # Paris to London is approximately 340-345km
        dist = _haversine_meters(48.8566, 2.3522, 51.5074, -0.1278)
        assert 340_000 < dist < 345_000

    def test_short_distance(self):
        # Two points ~111m apart (0.001 degree latitude ≈ 111m)
        dist = _haversine_meters(0.0, 0.0, 0.001, 0.0)
        assert 100 < dist < 120


class TestGpsWithin:
    """Tests for _gps_within helper."""

    def test_same_location(self):
        a = make_asset(id="a", latitude=48.0, longitude=2.0)
        b = make_asset(id="b", latitude=48.0, longitude=2.0)
        assert _gps_within(a, b, 100.0) is True

    def test_far_apart(self):
        a = make_asset(id="a", latitude=48.0, longitude=2.0)
        b = make_asset(id="b", latitude=51.0, longitude=-0.1)
        assert _gps_within(a, b, 100.0) is False

    def test_missing_latitude_returns_false(self):
        a = make_asset(id="a", latitude=None, longitude=2.0)
        b = make_asset(id="b", latitude=48.0, longitude=2.0)
        assert _gps_within(a, b, 100.0) is False

    def test_missing_longitude_returns_false(self):
        a = make_asset(id="a", latitude=48.0, longitude=None)
        b = make_asset(id="b", latitude=48.0, longitude=2.0)
        assert _gps_within(a, b, 100.0) is False


class TestPartitionTimeGpsStage:
    """Tests for the PartitionTimeGps stage."""

    @pytest.mark.asyncio
    async def test_empty_assets(self):
        ctx = make_context(
            assets=[],
            config={
                "partition.time_gps.time_window_minutes": 5.0,
                "partition.time_gps.gps_window_meters": 100.0,
            },
        )
        node = PartitionTimeGps({})
        result = await node.run(ctx)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_close_time_and_gps_same_partition(self):
        t = datetime(2024, 6, 1, 12, 0)
        assets = [
            make_asset(id="a1", captured_at=t, latitude=48.8566, longitude=2.3522),
            make_asset(id="a2", captured_at=t + timedelta(minutes=2), latitude=48.8566, longitude=2.3522),
        ]
        ctx = make_context(
            assets=assets,
            config={
                "partition.time_gps.time_window_minutes": 5.0,
                "partition.time_gps.gps_window_meters": 100.0,
            },
        )
        node = PartitionTimeGps({})
        result = await node.run(ctx)
        assert len(result) == 1
        assert len(result[0].assets) == 2

    @pytest.mark.asyncio
    async def test_close_time_but_far_gps_splits(self):
        t = datetime(2024, 6, 1, 12, 0)
        assets = [
            make_asset(id="a1", captured_at=t, latitude=48.8566, longitude=2.3522),
            make_asset(id="a2", captured_at=t + timedelta(minutes=2), latitude=51.5074, longitude=-0.1278),
        ]
        ctx = make_context(
            assets=assets,
            config={
                "partition.time_gps.time_window_minutes": 5.0,
                "partition.time_gps.gps_window_meters": 100.0,
            },
        )
        node = PartitionTimeGps({})
        result = await node.run(ctx)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_far_time_but_close_gps_splits(self):
        t = datetime(2024, 6, 1, 12, 0)
        assets = [
            make_asset(id="a1", captured_at=t, latitude=48.8566, longitude=2.3522),
            make_asset(id="a2", captured_at=t + timedelta(hours=1), latitude=48.8566, longitude=2.3522),
        ]
        ctx = make_context(
            assets=assets,
            config={
                "partition.time_gps.time_window_minutes": 5.0,
                "partition.time_gps.gps_window_meters": 100.0,
            },
        )
        node = PartitionTimeGps({})
        result = await node.run(ctx)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_missing_gps_always_splits(self):
        t = datetime(2024, 6, 1, 12, 0)
        assets = [
            make_asset(id="a1", captured_at=t, latitude=48.0, longitude=2.0),
            make_asset(id="a2", captured_at=t + timedelta(minutes=1)),  # no GPS
        ]
        ctx = make_context(
            assets=assets,
            config={
                "partition.time_gps.time_window_minutes": 5.0,
                "partition.time_gps.gps_window_meters": 100.0,
            },
        )
        node = PartitionTimeGps({})
        result = await node.run(ctx)
        assert len(result) == 2
