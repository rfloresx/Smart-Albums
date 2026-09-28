"""partition.time_gps — partition assets by temporal AND GPS proximity.

Sorts assets by captured_at (ascending, tie-break by id) and creates a new
partition whenever the gap between consecutive assets exceeds the configured
time window OR the GPS distance exceeds the configured distance window.

Both time AND GPS constraints must be satisfied for assets to remain in the
same partition. Missing GPS coordinates on either asset are treated as
infinite distance (always split).
"""

from __future__ import annotations

import math
from datetime import timedelta

from smart_albums.core.context import PipelineContext, ContextBatch
from protocols_system.protocols import Asset
from smart_albums.core.node import Stage, ConfigParam
from smart_albums.core.registry import stage
from smart_albums.utils.split import split_contexts


def _haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute the great-circle distance between two points using Haversine.

    Args:
        lat1: Latitude of point 1 in degrees.
        lon1: Longitude of point 1 in degrees.
        lat2: Latitude of point 2 in degrees.
        lon2: Longitude of point 2 in degrees.

    Returns:
        Distance in meters between the two points.
    """
    R = 6_371_000  # Earth radius in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _gps_within(prev: Asset, current: Asset, max_meters: float) -> bool:
    """Check whether two assets are within GPS distance threshold.

    If either asset is missing latitude or longitude, the distance is treated
    as infinite (returns False — always split).
    """
    if (
        prev.latitude is None
        or prev.longitude is None
        or current.latitude is None
        or current.longitude is None
    ):
        return False

    distance = _haversine_meters(prev.latitude, prev.longitude, current.latitude, current.longitude)
    return distance <= max_meters


@stage("partition.time_gps")
class PartitionTimeGps(Stage):
    """Partition assets by temporal AND GPS proximity.

    Both conditions must be satisfied (time within window AND GPS within
    window) for two consecutive assets to remain in the same partition.
    """

    _config_schema = (
        ConfigParam(
            key="time_window_minutes",
            type=float,
            default=5.0,
            min=0.1,
            description="Maximum gap in minutes between consecutive assets within a partition.",
        ),
        ConfigParam(
            key="gps_window_meters",
            type=float,
            default=100.0,
            min=0.0,
            description="Maximum GPS distance in meters between consecutive assets within a partition.",
        ),
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        if not ctx.assets:
            return [ctx]

        time_window = timedelta(
            minutes=self.get("time_window_minutes")
        )
        gps_window: float = self.get("gps_window_meters")

        # Separate assets without a timestamp — they cannot be time-partitioned
        timed_assets = [a for a in ctx.assets if a.captured_at is not None]
        untimed_assets = [a for a in ctx.assets if a.captured_at is None]

        if not timed_assets:
            return [ctx]

        sorted_assets = sorted(timed_assets, key=lambda a: (a.captured_at, a.id))

        partitions: list[list[Asset]] = [[sorted_assets[0]]]
        for asset in sorted_assets[1:]:
            prev = partitions[-1][-1]
            # Both are timed (filtered above); guard narrows the optional type.
            if asset.captured_at is None or prev.captured_at is None:
                partitions.append([asset])
                continue
            time_ok = (asset.captured_at - prev.captured_at) <= time_window
            gps_ok = _gps_within(prev, asset, gps_window)

            if time_ok and gps_ok:
                partitions[-1].append(asset)
            else:
                partitions.append([asset])

        # Add untimed assets as their own partition if any exist
        if untimed_assets:
            partitions.append(untimed_assets)

        results = split_contexts(ctx, partitions, "time_gps")
        if results:
            results[0].stats["partition.time_gps.total_assets"] = len(ctx.assets)
            results[0].stats["partition.time_gps.partitions_created"] = len(partitions)
            results[0].stats["partition.time_gps.untimed_assets"] = len(untimed_assets)
        return results
