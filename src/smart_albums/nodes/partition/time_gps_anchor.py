"""partition.time_gps_anchor — two-stage partitioning by time then GPS anchor.

Replicates the original BestOfPipeline partitioning strategy:

1. **Time partitioning**: Sorts assets by captured_at and groups consecutive
   assets that fall within `time_window_minutes` of each other.
   Assets without a timestamp go into a dedicated "unknown time" partition.

2. **GPS sub-partitioning**: Within each time cluster, uses a greedy
   anchor-based approach — each asset with GPS coordinates is assigned to
   the first existing sub-partition whose anchor is within
   `gps_distance_meters` (haversine). If no existing sub-partition qualifies,
   a new one is created with this asset as the anchor.

   Assets **without** GPS stay together in their time cluster (they are NOT
   split into individual partitions).
"""

from __future__ import annotations

import math
from datetime import timedelta
from typing import Optional

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.models import Asset
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


def _get_gps(asset: Asset) -> Optional[tuple[float, float]]:
    """Extract GPS coordinates from an asset.

    Returns:
        (latitude, longitude) tuple if both are present, else None.
    """
    if asset.latitude is not None and asset.longitude is not None:
        return (asset.latitude, asset.longitude)
    return None


def _partition_by_time(
    assets: list[Asset], time_window: timedelta
) -> tuple[list[list[Asset]], list[Asset]]:
    """Sort assets by captured_at and split into time clusters.

    A new cluster starts when the gap between consecutive assets exceeds
    `time_window`.

    Returns:
        Tuple of (time_clusters, untimed_assets). Assets with the epoch
        minimum datetime are treated as "no timestamp" and separated out.
    """
    timed: list[Asset] = []
    untimed: list[Asset] = []

    for asset in assets:
        # Treat assets with no meaningful timestamp as untimed
        if asset.captured_at is None:
            untimed.append(asset)
        else:
            timed.append(asset)

    if not timed:
        return [], untimed

    sorted_assets = sorted(timed, key=lambda a: (a.captured_at, a.id))

    clusters: list[list[Asset]] = [[sorted_assets[0]]]
    for asset in sorted_assets[1:]:
        prev = clusters[-1][-1]
        if (asset.captured_at - prev.captured_at) <= time_window:
            clusters[-1].append(asset)
        else:
            clusters.append([asset])

    return clusters, untimed


def _partition_by_gps_anchor(
    cluster: list[Asset], gps_distance_meters: float
) -> list[list[Asset]]:
    """Sub-partition a time cluster by GPS using greedy anchor-based clustering.

    Iterates through assets with GPS coords and assigns each to the first
    existing sub-partition whose anchor is within `gps_distance_meters`.
    If none matches, a new sub-partition is created with this asset as anchor.

    Assets without GPS stay together in a single group.

    Args:
        cluster: A time cluster of assets.
        gps_distance_meters: Maximum distance to an anchor for membership.

    Returns:
        List of asset groups (sub-partitions).
    """
    if not cluster:
        return []

    gps_assets: list[tuple[Asset, float, float]] = []
    no_gps_assets: list[Asset] = []

    for asset in cluster:
        coords = _get_gps(asset)
        if coords is not None:
            gps_assets.append((asset, coords[0], coords[1]))
        else:
            no_gps_assets.append(asset)

    # If no assets have GPS, return the whole cluster as one partition
    if not gps_assets:
        return [cluster]

    # Greedy anchor-based clustering
    # Each sub-partition is (anchor_lat, anchor_lon, members)
    gps_partitions: list[tuple[float, float, list[Asset]]] = []

    for asset, lat, lon in gps_assets:
        placed = False
        for anchor_lat, anchor_lon, members in gps_partitions:
            dist = _haversine_meters(lat, lon, anchor_lat, anchor_lon)
            if dist <= gps_distance_meters:
                members.append(asset)
                placed = True
                break
        if not placed:
            gps_partitions.append((lat, lon, [asset]))

    # Build result: GPS sub-partitions + no-GPS partition
    result: list[list[Asset]] = [members for _, _, members in gps_partitions]

    if no_gps_assets:
        result.append(no_gps_assets)

    return result


@stage("partition.time_gps_anchor")
class PartitionTimeGpsAnchor(Stage):
    """Two-stage partitioning: time clusters then GPS anchor-based sub-partitions.

    Stage 1: Sort by captured_at, group consecutive assets within
    `time_window_minutes`. Assets without timestamps get their own partition.

    Stage 2: Within each time cluster, sub-partition by GPS using a greedy
    anchor approach. Assets without GPS coordinates remain grouped within
    their time cluster (they are never split into individual partitions).
    """

    _config_schema = (
        ConfigParam(
            key="time_window_minutes",
            type=float,
            default=5.0,
            min=0.1,
            description="Maximum gap in minutes between consecutive assets within a time cluster.",
        ),
        ConfigParam(
            key="gps_window_meters",
            type=float,
            default=100.0,
            min=0.0,
            description="Maximum haversine distance in meters to an anchor for GPS sub-partitioning.",
        ),
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        if not ctx.assets:
            return [ctx]

        time_window = timedelta(minutes=self.get("time_window_minutes"))
        gps_distance: float = self.get("gps_window_meters")

        # Stage 1: Time partitioning
        time_clusters, untimed = _partition_by_time(ctx.assets, time_window)

        # Stage 2: GPS sub-partitioning within each time cluster
        all_partitions: list[list[Asset]] = []
        for cluster in time_clusters:
            sub_partitions = _partition_by_gps_anchor(cluster, gps_distance)
            all_partitions.extend(sub_partitions)

        # Add untimed assets as their own partition
        if untimed:
            all_partitions.append(untimed)

        results = split_contexts(ctx, all_partitions, "time_gps_anchor")
        if results:
            results[0].stats["partition.time_gps_anchor.total_assets"] = len(ctx.assets)
            results[0].stats["partition.time_gps_anchor.partitions_created"] = len(all_partitions)
            results[0].stats["partition.time_gps_anchor.time_clusters"] = len(time_clusters)
            results[0].stats["partition.time_gps_anchor.untimed_assets"] = len(untimed)
        return results
