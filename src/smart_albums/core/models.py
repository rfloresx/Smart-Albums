"""Data models for the smart_albums pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


@dataclass
class Asset:
    """A photo asset flowing through the pipeline.

    Stages enrich the asset by writing into `metadata`. This keeps the core
    class stable while allowing arbitrary stage-specific data to travel with
    the asset through the entire pipeline.

    The `latitude` and `longitude` fields carry GPS coordinates for
    location-aware partitioning stages (e.g. `partition.time_gps`).
    """

    id: str
    filename: str
    captured_at: datetime
    mime_type: str
    metadata: dict[str, Any] = field(default_factory=dict)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
