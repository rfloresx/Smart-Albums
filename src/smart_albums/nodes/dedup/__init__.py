"""Deduplication stages — collapse near-duplicate groups.

Stages in this package group perceptually similar assets and retain only
the highest-scoring member of each group.
"""

from __future__ import annotations

from smart_albums.nodes.dedup import phash  # noqa: F401
from smart_albums.nodes.dedup import phash_composite  # noqa: F401
from smart_albums.nodes.dedup import similar_composite  # noqa: F401
from smart_albums.nodes.dedup import scenes_composite  # noqa: F401
