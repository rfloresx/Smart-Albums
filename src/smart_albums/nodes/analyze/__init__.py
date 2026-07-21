"""Analysis stages — enrich ``ctx.assets`` with scores and hashes.

Stages in this package write metadata keys (score, phash, is_screenshot)
onto each asset without removing any assets from the pool.
"""

from __future__ import annotations

from smart_albums.nodes.analyze import embedding  # noqa: F401
from smart_albums.nodes.analyze import phash  # noqa: F401
from smart_albums.nodes.analyze import score  # noqa: F401
