"""Selection stages — pick the final subset of assets for publishing.

Stages in this package apply balanced selection and freshness filtering
to choose assets for the output album.
"""

from __future__ import annotations

from smart_albums.nodes.select import avoid_repeat  # noqa: F401
from smart_albums.nodes.select import balanced  # noqa: F401
from smart_albums.nodes.select import best  # noqa: F401
from smart_albums.nodes.select import diverse  # noqa: F401
from smart_albums.nodes.select import top_n  # noqa: F401

