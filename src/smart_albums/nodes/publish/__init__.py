"""Publishing stages — create or update albums in the image provider.

Stages in this package are read-only on ``ctx.assets`` and interact with
the image client to persist the final selection as an album.
"""

from __future__ import annotations

from smart_albums.nodes.publish import create_album  # noqa: F401
from smart_albums.nodes.publish import replace_album  # noqa: F401
