"""Retrieval stages — populate ``ctx.assets`` from external sources.

Stages in this package fetch assets from the image provider and append
them to the pipeline context.
"""

from __future__ import annotations

from smart_albums.nodes.retrieve import by_year  # noqa: F401
from smart_albums.nodes.retrieve import source  # noqa: F401
