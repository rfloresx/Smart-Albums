"""Export stages — write pipeline context and assets to the local filesystem.

Stages in this package serialize the current pipeline state (assets +
metadata) to a structured output directory for downstream consumption
(printing, external tools, backup, etc.).
"""

from __future__ import annotations

from smart_albums.nodes.export import context  # noqa: F401
