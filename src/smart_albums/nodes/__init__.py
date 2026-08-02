"""Pipeline nodes — auto-registration via category package imports.

Importing this package triggers all ``@stage`` registrations so that
``STAGE_REGISTRY`` is fully populated before ``run_pipeline()`` executes.
"""

from __future__ import annotations

# Import all category packages to trigger @stage decorator registration.
# Each category __init__.py imports its stage modules.
from smart_albums.nodes import (  # noqa: F401
    analyze,
    dedup,
    enrich,
    export,
    filter,
    fork,
    merge,
    partition,
    publish,
    retrieve,
    select,
)
