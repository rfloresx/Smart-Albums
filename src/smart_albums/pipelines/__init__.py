"""Pipeline definitions — side-effect imports to trigger registration."""

from __future__ import annotations

# Import pipeline modules to trigger @register_pipeline registration.
from smart_albums.pipelines import best_of_year  # noqa: F401
from smart_albums.pipelines import rotation  # noqa: F401
