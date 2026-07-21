"""Filter stages — narrow ``ctx.assets`` by content or quality criteria.

Includes both content-filter stages (smart_search, people, on_this_day,
random, none) that use the shared empty-pool handler, and quality-filter
stages (videos, min_score, sensitive) that apply per-asset predicates.
"""

from __future__ import annotations

from smart_albums.nodes.filter import min_score  # noqa: F401
from smart_albums.nodes.filter import none  # noqa: F401
from smart_albums.nodes.filter import on_this_day  # noqa: F401
from smart_albums.nodes.filter import people  # noqa: F401
from smart_albums.nodes.filter import random  # noqa: F401
from smart_albums.nodes.filter import screenshots  # noqa: F401
from smart_albums.nodes.filter import sensitive  # noqa: F401
from smart_albums.nodes.filter import smart_search  # noqa: F401
from smart_albums.nodes.filter import videos  # noqa: F401
