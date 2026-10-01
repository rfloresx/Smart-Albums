"""Shared "last shown" publish-history helper.

``select.avoid_repeat`` filters out assets that were shown (published to an
album) within a cooldown window, by reading ``metadata["last_shown_date"]``
on each asset. For that to do anything, *something* has to record when an
asset was last published — but nothing did (ND-11): the publish stages
created/updated albums without ever writing history back anywhere, and the
image-library protocol (``IImageClient``) exposes no per-asset metadata
write, so there is no server-side place to store it either.

This module records publish history in the same JSONL-backed cache the
analyze stages already use (``ICacheManager``), under a dedicated cache
name. ``avoid_repeat`` then reads it as a fallback when an asset's own
metadata doesn't already carry a ``last_shown_date`` (e.g. set by a
caller that tracks history elsewhere), so the cooldown actually takes
effect across runs.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Iterable, Optional

from protocols_system import ProtocolsRegistry
from protocols_system.protocols import ICacheManager

logger = logging.getLogger(__name__)

# Cache name (becomes "<name>.jsonl" in the cache dir).
HISTORY_CACHE_NAME = "publish_history"


def record_shown(asset_ids: Iterable[str], shown_on: date) -> int:
    """Record that ``asset_ids`` were shown (published) on ``shown_on``.

    Stores one entry per asset id → ISO date string in the publish-history
    cache. A later shown date for the same asset overwrites an earlier one.

    Returns the number of ids recorded (0 if no cache manager is
    registered, in which case history is silently a no-op — the same
    degraded behavior the analyze stages have when run without a cache).
    """
    cache_manager = ProtocolsRegistry.get_instance(ICacheManager)
    if cache_manager is None:
        logger.debug("No cache manager registered; publish history not recorded.")
        return 0

    cache = cache_manager.get_cache(HISTORY_CACHE_NAME)
    if cache is None:
        return 0
    iso = shown_on.isoformat()
    count = 0
    for asset_id in asset_ids:
        cache.put(asset_id, iso)
        count += 1
    return count


def get_last_shown(asset_id: str) -> Optional[str]:
    """Return the recorded ISO ``last_shown_date`` for ``asset_id``, if any.

    Returns ``None`` when no cache manager is registered or the asset has
    no recorded history. The returned value is the raw stored string;
    callers parse it via ``avoid_repeat._parse_last_shown`` which already
    handles ISO date/datetime strings.
    """
    cache_manager = ProtocolsRegistry.get_instance(ICacheManager)
    if cache_manager is None:
        return None
    cache = cache_manager.get_cache(HISTORY_CACHE_NAME)
    if cache is None:
        return None
    value = cache.get(asset_id)
    return value if isinstance(value, str) else None
