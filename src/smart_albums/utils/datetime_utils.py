"""Shared datetime normalization helpers.

Asset ``captured_at`` values can arrive either timezone-aware or naive
depending on the source client (the Immich and local clients disagree —
see CL-12). Mixing the two in a single comparison or subtraction raises
``TypeError: can't compare offset-naive and offset-aware datetimes``,
which crashed the time-based partitioners whenever a batch happened to
contain both (ND-14).

These helpers coerce everything to a single convention (timezone-aware,
UTC) so sorting, gap computation, and range comparisons are always
well-defined regardless of what the source provided.
"""

from __future__ import annotations

from datetime import datetime, timezone


def to_aware_utc(dt: datetime | None) -> datetime:
    """Return ``dt`` as a timezone-aware UTC datetime.

    A naive datetime is assumed to already be in UTC (the convention the
    retrieve layer uses for its query bounds) and simply tagged with UTC.
    An aware datetime is converted to UTC.

    ``None`` maps to ``datetime.min`` (tagged UTC) so callers that have
    already partitioned out untimed assets can still sort/compare a
    possibly-``None``-typed field without a separate guard; a real pipeline
    only ever passes non-``None`` values here (untimed assets are filtered
    upstream), so this is just a type-level convenience.
    """
    if dt is None:
        return datetime.min.replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
