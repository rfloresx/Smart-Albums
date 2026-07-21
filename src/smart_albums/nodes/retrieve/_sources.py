"""Source providers for the rotation 'source' stage.

Each provider resolves the *scope* dimension of a rotation recipe — which
assets are candidates for downstream stages. Providers are constructed via
the :func:`build_source` factory from a plain config dict and expose a single
``fetch`` coroutine that retrieves matching assets from the image client.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any, Protocol, runtime_checkable

from smart_albums.core.models import Asset
from smart_albums.core.protocols import IImageClient


@runtime_checkable
class SourceProvider(Protocol):
    """Resolves the scope dimension of a rotation recipe."""

    async def fetch(self, client: IImageClient, run_date: date) -> list[Asset]: ...


class LibrarySource:
    """Library-wide source: fetches all assets using a very wide date range."""

    async def fetch(self, client: IImageClient, run_date: date) -> list[Asset]:
        """Retrieve all assets via a wide date-range search (1970 to 2100)."""
        taken_after = datetime(1970, 1, 1, 0, 0, 0)
        taken_before = datetime(2100, 12, 31, 23, 59, 59)
        return await client.search_assets(taken_after, taken_before)


class YearSource:
    """Calendar-year source: assets captured within a single year."""

    def __init__(self, year: int) -> None:
        self.year = year

    async def fetch(self, client: IImageClient, run_date: date) -> list[Asset]:
        """Retrieve assets captured within the configured calendar year."""
        taken_after = datetime(self.year, 1, 1, 0, 0, 0)
        taken_before = datetime(self.year, 12, 31, 23, 59, 59)
        return await client.search_assets(taken_after, taken_before)


class DateRangeSource:
    """Explicit date-range source: assets within a from/to date window."""

    def __init__(self, from_date: date, to_date: date) -> None:
        self.from_date = from_date
        self.to_date = to_date

    async def fetch(self, client: IImageClient, run_date: date) -> list[Asset]:
        """Retrieve assets captured within the configured date range."""
        taken_after = datetime.combine(self.from_date, time.min)
        taken_before = datetime.combine(self.to_date, time.max)
        return await client.search_assets(taken_after, taken_before)


class RecentSource:
    """Rolling recent-window source relative to the run date.

    Computes the window start as ``run_date - timedelta(...)`` using:
    - days directly
    - weeks * 7 for week-based windows
    - months * 30 for month-based windows
    """

    def __init__(
        self,
        days: int | None = None,
        weeks: int | None = None,
        months: int | None = None,
    ) -> None:
        provided = [v for v in (days, weeks, months) if v is not None]
        if len(provided) != 1:
            raise ValueError(
                "RecentSource requires exactly one of 'days', 'weeks', or 'months'"
            )
        self.days = days
        self.weeks = weeks
        self.months = months

    def _start_date(self, run_date: date) -> date:
        """Compute the inclusive window start date."""
        if self.days is not None:
            return run_date - timedelta(days=self.days)
        if self.weeks is not None:
            return run_date - timedelta(weeks=self.weeks)
        # months: approximate as months * 30
        return run_date - timedelta(days=(self.months or 0) * 30)

    async def fetch(self, client: IImageClient, run_date: date) -> list[Asset]:
        """Retrieve assets captured within the rolling recent window."""
        taken_after = datetime.combine(self._start_date(run_date), time.min)
        taken_before = datetime.combine(run_date, time.max)
        return await client.search_assets(taken_after, taken_before)


class AlbumSource:
    """Album-membership source: fetch assets from a specific album."""

    def __init__(self, album_id: str) -> None:
        self.album_id = album_id

    async def fetch(self, client: IImageClient, run_date: date) -> list[Asset]:
        """Retrieve all assets contained in the configured album."""
        return await client.list_album_assets(self.album_id)


def build_source(spec: dict[str, Any]) -> SourceProvider:
    """Construct a source provider from a config dict.

    Args:
        spec: A dictionary with a ``"type"`` key and type-specific parameters.
              Examples::

                  {"type": "library"}
                  {"type": "year", "year": 2024}
                  {"type": "date_range", "from": date(2024, 1, 1), "to": date(2024, 6, 30)}
                  {"type": "recent", "days": 30}
                  {"type": "album", "album_id": "abc-123"}

    Returns:
        The concrete :class:`SourceProvider` matching the spec type.

    Raises:
        ValueError: If the type is unrecognised or required parameters are missing.
    """
    source_type = spec.get("type")

    if source_type == "library":
        return LibrarySource()

    if source_type == "year":
        year = spec.get("year")
        if year is None:
            raise ValueError("source type 'year' requires a 'year' value")
        return YearSource(year)

    if source_type == "date_range":
        from_date = spec.get("from")
        to_date = spec.get("to")
        if from_date is None or to_date is None:
            raise ValueError("source type 'date_range' requires 'from' and 'to'")
        return DateRangeSource(from_date, to_date)

    if source_type == "recent":
        return RecentSource(
            days=spec.get("days"),
            weeks=spec.get("weeks"),
            months=spec.get("months"),
        )

    if source_type == "album":
        album_id = spec.get("album_id")
        if album_id is None:
            raise ValueError("source type 'album' requires an 'album_id' value")
        return AlbumSource(album_id)

    raise ValueError(f"unsupported source type: {source_type!r}")
