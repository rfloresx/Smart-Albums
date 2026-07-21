"""Tests for smart_albums.nodes.retrieve._sources."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from smart_albums.nodes.retrieve._sources import (
    build_source,
    LibrarySource,
    YearSource,
    DateRangeSource,
    RecentSource,
    AlbumSource,
)

from conftest import make_asset, FakeImageClient


class TestBuildSource:
    """Tests for the build_source factory function."""

    def test_library_type(self):
        provider = build_source({"type": "library"})
        assert isinstance(provider, LibrarySource)

    def test_year_type(self):
        provider = build_source({"type": "year", "year": 2024})
        assert isinstance(provider, YearSource)
        assert provider.year == 2024

    def test_year_missing_raises(self):
        with pytest.raises(ValueError, match="requires a 'year'"):
            build_source({"type": "year"})

    def test_date_range_type(self):
        provider = build_source({
            "type": "date_range",
            "from": date(2024, 1, 1),
            "to": date(2024, 6, 30),
        })
        assert isinstance(provider, DateRangeSource)

    def test_date_range_missing_raises(self):
        with pytest.raises(ValueError, match="requires 'from' and 'to'"):
            build_source({"type": "date_range", "from": date(2024, 1, 1)})

    def test_recent_days(self):
        provider = build_source({"type": "recent", "days": 30})
        assert isinstance(provider, RecentSource)
        assert provider.days == 30

    def test_recent_no_dimension_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            build_source({"type": "recent"})

    def test_recent_multiple_dimensions_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            build_source({"type": "recent", "days": 7, "weeks": 2})

    def test_album_type(self):
        provider = build_source({"type": "album", "album_id": "abc-123"})
        assert isinstance(provider, AlbumSource)
        assert provider.album_id == "abc-123"

    def test_album_missing_raises(self):
        with pytest.raises(ValueError, match="requires an 'album_id'"):
            build_source({"type": "album"})

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="unsupported source type"):
            build_source({"type": "fantasy"})


class TestYearSource:
    """Tests for YearSource.fetch."""

    @pytest.mark.asyncio
    async def test_fetches_year_range(self):
        assets = [
            make_asset(id="in-year", captured_at=datetime(2024, 6, 15)),
            make_asset(id="out-year", captured_at=datetime(2023, 12, 31)),
        ]
        client = FakeImageClient(assets=assets)
        source = YearSource(2024)
        result = await source.fetch(client, date(2024, 7, 1))
        assert len(result) == 1
        assert result[0].id == "in-year"


class TestRecentSource:
    """Tests for RecentSource._start_date."""

    def test_days_start_date(self):
        source = RecentSource(days=7)
        run_date = date(2024, 6, 15)
        start = source._start_date(run_date)
        assert start == date(2024, 6, 8)

    def test_weeks_start_date(self):
        source = RecentSource(weeks=2)
        run_date = date(2024, 6, 15)
        start = source._start_date(run_date)
        assert start == date(2024, 6, 1)

    def test_months_start_date(self):
        source = RecentSource(months=1)
        run_date = date(2024, 6, 15)
        start = source._start_date(run_date)
        # 1 month ≈ 30 days
        assert start == date(2024, 5, 16)
