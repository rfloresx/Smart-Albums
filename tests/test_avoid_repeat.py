"""Tests for smart_albums.nodes.select.avoid_repeat."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from smart_albums.nodes.select.avoid_repeat import AvoidRepeat

from conftest import make_asset, make_context


class TestAvoidRepeat:
    """Tests for the AvoidRepeat stage."""

    @pytest.mark.asyncio
    async def test_zero_days_keeps_all(self):
        assets = [
            make_asset(id="a1", metadata={"last_shown_date": date(2024, 6, 14)}),
            make_asset(id="a2", metadata={"last_shown_date": date(2024, 6, 10)}),
        ]
        ctx = make_context(assets=assets)
        node = AvoidRepeat({"days": 0, "photo_count": 500, "run_date": date(2024, 6, 15)})
        result = await node.run(ctx)
        assert len(result[0].assets) == 2
        assert result[0].stats["avoid_repeat.fresh_count"] == 2
        assert result[0].stats["avoid_repeat.backfilled_count"] == 0

    @pytest.mark.asyncio
    async def test_fresh_assets_kept(self):
        run_date = date(2024, 6, 15)
        cutoff = run_date - timedelta(days=7)
        assets = [
            # Never shown — fresh
            make_asset(id="a1"),
            # Shown before cutoff — fresh
            make_asset(id="a2", metadata={"last_shown_date": cutoff - timedelta(days=1)}),
            # Shown exactly at cutoff — fresh (<=)
            make_asset(id="a3", metadata={"last_shown_date": cutoff}),
        ]
        ctx = make_context(assets=assets)
        node = AvoidRepeat({"days": 7, "photo_count": 500, "run_date": run_date})
        result = await node.run(ctx)
        assert len(result[0].assets) == 3
        assert result[0].stats["avoid_repeat.fresh_count"] == 3

    @pytest.mark.asyncio
    async def test_recently_shown_excluded(self):
        run_date = date(2024, 6, 15)
        # Generate enough fresh assets to exceed photo_count so no backfill occurs
        fresh_assets = [make_asset(id=f"fresh{i}") for i in range(5)]
        recent_asset = make_asset(id="recent", metadata={"last_shown_date": date(2024, 6, 14)})
        assets = fresh_assets + [recent_asset]
        ctx = make_context(assets=assets)
        # photo_count=5 means we need 5; we have 5 fresh, so no backfill
        node = AvoidRepeat({"days": 7, "photo_count": 5, "run_date": run_date})
        result = await node.run(ctx)
        assert len(result[0].assets) == 5
        result_ids = [a.id for a in result[0].assets]
        assert "recent" not in result_ids
        assert result[0].stats["avoid_repeat.fresh_count"] == 5
        assert result[0].stats["avoid_repeat.backfilled_count"] == 0

    @pytest.mark.asyncio
    async def test_backfill_oldest_shown_first(self):
        run_date = date(2024, 6, 15)
        # All ineligible (shown within last 7 days), but we need 5 photos
        assets = [
            make_asset(id="shown_yesterday", metadata={"last_shown_date": date(2024, 6, 14)}),
            make_asset(id="shown_3_days_ago", metadata={"last_shown_date": date(2024, 6, 12)}),
            make_asset(id="shown_2_days_ago", metadata={"last_shown_date": date(2024, 6, 13)}),
        ]
        ctx = make_context(assets=assets)
        node = AvoidRepeat({"days": 7, "photo_count": 5, "run_date": run_date})
        result = await node.run(ctx)
        # All 3 should be backfilled (deficit = 5, only 3 available)
        assert len(result[0].assets) == 3
        assert result[0].stats["avoid_repeat.fresh_count"] == 0
        assert result[0].stats["avoid_repeat.backfilled_count"] == 3
        # Check backfill flag
        for asset in result[0].assets:
            assert asset.metadata.get("fallback_backfilled") is True

    @pytest.mark.asyncio
    async def test_backfill_respects_deficit(self):
        run_date = date(2024, 6, 15)
        assets = [
            make_asset(id="fresh1"),
            make_asset(id="fresh2"),
            make_asset(id="ineligible1", metadata={"last_shown_date": date(2024, 6, 14)}),
            make_asset(id="ineligible2", metadata={"last_shown_date": date(2024, 6, 13)}),
            make_asset(id="ineligible3", metadata={"last_shown_date": date(2024, 6, 12)}),
        ]
        ctx = make_context(assets=assets)
        # Need 4, have 2 fresh → backfill 2 from ineligible (oldest first)
        node = AvoidRepeat({"days": 7, "photo_count": 4, "run_date": run_date})
        result = await node.run(ctx)
        assert len(result[0].assets) == 4
        assert result[0].stats["avoid_repeat.fresh_count"] == 2
        assert result[0].stats["avoid_repeat.backfilled_count"] == 2
        ids = [a.id for a in result[0].assets]
        assert "fresh1" in ids
        assert "fresh2" in ids
        # The two oldest-shown should be backfilled
        assert "ineligible3" in ids  # shown June 12 (oldest)
        assert "ineligible2" in ids  # shown June 13

    @pytest.mark.asyncio
    async def test_no_backfill_when_fresh_sufficient(self):
        run_date = date(2024, 6, 15)
        assets = [
            make_asset(id=f"fresh{i}") for i in range(10)
        ]
        ctx = make_context(assets=assets)
        node = AvoidRepeat({"days": 7, "photo_count": 5, "run_date": run_date})
        result = await node.run(ctx)
        assert len(result[0].assets) == 10
        assert result[0].stats["avoid_repeat.backfilled_count"] == 0

    @pytest.mark.asyncio
    async def test_empty_assets(self):
        ctx = make_context(assets=[])
        node = AvoidRepeat({"days": 7, "photo_count": 500, "run_date": date(2024, 6, 15)})
        result = await node.run(ctx)
        assert len(result[0].assets) == 0
        assert result[0].stats["avoid_repeat.fresh_count"] == 0
        assert result[0].stats["avoid_repeat.backfilled_count"] == 0
