"""Tests for smart_albums.nodes.dedup.phash."""

from __future__ import annotations

from datetime import datetime

import pytest

from smart_albums.nodes.dedup.phash import (
    DedupPHash,
    _hamming_distance,
    _find_duplicate_groups,
)

from conftest import make_asset, make_scored_asset, make_context


class TestFindDuplicateGroups:
    """Tests for _find_duplicate_groups."""

    def test_no_duplicates(self):
        assets = [
            make_asset(id="a1", captured_at=datetime(2024, 1, 1), metadata={"phash": "0000000000000000"}),
            make_asset(id="a2", captured_at=datetime(2024, 1, 1), metadata={"phash": "ffffffffffffffff"}),
        ]
        groups = _find_duplicate_groups(assets, max_distance=5)
        assert groups == []

    def test_exact_duplicates_same_day(self):
        assets = [
            make_asset(id="a1", captured_at=datetime(2024, 1, 1), metadata={"phash": "abcdef1234567890"}),
            make_asset(id="a2", captured_at=datetime(2024, 1, 1), metadata={"phash": "abcdef1234567890"}),
        ]
        groups = _find_duplicate_groups(assets, max_distance=5)
        assert len(groups) == 1
        assert len(groups[0]) == 2

    def test_near_duplicates_same_day(self):
        assets = [
            make_asset(id="a1", captured_at=datetime(2024, 1, 1), metadata={"phash": "0000000000000000"}),
            make_asset(id="a2", captured_at=datetime(2024, 1, 1), metadata={"phash": "0000000000000003"}),  # dist 2
        ]
        groups = _find_duplicate_groups(assets, max_distance=5)
        assert len(groups) == 1

    def test_different_days_not_grouped(self):
        assets = [
            make_asset(id="a1", captured_at=datetime(2024, 1, 1), metadata={"phash": "abcdef1234567890"}),
            make_asset(id="a2", captured_at=datetime(2024, 1, 2), metadata={"phash": "abcdef1234567890"}),
        ]
        groups = _find_duplicate_groups(assets, max_distance=5)
        # Different days → not compared → no groups
        assert groups == []

    def test_assets_without_phash_ignored(self):
        assets = [
            make_asset(id="a1", captured_at=datetime(2024, 1, 1), metadata={"phash": "abcdef1234567890"}),
            make_asset(id="a2", captured_at=datetime(2024, 1, 1), metadata={}),
        ]
        groups = _find_duplicate_groups(assets, max_distance=5)
        assert groups == []


class TestDedupPHashStage:
    """Tests for the DedupPHash stage."""

    @pytest.mark.asyncio
    async def test_no_duplicates_preserves_all(self):
        assets = [
            make_scored_asset(id="a1", score=0.5, captured_at=datetime(2024, 1, 1), metadata={"score": 0.5, "phash": "0000000000000000"}),
            make_scored_asset(id="a2", score=0.7, captured_at=datetime(2024, 1, 1), metadata={"score": 0.7, "phash": "ffffffffffffffff"}),
        ]
        ctx = make_context(assets=assets)
        node = DedupPHash({"threshold": 0.90})
        result = await node.run(ctx)
        assert len(result[0].assets) == 2
        assert result[0].stats["dedup.phash.excluded"] == 0

    @pytest.mark.asyncio
    async def test_duplicate_keeps_highest_score(self):
        assets = [
            make_asset(id="low", captured_at=datetime(2024, 1, 1), metadata={"score": 0.3, "phash": "0000000000000000"}),
            make_asset(id="high", captured_at=datetime(2024, 1, 1), metadata={"score": 0.9, "phash": "0000000000000001"}),
        ]
        ctx = make_context(assets=assets)
        node = DedupPHash({"threshold": 0.90})
        result = await node.run(ctx)
        assert len(result[0].assets) == 1
        assert result[0].assets[0].id == "high"
        assert result[0].stats["dedup.phash.excluded"] == 1

    @pytest.mark.asyncio
    async def test_tie_break_by_smallest_id(self):
        assets = [
            make_asset(id="zzz", captured_at=datetime(2024, 1, 1), metadata={"score": 0.5, "phash": "0000000000000000"}),
            make_asset(id="aaa", captured_at=datetime(2024, 1, 1), metadata={"score": 0.5, "phash": "0000000000000000"}),
        ]
        ctx = make_context(assets=assets)
        node = DedupPHash({"threshold": 0.90})
        result = await node.run(ctx)
        assert len(result[0].assets) == 1
        assert result[0].assets[0].id == "aaa"

    @pytest.mark.asyncio
    async def test_empty_assets(self):
        ctx = make_context(assets=[])
        node = DedupPHash({"threshold": 0.90})
        result = await node.run(ctx)
        assert len(result[0].assets) == 0
        assert result[0].stats["dedup.phash.excluded"] == 0
