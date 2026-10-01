"""Tests for smart_albums.nodes.partition.phash."""

from __future__ import annotations

from datetime import datetime

import pytest

from smart_albums.nodes.partition.phash import (
    PartitionPHash,
    _hamming_distance,
    _union_find_phash,
)

from conftest import make_asset, make_context


class TestHammingDistance:
    """Tests for _hamming_distance helper."""

    def test_identical_hashes(self):
        assert _hamming_distance("0000000000000000", "0000000000000000") == 0

    def test_single_bit_difference(self):
        # 0x0000000000000001 vs 0x0000000000000000 = 1 bit
        assert _hamming_distance("0000000000000001", "0000000000000000") == 1

    def test_all_bits_different(self):
        assert _hamming_distance("0000000000000000", "ffffffffffffffff") == 64

    def test_known_distance(self):
        # 0xF = 1111, 0x0 = 0000 → 4 bits in last nibble
        assert _hamming_distance("000000000000000f", "0000000000000000") == 4


class TestUnionFindPHash:
    """Tests for _union_find_phash grouping."""

    def test_no_assets(self):
        groups = _union_find_phash([], threshold=0.90)
        assert groups == {}

    def test_single_asset(self):
        a = make_asset(id="a1", metadata={"phash": "0000000000000000"})
        groups = _union_find_phash([a], threshold=0.90)
        assert len(groups) == 1

    def test_identical_hashes_grouped(self):
        a1 = make_asset(id="a1", metadata={"phash": "abcdef1234567890"})
        a2 = make_asset(id="a2", metadata={"phash": "abcdef1234567890"})
        # threshold=0.90 → max_distance = int(0.10 * 64) = 6
        groups = _union_find_phash([a1, a2], threshold=0.90)
        assert len(groups) == 1
        group = list(groups.values())[0]
        assert len(group) == 2

    def test_distant_hashes_separate(self):
        a1 = make_asset(id="a1", metadata={"phash": "0000000000000000"})
        a2 = make_asset(id="a2", metadata={"phash": "ffffffffffffffff"})
        # distance=64, way above max_distance=6
        groups = _union_find_phash([a1, a2], threshold=0.90)
        assert len(groups) == 2

    def test_transitive_grouping_when_all_pairs_match(self):
        # threshold=0.90 → max_distance = int(0.10 * 64) = 6
        # All three pairs are mutually within max_distance — grouping all
        # three together is correct here since every pair genuinely
        # qualifies, including against the anchor.
        a1 = make_asset(id="a1", metadata={"phash": "0000000000000000"})
        a2 = make_asset(id="a2", metadata={"phash": "0000000000000001"})  # dist 1 from a1
        a3 = make_asset(id="a3", metadata={"phash": "0000000000000003"})  # dist 2 from a1, dist 1 from a2
        groups = _union_find_phash([a1, a2, a3], threshold=0.90)
        assert len(groups) == 1

    def test_anchor_bound_prevents_unbounded_chain(self):
        # threshold=0.90 → max_distance = int(0.10 * 64) = 6
        # a1-a2 dist=4 (within threshold), a2-a3 dist=4 (within threshold),
        # but a1-a3 dist=8 (NOT within threshold). Plain single-linkage
        # union-find would chain a1-a2-a3 into one group purely because
        # each is close to its immediate neighbor, even though a1 and a3
        # are not actually similar — this is exactly the "long chain
        # collapses into one giant group" failure mode described in
        # ND-10. The anchor-bounded union-find requires a3 to also match
        # the group's anchor (a1) before joining, so it must form its own
        # separate group instead.
        a1 = make_asset(id="a1", metadata={"phash": "0000000000000000"})
        a2 = make_asset(id="a2", metadata={"phash": "000000000000000f"})  # dist 4 from a1
        a3 = make_asset(id="a3", metadata={"phash": "00000000000000ff"})  # dist 4 from a2, 8 from a1
        groups = _union_find_phash([a1, a2, a3], threshold=0.90)
        assert len(groups) == 2
        sizes = sorted(len(g) for g in groups.values())
        assert sizes == [1, 2]

    def test_threshold_boundary(self):
        # threshold=0.90 → max_distance = 6
        # distance exactly 6 should still group (≤)
        # 0x3f = 0b00111111 = 6 bits set
        a1 = make_asset(id="a1", metadata={"phash": "0000000000000000"})
        a2 = make_asset(id="a2", metadata={"phash": "000000000000003f"})  # dist 6
        groups = _union_find_phash([a1, a2], threshold=0.90)
        assert len(groups) == 1

    def test_above_threshold_boundary(self):
        # threshold=0.90 → max_distance = 6
        # distance 7 should NOT group
        # 0x7f = 0b01111111 = 7 bits set
        a1 = make_asset(id="a1", metadata={"phash": "0000000000000000"})
        a2 = make_asset(id="a2", metadata={"phash": "000000000000007f"})  # dist 7
        groups = _union_find_phash([a1, a2], threshold=0.90)
        assert len(groups) == 2


class TestPartitionPHashStage:
    """Tests for the PartitionPHash stage."""

    @pytest.mark.asyncio
    async def test_empty_assets(self):
        ctx = make_context(assets=[])
        node = PartitionPHash({"threshold": 0.90})
        result = await node.run(ctx)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_single_asset_unchanged(self):
        a = make_asset(id="a1", metadata={"phash": "abcdef1234567890"})
        ctx = make_context(assets=[a])
        node = PartitionPHash({"threshold": 0.90})
        result = await node.run(ctx)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_duplicates_grouped(self):
        a1 = make_asset(id="a1", metadata={"phash": "0000000000000000"})
        a2 = make_asset(id="a2", metadata={"phash": "0000000000000001"})  # dist 1
        a3 = make_asset(id="a3", metadata={"phash": "ffffffffffffffff"})  # far
        ctx = make_context(assets=[a1, a2, a3])
        node = PartitionPHash({"threshold": 0.90})
        result = await node.run(ctx)
        # a1 and a2 in one group, a3 alone
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_assets_without_phash_get_own_partition(self):
        a1 = make_asset(id="a1", metadata={"phash": "0000000000000000"})
        a2 = make_asset(id="a2", metadata={"phash": "0000000000000001"})
        a3 = make_asset(id="no-phash", metadata={})
        ctx = make_context(assets=[a1, a2, a3])
        node = PartitionPHash({"threshold": 0.90})
        result = await node.run(ctx)
        # a1+a2 grouped, a3 individual
        assert len(result) == 2
        sizes = sorted(len(r.assets) for r in result)
        assert sizes == [1, 2]
