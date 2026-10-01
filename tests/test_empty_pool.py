"""Tests for smart_albums.nodes.filter._empty_pool."""

from __future__ import annotations

import pytest

from smart_albums.nodes.filter._empty_pool import (
    handle_empty_pool,
    EmptyPoolError,
    EmptyBehavior,
)
from smart_albums.core.node import PipelineNode, ConfigParam
from smart_albums.core.context import PipelineContext

from conftest import make_asset, make_context


class _FakeNode:
    """Minimal node-like object for testing handle_empty_pool."""

    def __init__(self, config: dict):
        self.config = config

    def get(self, key, default=None):
        return self.config.get(key, default)


class TestHandleEmptyPool:
    """Tests for handle_empty_pool behavior."""

    def test_non_empty_candidates_truncated(self):
        ctx = make_context()
        node = _FakeNode({})
        candidates = [make_asset(id=f"a{i}") for i in range(10)]
        pre_filter = candidates.copy()
        result = handle_empty_pool(node, ctx, candidates, pre_filter, photo_count=5)
        assert len(result) == 5

    def test_non_empty_candidates_fewer_than_count(self):
        ctx = make_context()
        node = _FakeNode({})
        candidates = [make_asset(id="a1"), make_asset(id="a2")]
        result = handle_empty_pool(node, ctx, candidates, candidates, photo_count=10)
        assert len(result) == 2

    def test_empty_skip_returns_empty(self):
        ctx = make_context()
        node = _FakeNode({"on_empty_pool": EmptyBehavior.SKIP})
        result = handle_empty_pool(node, ctx, [], [make_asset()], photo_count=5)
        assert result == []
        assert ctx.stats["filter.skipped_empty_pool"] is True

    def test_empty_fail_raises(self):
        ctx = make_context()
        node = _FakeNode({"on_empty_pool": EmptyBehavior.FAIL, "recipe_name": "test_recipe"})
        with pytest.raises(EmptyPoolError, match="test_recipe"):
            handle_empty_pool(node, ctx, [], [make_asset()], photo_count=5)

    def test_empty_fallback_random_samples(self):
        ctx = make_context()
        pre_filter = [make_asset(id=f"a{i}") for i in range(20)]
        node = _FakeNode({"on_empty_pool": EmptyBehavior.FALLBACK_RANDOM, "rng_seed": 42})
        result = handle_empty_pool(node, ctx, [], pre_filter, photo_count=5)

        # Beyond the count (TS-03), assert the result is a genuine *sample*
        # of the pre-filter pool: the right size, drawn only from the pool,
        # with no duplicates, and — because a seed is set — not simply the
        # first N in input order (which would betray a head-slice rather
        # than a sample).
        assert len(result) == 5
        pool_ids = {a.id for a in pre_filter}
        result_ids = [a.id for a in result]
        assert all(rid in pool_ids for rid in result_ids), "sampled from the pool"
        assert len(set(result_ids)) == 5, "no duplicate picks"
        assert result_ids != [f"a{i}" for i in range(5)], (
            "a seeded sample should not coincide with the first 5 in order"
        )

    def test_empty_fallback_random_deterministic(self):
        ctx1 = make_context()
        ctx2 = make_context()
        pre_filter = [make_asset(id=f"a{i}") for i in range(20)]
        node = _FakeNode({"on_empty_pool": EmptyBehavior.FALLBACK_RANDOM, "rng_seed": 42})
        r1 = handle_empty_pool(node, ctx1, [], pre_filter, photo_count=5)
        r2 = handle_empty_pool(node, ctx2, [], pre_filter, photo_count=5)
        assert [a.id for a in r1] == [a.id for a in r2]

    def test_empty_fallback_random_empty_pre_filter(self):
        ctx = make_context()
        node = _FakeNode({"on_empty_pool": EmptyBehavior.FALLBACK_RANDOM})
        result = handle_empty_pool(node, ctx, [], [], photo_count=5)
        assert result == []
        assert ctx.stats["filter.skipped_empty_pool"] is True
