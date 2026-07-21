"""Tests for smart_albums.core.context."""

from __future__ import annotations

from datetime import datetime

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.models import Asset

from conftest import make_asset, make_context


class TestPipelineContext:
    """Tests for PipelineContext creation and defaults."""

    def test_default_fields(self):
        ctx = PipelineContext()
        assert ctx.config == {}
        assert ctx.assets == []
        assert ctx.stats == {}
        assert ctx.metadata == {}
        assert ctx.image_client is None
        assert ctx.llm_client is None
        assert ctx.embedding_client is None
        assert ctx.progress is None
        assert ctx.partition_name == ""
        assert ctx.partition_id == ""
        assert ctx.parent_partition_id is None
        assert ctx.partition_depth == 0

    def test_with_assets(self):
        assets = [make_asset(id="a1"), make_asset(id="a2")]
        ctx = make_context(assets=assets)
        assert len(ctx.assets) == 2
        assert ctx.assets[0].id == "a1"

    def test_with_config(self):
        ctx = make_context(config={"year": 2024, "threshold": 0.9})
        assert ctx.config["year"] == 2024
        assert ctx.config["threshold"] == 0.9

    def test_stats_mutable(self):
        ctx = make_context()
        ctx.stats["count"] = 42
        assert ctx.stats["count"] == 42

    def test_partition_tracking(self):
        ctx = PipelineContext(
            partition_name="time",
            partition_id="time_0",
            parent_partition_id="root",
            partition_depth=1,
        )
        assert ctx.partition_name == "time"
        assert ctx.partition_id == "time_0"
        assert ctx.parent_partition_id == "root"
        assert ctx.partition_depth == 1


class TestContextBatch:
    """Tests for ContextBatch type alias."""

    def test_is_list_of_contexts(self):
        batch: ContextBatch = [make_context(), make_context()]
        assert len(batch) == 2
        assert all(isinstance(c, PipelineContext) for c in batch)
