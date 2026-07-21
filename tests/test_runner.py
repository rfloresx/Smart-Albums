"""Tests for smart_albums.core.runner."""

from __future__ import annotations

import pytest

from smart_albums.core.runner import build_node, run_pipeline
from smart_albums.core.spec import StepSpec
from smart_albums.core.context import PipelineContext

from conftest import make_asset, make_context


class TestBuildNode:
    """Tests for build_node() validation and instantiation."""

    def test_known_stage(self):
        spec = StepSpec(name="select.best")
        node = build_node(spec)
        assert node.name == "select.best"

    def test_unknown_stage_raises(self):
        spec = StepSpec(name="does_not_exist")
        with pytest.raises(ValueError, match="Unknown stage"):
            build_node(spec)

    def test_applies_defaults(self):
        spec = StepSpec(name="filter.min_score")
        node = build_node(spec)
        # threshold has default 0.0
        assert node.config["threshold"] == 0.0

    def test_overrides_defaults_with_config(self):
        spec = StepSpec(name="filter.min_score", config={"threshold": 0.5})
        node = build_node(spec)
        assert node.config["threshold"] == 0.5

    def test_unknown_config_key_raises(self):
        spec = StepSpec(name="filter.min_score", config={"threshold": 0.5, "bogus_key": True})
        with pytest.raises(ValueError, match="Unknown config 'bogus_key'"):
            build_node(spec)


class TestRunPipeline:
    """Tests for run_pipeline() end-to-end execution."""

    @pytest.mark.asyncio
    async def test_single_stage_pipeline(self):
        """select.best on a context with scored assets picks the best."""
        from smart_albums.nodes.select.best import SelectBest

        assets = [
            make_asset(id="low", metadata={"score": 0.2}),
            make_asset(id="high", metadata={"score": 0.9}),
        ]
        ctx = make_context(assets=assets)
        result = await run_pipeline([SelectBest], [ctx])
        assert len(result) == 1
        assert len(result[0].assets) == 1
        assert result[0].assets[0].id == "high"

    @pytest.mark.asyncio
    async def test_filter_then_select(self):
        """Pipeline: filter non-images -> select best."""
        from smart_albums.nodes.filter.videos import RetainImages
        from smart_albums.nodes.select.best import SelectBest

        assets = [
            make_asset(id="img1", mime_type="image/jpeg", metadata={"score": 0.5}),
            make_asset(id="vid1", mime_type="video/mp4", metadata={"score": 0.9}),
            make_asset(id="img2", mime_type="image/png", metadata={"score": 0.8}),
        ]
        ctx = make_context(assets=assets)
        result = await run_pipeline([RetainImages, SelectBest], [ctx])
        assert len(result) == 1
        # Video excluded, img2 is best among images
        assert result[0].assets[0].id == "img2"
