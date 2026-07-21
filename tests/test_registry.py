"""Tests for smart_albums.core.registry."""

from __future__ import annotations

from smart_albums.core.registry import (
    STAGE_REGISTRY,
    has_stage,
    list_stages,
    get_stage,
    get_stage_config,
    get_stage_description,
)


class TestRegistry:
    """Tests for stage registry lookup functions."""

    def test_has_stage_known(self):
        # select.best is registered by the import side-effect
        assert has_stage("select.best")

    def test_has_stage_unknown(self):
        assert not has_stage("nonexistent.stage.xyz")

    def test_list_stages_returns_sorted(self):
        stages = list_stages()
        assert stages == sorted(stages)
        assert len(stages) > 0

    def test_list_stages_includes_known_stages(self):
        stages = list_stages()
        assert "select.best" in stages
        assert "filter.non_images" in stages
        assert "merge.concat" in stages

    def test_get_stage_returns_class(self):
        cls = get_stage("select.best")
        assert cls is not None
        assert cls._stage_name == "select.best"

    def test_get_stage_unknown_returns_none(self):
        assert get_stage("does_not_exist") is None

    def test_get_stage_config(self):
        config = get_stage_config("filter.min_score")
        assert len(config) > 0
        keys = [p.key for p in config]
        assert "threshold" in keys

    def test_get_stage_config_unknown_returns_empty(self):
        assert get_stage_config("nonexistent") == ()

    def test_get_stage_description(self):
        desc = get_stage_description("select.best")
        assert "best" in desc.lower()

    def test_get_stage_description_unknown_returns_empty(self):
        assert get_stage_description("nonexistent") == ""
