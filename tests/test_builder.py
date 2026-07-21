"""Tests for smart_albums.core.builder."""

from __future__ import annotations

import pytest

from smart_albums.core.builder import normalize_step, build_pipeline
from smart_albums.core.spec import StepSpec
from smart_albums.core.registry import get_stage


class TestNormalizeStep:
    """Tests for normalize_step() handling various input types."""

    def test_string_stage_name(self):
        spec = normalize_step("select.best")
        assert isinstance(spec, StepSpec)
        assert spec.name == "select.best"
        assert spec.config == {}

    def test_string_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown stage"):
            normalize_step("totally.fake.stage")

    def test_class_input(self):
        cls = get_stage("select.best")
        spec = normalize_step(cls)
        assert spec.name == "select.best"

    def test_tuple_input(self):
        spec = normalize_step(("select.best", {"foo": "bar"}))
        assert spec.name == "select.best"
        assert spec.config == {"foo": "bar"}

    def test_tuple_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown stage"):
            normalize_step(("fake.stage", {}))

    def test_step_spec_passthrough(self):
        original = StepSpec(name="select.best", config={"x": 1})
        result = normalize_step(original)
        assert result is original

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="Invalid pipeline step"):
            normalize_step(12345)


class TestBuildPipeline:
    """Tests for build_pipeline() producing a list of StepSpecs."""

    def test_simple_pipeline(self):
        pipeline = ["select.best", "merge.concat"]
        specs = build_pipeline(pipeline)
        assert len(specs) == 2
        assert specs[0].name == "select.best"
        assert specs[1].name == "merge.concat"

    def test_mixed_input_types(self):
        cls = get_stage("select.best")
        pipeline = [
            cls,
            "merge.concat",
            StepSpec(name="filter.non_images"),
        ]
        specs = build_pipeline(pipeline)
        assert len(specs) == 3
        assert specs[0].name == "select.best"
        assert specs[1].name == "merge.concat"
        assert specs[2].name == "filter.non_images"

    def test_empty_pipeline(self):
        specs = build_pipeline([])
        assert specs == []
