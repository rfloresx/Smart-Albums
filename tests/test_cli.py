"""Tests for smart_albums.cli.app — config loading and logging configuration."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest
import typer

from smart_albums.cli.app import _load_json_file, _configure_logging
from smart_albums.cli.config import load_config_file


class TestLoadJsonFile:
    """Tests for _load_json_file helper."""

    def test_valid_json(self, tmp_path: Path):
        f = tmp_path / "config.json"
        f.write_text('{"year": 2024}')
        result = _load_json_file(f)
        assert result == {"year": 2024}

    def test_missing_file_exits(self, tmp_path: Path):
        with pytest.raises(typer.Exit):
            _load_json_file(tmp_path / "nonexistent.json")

    def test_invalid_json_exits(self, tmp_path: Path):
        f = tmp_path / "bad.json"
        f.write_text("{not valid json")
        with pytest.raises(typer.Exit):
            _load_json_file(f)

    def test_empty_object(self, tmp_path: Path):
        f = tmp_path / "empty.json"
        f.write_text("{}")
        result = _load_json_file(f)
        assert result == {}

    def test_nested_config(self, tmp_path: Path):
        config = {
            "pipeline_settings": {
                "score": {"concurrency": 8},
                "balanced": {"min_photos": 100, "max_photos": 300},
            }
        }
        f = tmp_path / "config.json"
        f.write_text(json.dumps(config))
        result = _load_json_file(f)
        assert result["pipeline_settings"]["score"]["concurrency"] == 8


class TestLoadConfigFileYaml:
    """Tests for load_config_file's YAML support (CO-01).

    The Dockerfile CMD and README both invoke
    ``smart-albums rotation --config /config/rotation.yaml``, but the
    loader used to only ever parse JSON, so every documented Docker
    invocation failed with a JSON decode error before this fix.
    """

    def test_yaml_file_is_parsed_as_yaml(self, tmp_path: Path):
        f = tmp_path / "rotation.yaml"
        f.write_text(
            "log_level: INFO\n"
            "image: immich\n"
            "image.immich.base_url: http://localhost:2283/api\n"
            "pipeline_settings:\n"
            "  score:\n"
            "    concurrency: 8\n"
        )
        result = load_config_file(f)
        assert result["log_level"] == "INFO"
        assert result["image"] == "immich"
        assert result["pipeline_settings"]["score"]["concurrency"] == 8

    def test_yml_extension_also_parsed_as_yaml(self, tmp_path: Path):
        f = tmp_path / "config.yml"
        f.write_text("year: 2024\n")
        result = load_config_file(f)
        assert result == {"year": 2024}

    def test_json_file_still_parsed_as_json(self, tmp_path: Path):
        f = tmp_path / "config.json"
        f.write_text(json.dumps({"year": 2024}))
        result = load_config_file(f)
        assert result == {"year": 2024}

    def test_invalid_yaml_exits(self, tmp_path: Path):
        f = tmp_path / "bad.yaml"
        f.write_text("key: [unterminated\n")
        with pytest.raises(typer.Exit):
            load_config_file(f)

    def test_empty_yaml_returns_empty_dict(self, tmp_path: Path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        result = load_config_file(f)
        assert result == {}

    def test_yaml_list_at_top_level_exits(self, tmp_path: Path):
        f = tmp_path / "list.yaml"
        f.write_text("- one\n- two\n")
        with pytest.raises(typer.Exit):
            load_config_file(f)

    def test_missing_yaml_file_exits(self, tmp_path: Path):
        with pytest.raises(typer.Exit):
            load_config_file(tmp_path / "nonexistent.yaml")


class TestConfigureLogging:
    """Tests for _configure_logging helper."""

    def test_sets_debug_level(self):
        _configure_logging("DEBUG")
        assert logging.getLogger().level == logging.DEBUG
        # Reset
        _configure_logging("WARNING")

    def test_sets_info_level(self):
        _configure_logging("INFO")
        assert logging.getLogger().level == logging.INFO
        _configure_logging("WARNING")

    def test_with_log_file(self, tmp_path: Path):
        log_file = tmp_path / "test.log"
        _configure_logging("INFO", log_file)
        # The handler should be created (file may not have content yet)
        root_logger = logging.getLogger()
        file_handlers = [
            h for h in root_logger.handlers
            if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) >= 1
        _configure_logging("WARNING")

    def test_case_insensitive(self):
        _configure_logging("debug")
        assert logging.getLogger().level == logging.DEBUG
        _configure_logging("WARNING")
