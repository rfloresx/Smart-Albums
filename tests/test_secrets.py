"""Tests for the ``_FILE``-aware environment secret reader."""

from __future__ import annotations

import pytest

from smart_albums.utils.secrets import read_env_secret


def test_plain_env_var(monkeypatch):
    monkeypatch.setenv("MY_SECRET", "plain-value")
    monkeypatch.delenv("MY_SECRET_FILE", raising=False)
    assert read_env_secret("MY_SECRET") == "plain-value"


def test_default_when_unset(monkeypatch):
    monkeypatch.delenv("MY_SECRET", raising=False)
    monkeypatch.delenv("MY_SECRET_FILE", raising=False)
    assert read_env_secret("MY_SECRET", "fallback") == "fallback"
    assert read_env_secret("MY_SECRET") == ""


def test_file_variant_wins(monkeypatch, tmp_path):
    secret_file = tmp_path / "secret.txt"
    secret_file.write_text("from-file\n", encoding="utf-8")
    monkeypatch.setenv("MY_SECRET", "plain-value")
    monkeypatch.setenv("MY_SECRET_FILE", str(secret_file))
    # The _FILE variant takes precedence and is stripped of trailing newline.
    assert read_env_secret("MY_SECRET") == "from-file"


def test_missing_file_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("MY_SECRET_FILE", str(tmp_path / "does-not-exist"))
    with pytest.raises(OSError):
        read_env_secret("MY_SECRET")
