"""Configuration loader for the NiceGUI web interface.

Reads settings from the YAML config file (connection URLs, API keys).
The admin panel can update these settings at runtime.

Provider configuration is stored per-protocol with a selected provider
and provider-specific settings, mirroring the CLI's ProtocolsRegistry
resolution pattern.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel


class ProviderConfig(BaseModel):
    """Configuration for a single protocol slot.

    Stores the selected provider name and a flat dict of provider-specific
    parameters (matching the provider class __init__ signature).
    """

    selected: str = ""
    params: dict[str, Any] = {}


class AuthConfig(BaseModel):
    """Authentication settings."""

    enabled: bool = True


class AppConfig(BaseModel):
    """Top-level application configuration loaded from YAML.

    Provider configuration lives in the database (see ``webgui.state``);
    the ``providers`` dict here only holds whatever is present in
    ``config.yaml`` for a fresh deployment before any settings are saved
    through the admin UI.
    """

    providers: dict[str, ProviderConfig] = {}
    auth: AuthConfig = AuthConfig()
    cache_dir: str = "/data/cache"
    prompts_dir: str = "/data/prompts"
    exports_dir: str = "/data/exports"
    db_path: str = "/data/db/smart_albums.db"
    log_file: Optional[str] = None

    def get_provider(self, protocol_name: str) -> ProviderConfig:
        """Get provider config for a protocol, returning empty default if missing."""
        return self.providers.get(protocol_name, ProviderConfig())


def _resolve_data_dir(configured_path: str, fallback_name: str) -> str:
    """Resolve a data directory path, falling back to a local directory if the
    configured path isn't writable (e.g. running outside Docker)."""
    p = Path(configured_path)
    if p.exists() and os.access(str(p), os.W_OK):
        return str(p)
    if p.parent.exists() and os.access(str(p.parent), os.W_OK):
        p.mkdir(parents=True, exist_ok=True)
        return str(p)
    local = Path(__file__).parent / ".data" / fallback_name
    local.mkdir(parents=True, exist_ok=True)
    return str(local)


def _resolve_db_path(configured_path: str) -> str:
    """Resolve the database file path, falling back to a local directory."""
    p = Path(configured_path)
    if p.parent.exists() and os.access(str(p.parent), os.W_OK):
        return str(p)
    if p.parent.parent.exists() and os.access(str(p.parent.parent), os.W_OK):
        p.parent.mkdir(parents=True, exist_ok=True)
        return str(p)
    local = Path(__file__).parent / ".data" / "db" / "smart_albums.db"
    local.parent.mkdir(parents=True, exist_ok=True)
    return str(local)


def find_config_path(config_path: Optional[str] = None) -> Optional[str]:
    """Resolve the config file path without loading it.

    Resolution order:
    1. Explicit path argument
    2. CONFIG_PATH environment variable
    3. /config/config.yaml (Docker convention)
    4. ./config.yaml (local development)
    """
    paths_to_try = [
        config_path,
        os.environ.get("CONFIG_PATH"),
        "/config/config.yaml",
        str(Path.cwd() / "config.yaml"),
    ]
    for p in paths_to_try:
        if p and Path(p).exists():
            return p
    return None


def providers_from_dict(raw: dict[str, Any]) -> dict[str, ProviderConfig]:
    """Convert a plain dict (as read from the DB settings table) into
    ``ProviderConfig`` instances keyed by protocol slot name."""
    return {name: ProviderConfig(**pc) for name, pc in raw.items()}


def resolve_provider_paths(cfg: AppConfig) -> None:
    """Sync the cachemanager provider's ``cache_dir`` with ``cfg.cache_dir``.

    Resolves the configured cache directory to a writable path (falling back
    to a local directory when the configured one isn't usable) and keeps the
    "cache" provider's params and the top-level ``cache_dir`` in agreement.
    Call this after mutating ``cfg.providers`` (e.g. after loading overrides
    from the database) so the cache path stays consistent.
    """
    cm = cfg.get_provider("cachemanager")
    if cm.selected == "cache":
        provider_cache_dir = cm.params.get("cache_dir")
        if provider_cache_dir:
            resolved = _resolve_data_dir(provider_cache_dir, "cache")
            cm.params["cache_dir"] = resolved
            cfg.cache_dir = resolved
        else:
            cm.params["cache_dir"] = cfg.cache_dir
        cfg.providers["cachemanager"] = cm


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Load configuration from a YAML file.

    Resolution order:
    1. Explicit path argument
    2. CONFIG_PATH environment variable
    3. /config/config.yaml (Docker convention)
    4. ./config.yaml (local development)
    """
    found = find_config_path(config_path)

    cfg = AppConfig()
    if found:
        with open(found, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        cfg = AppConfig(**raw)

    cfg.cache_dir = _resolve_data_dir(cfg.cache_dir, "cache")
    cfg.prompts_dir = _resolve_data_dir(cfg.prompts_dir, "prompts")
    cfg.exports_dir = _resolve_data_dir(cfg.exports_dir, "exports")
    cfg.db_path = _resolve_db_path(cfg.db_path)

    resolve_provider_paths(cfg)

    return cfg
