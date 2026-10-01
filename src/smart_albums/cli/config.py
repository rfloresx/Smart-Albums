"""CLI configuration loading and validation helpers."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import typer
import yaml
from rich.console import Console

console = Console()

# Extensions that select the YAML parser in load_config_file(). Any other
# extension (including the common ".json" and no extension at all) is
# parsed as JSON, matching the CLI's historical default.
_YAML_SUFFIXES = frozenset({".yaml", ".yml"})


def load_config_file(path: Path) -> dict[str, Any]:
    """Load a JSON or YAML config file (selected by file extension), exit on error.

    Both the Docker CLI image and the README document YAML config files
    (e.g. ``rotation.yaml``), but this function historically only ever
    parsed JSON, so every documented Docker invocation failed immediately
    with a JSON decode error. YAML is a superset of JSON syntax-wise, but
    ``json.load`` doesn't accept YAML's non-JSON constructs (unquoted keys
    are fine, but comments, block scalars, etc. are not), so a real YAML
    parser is required for ``.yaml``/``.yml`` paths.

    Args:
        path: Path to the JSON or YAML config file.

    Returns:
        Parsed dict from the file. An empty file parses to ``{}``.

    Raises:
        typer.Exit: If the file is missing or contains invalid JSON/YAML,
            or if a YAML file parses to something other than a mapping.
    """
    if not path.exists():
        console.print(f"[red]Error: config file not found: {path}[/red]")
        raise typer.Exit(1)

    is_yaml = path.suffix.lower() in _YAML_SUFFIXES
    try:
        with path.open("r", encoding="utf-8") as fh:
            if is_yaml:
                loaded = yaml.safe_load(fh)
            else:
                loaded = json.load(fh)
    except (json.JSONDecodeError, yaml.YAMLError, OSError) as exc:
        kind = "YAML" if is_yaml else "JSON"
        console.print(f"[red]Error: failed to load {kind} config {path}: {exc}[/red]")
        raise typer.Exit(1)

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        console.print(
            f"[red]Error: config file {path} must contain a mapping/object at "
            f"the top level, got {type(loaded).__name__}[/red]"
        )
        raise typer.Exit(1)
    return loaded


def load_json_file(path: Path) -> dict[str, Any]:
    """Load a JSON file, exit on error.

    Kept for backward compatibility (tests and any external callers that
    import this name directly); it now dispatches to
    :func:`load_config_file`, which also handles YAML. New code should call
    :func:`load_config_file` directly.

    Args:
        path: Path to the JSON (or YAML) config file.

    Returns:
        Parsed dict from the file.

    Raises:
        typer.Exit: If the file is missing or contains invalid content.
    """
    return load_config_file(path)


def configure_logging(level: str, log_file: Optional[Path] = None) -> None:
    """Configure root logger with the given level and optional file handler.

    Args:
        level: Logging level string (DEBUG, INFO, WARNING, ERROR).
        log_file: Optional path for a file handler.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        handlers.append(logging.FileHandler(str(log_file), encoding="utf-8"))
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        handlers=handlers,
        force=True,
    )
    # Suppress noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
