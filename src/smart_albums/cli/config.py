"""CLI configuration loading and validation helpers."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console

console = Console()


def load_json_file(path: Path) -> dict[str, Any]:
    """Load a JSON file, exit on error.

    Args:
        path: Path to the JSON config file.

    Returns:
        Parsed dict from the JSON file.

    Raises:
        typer.Exit: If the file is missing or contains invalid JSON.
    """
    if not path.exists():
        console.print(f"[red]Error: config file not found: {path}[/red]")
        raise typer.Exit(1)
    try:
        with path.open("r", encoding="utf-8") as fh:
            result: dict[str, Any] = json.load(fh)
            return result
    except (json.JSONDecodeError, OSError) as exc:
        console.print(f"[red]Error: failed to load {path}: {exc}[/red]")
        raise typer.Exit(1)


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
