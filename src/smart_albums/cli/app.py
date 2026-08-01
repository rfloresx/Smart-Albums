"""CLI entry point for smart-albums.

Usage:
    smart-albums best-of-year --config config.json
    smart-albums rotation --config config.json

The config.json file contains all configuration: client settings and
per-stage pipeline overrides. Client options can also be passed as CLI
flags (they override the config file).
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from pathlib import Path
from typing import Any, Optional

import typer
from rich.console import Console

from smart_albums.cli.config import configure_logging, load_json_file
from smart_albums.cli.execution import execute_pipeline
from smart_albums.cli.imports import import_all_stages, import_clients
from smart_albums.core.node import ConfigParam
from smart_albums.core.protocol_registry import ProtocolsRegistry
import smart_albums.core.builder as builder

app = typer.Typer(
    name="smart-albums",
    help="Create curated photo albums in Immich using AI-powered quality scoring.",
    add_completion=False,
)

logger = logging.getLogger(__name__)
console = Console()


# Keep backward-compatible aliases for tests that import these directly
_load_json_file = load_json_file
_configure_logging = configure_logging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Map string type annotations (from __future__ annotations) to real types
_TYPE_MAP: dict[str, Any] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
    "str | None": Optional[str],
    "int | None": Optional[int],
    "float | None": Optional[float],
    "bool | None": Optional[bool],
    "Optional[str]": Optional[str],
    "Optional[int]": Optional[int],
    "Optional[float]": Optional[float],
    "Optional[bool]": Optional[bool],
}

# Type annotations that cannot be meaningfully mapped to a CLI option.
# Parameters with these annotations are skipped during dynamic CLI generation.
_SKIP_TYPES: frozenset[str] = frozenset({"Any", "dict", "list"})


def _resolve_type(t: Any) -> Any | None:
    """Resolve a type annotation (possibly a deferred string) to an actual type.

    Returns None for types that should be skipped (not exposed as CLI options).
    """
    if isinstance(t, str):
        if t in _SKIP_TYPES:
            return None
        return _TYPE_MAP.get(t, str)
    return t


def _make_option(config: ConfigParam) -> Any:
    """Create a typer.Option from a ConfigParam.

    Always uses None as the default so that config file values are not
    overridden by constructor defaults when the user doesn't explicitly
    provide the CLI flag.
    """
    return typer.Option(
        default=None,
        help=config.description or None,
    )


# ---------------------------------------------------------------------------
# Dynamic CLI signature builder
# ---------------------------------------------------------------------------


def _build_run_signature() -> list[inspect.Parameter]:
    """Build CLI parameters dynamically from registered protocol providers."""
    params: list[inspect.Parameter] = []

    for protocol, providers in ProtocolsRegistry.get_registry().items():
        protocol_name = protocol.__name__.removeprefix("I").removesuffix("Client").lower()

        # Provider selector: --embedding ollama_v2
        options_name = "(" + ",".join(providers.keys()) + ")"
        params.append(
            inspect.Parameter(
                protocol_name,
                inspect.Parameter.KEYWORD_ONLY,
                annotation=Optional[str],
                default=typer.Option(
                    None,
                    help=f"{protocol_name}: {options_name}",
                ),
            )
        )

        # Per-provider params: --embedding-ollama-v2-model
        for provider_name, cls in providers.items():
            schema = builder.get_init_schema(cls)
            for param_name, param_config in schema.items():
                resolved = _resolve_type(param_config.type)
                if resolved is None:
                    # Skip params with unmappable types (dict, list, Any)
                    continue
                option_name = (
                    f"{protocol_name}_"
                    f"{provider_name}_"
                    f"{param_name}"
                )
                params.append(
                    inspect.Parameter(
                        option_name,
                        inspect.Parameter.KEYWORD_ONLY,
                        annotation=resolved,
                        default=_make_option(param_config),
                    )
                )

    # Global options
    params.append(
        inspect.Parameter(
            "log_level",
            inspect.Parameter.KEYWORD_ONLY,
            annotation=str,
            default=typer.Option("INFO", help="Logging level: DEBUG/INFO/WARNING/ERROR."),
        )
    )
    params.append(
        inspect.Parameter(
            "log_file",
            inspect.Parameter.KEYWORD_ONLY,
            annotation=Optional[Path],
            default=typer.Option(None, help="Write logs to this file."),
        )
    )
    params.append(
        inspect.Parameter(
            "config",
            inspect.Parameter.KEYWORD_ONLY,
            annotation=Path,
            default=typer.Option(
                Path("config.json"),
                "--config", "-c",
                help="Path to JSON config file.",
                envvar="SMART_ALBUMS_CONFIG",
            ),
        )
    )
    params.append(
        inspect.Parameter(
            "progress_json",
            inspect.Parameter.KEYWORD_ONLY,
            annotation=bool,
            default=typer.Option(
                False,
                "--progress-json",
                help="Emit progress events as JSON lines to stdout.",
            ),
        )
    )

    return params


# ---------------------------------------------------------------------------
# Register commands with dynamic signatures
# ---------------------------------------------------------------------------

import_clients()

import smart_albums.pipelines as _import_pipelines  # noqa: F401, E402

from smart_albums.pipelines.registry import PipelineInfo, get_pipelines, get_pipeline  # noqa: E402


def _build_run_pipeline(pipeline_info: PipelineInfo) -> Any:
    """Build a Typer command function for a registered pipeline."""

    def run(**kwargs: Any) -> None:
        config_path: Path = kwargs.pop("config", Path("config.json"))
        log_level: str = kwargs.pop("log_level", "INFO")
        log_file: Optional[Path] = kwargs.pop("log_file", None)
        progress_json: bool = kwargs.pop("progress_json", False)

        # Load config file
        cfg = load_json_file(config_path) if config_path.exists() else {}
        # Config file log_level overrides default but CLI flag takes priority
        if log_level == "INFO" and "log_level" in cfg:
            log_level = cfg["log_level"]

        configure_logging(log_level, log_file)
        import_all_stages()

        # Collect alias-keyed overrides from pipeline_settings
        overrides: dict[str, dict[str, Any]] = cfg.get("pipeline_settings", {})
        pipeline = builder.resolve_overrides(pipeline_info.pipeline, overrides)

        try:
            asyncio.run(execute_pipeline(pipeline, kwargs, cfg, progress_json=progress_json))
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user.[/yellow]")
            raise typer.Exit(130)
        except Exception as exc:
            logger.exception("Pipeline failed")
            console.print(f"[red]Pipeline failed: {exc}[/red]")
            raise typer.Exit(1)

    run.__doc__ = pipeline_info.description
    run.__signature__ = inspect.Signature(_build_run_signature())  # type: ignore[attr-defined]
    return run


for name in get_pipelines():
    pipelineInfo = get_pipeline(name)
    if pipelineInfo is not None:
        cmd = app.command(name)(_build_run_pipeline(pipelineInfo))


# ---------------------------------------------------------------------------
# `run` command — execute an arbitrary pipeline defined in a JSON config
# ---------------------------------------------------------------------------


def _parse_pipeline_definition(raw: list[Any]) -> list[Any]:
    """Parse a JSON pipeline definition into framework Pipeline objects.

    Each entry in *raw* is a JSON array (list) of 2 or 3 elements:
        [stage_name, config_dict]
        [stage_name, config_dict, alias_name]

    For fork/fork.by_selection stages, the config_dict should contain a
    "branches" key with nested pipeline definitions::

        ["fork", {"branches": [["curation", [...steps...]], ["remainder", [...steps...]]]}, "output"]
        ["fork.by_selection", {"branches": {"main": [...steps...], "rest": [...steps...]}}, "output"]

    Returns a Pipeline list suitable for resolve_overrides / run_pipeline.
    """
    from smart_albums.core.spec import alias as make_alias

    pipeline: list[Any] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, (list, tuple)):
            raise typer.Exit(1)
        if len(entry) == 2:
            stage_name, config = entry
            config = _parse_step_config(stage_name, config)
            pipeline.append((stage_name, config))
        elif len(entry) == 3:
            stage_name, config, alias_name = entry
            config = _parse_step_config(stage_name, config)
            if alias_name:
                pipeline.append(make_alias(alias_name, (stage_name, config)))
            else:
                pipeline.append((stage_name, config))
        else:
            console.print(
                f"[red]Error: pipeline entry {i} must have 2 or 3 elements, "
                f"got {len(entry)}[/red]"
            )
            raise typer.Exit(1)
    return pipeline


def _parse_step_config(stage_name: str, config: dict[str, Any]) -> dict[str, Any]:
    """Parse step config, recursively handling fork branch pipelines.

    If the stage is a fork type and contains a "branches" key, each branch's
    pipeline is recursively parsed from JSON format into framework Pipeline
    objects.
    """
    if not isinstance(config, dict):
        return config

    if stage_name not in ("fork", "fork.by_selection"):
        return config

    if "branches" not in config:
        return config

    branches_raw = config["branches"]

    if isinstance(branches_raw, list):
        # List of [name, pipeline_steps] pairs
        parsed_branches: list[list[Any]] = []
        for branch in branches_raw:
            if isinstance(branch, (list, tuple)) and len(branch) == 2:
                branch_name, branch_steps = branch
                if isinstance(branch_steps, list):
                    parsed_pipeline = _parse_pipeline_definition(branch_steps)
                    parsed_branches.append([branch_name, parsed_pipeline])
                else:
                    parsed_branches.append(list(branch))
            else:
                parsed_branches.append(list(branch) if isinstance(branch, (list, tuple)) else branch)
        return {**config, "branches": parsed_branches}

    elif isinstance(branches_raw, dict):
        # Dict of {name: pipeline_steps}
        parsed_dict: dict[str, Any] = {}
        for branch_name, branch_steps in branches_raw.items():
            if isinstance(branch_steps, list):
                parsed_dict[branch_name] = _parse_pipeline_definition(branch_steps)
            else:
                parsed_dict[branch_name] = branch_steps
        return {**config, "branches": parsed_dict}

    return config


@app.command("run")
def run_command(
    config: Path = typer.Option(
        ...,
        "--config", "-c",
        help="Path to JSON config file defining the pipeline.",
        envvar="SMART_ALBUMS_CONFIG",
    ),
    log_level: str = typer.Option("INFO", help="Logging level: DEBUG/INFO/WARNING/ERROR."),
    log_file: Optional[Path] = typer.Option(None, help="Write logs to this file."),
    progress_json: bool = typer.Option(
        False,
        "--progress-json",
        help="Emit progress events as JSON lines to stdout.",
    ),
) -> None:
    """Run an arbitrary pipeline defined in a JSON config file.

    The config file must contain a "pipeline" key with a list of steps.
    Each step is a list of [stage_name, config_dict] or
    [stage_name, config_dict, alias].

    Example config:

    \b
    {
      "log_level": "INFO",
      "image": "immich",
      "image.immich.base_url": "http://localhost:2283/api",
      "image.immich.api_key": "your-api-key",
      "pipeline": [
        ["retrieve_by_year", {"year": 2024}, "retrieve"],
        ["retain_images", {}, "filter_images"],
        ["analyze_score", {}, "score"]
      ]
    }
    """
    cfg = load_json_file(config)

    # Extract and validate pipeline definition
    raw_pipeline = cfg.pop("pipeline", None)
    if raw_pipeline is None:
        console.print("[red]Error: config file must contain a 'pipeline' key.[/red]")
        raise typer.Exit(1)
    if not isinstance(raw_pipeline, list):
        console.print("[red]Error: 'pipeline' must be a list of steps.[/red]")
        raise typer.Exit(1)

    # Respect log_level from config, CLI flag overrides
    effective_log_level = log_level
    if log_level == "INFO" and "log_level" in cfg:
        effective_log_level = cfg.pop("log_level")
    else:
        cfg.pop("log_level", None)

    configure_logging(effective_log_level, log_file)
    import_all_stages()

    # Parse the pipeline definition
    pipeline = _parse_pipeline_definition(raw_pipeline)

    # Apply pipeline_settings overrides if present
    overrides: dict[str, dict[str, Any]] = cfg.pop("pipeline_settings", {})
    if overrides:
        pipeline = builder.resolve_overrides(pipeline, overrides)

    # Remaining cfg keys are treated as client/provider configuration
    kwargs: dict[str, Any] = {}

    try:
        asyncio.run(execute_pipeline(pipeline, kwargs, cfg, progress_json=progress_json))
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user.[/yellow]")
        raise typer.Exit(130)
    except Exception as exc:
        logger.exception("Pipeline failed")
        console.print(f"[red]Pipeline failed: {exc}[/red]")
        raise typer.Exit(1)
