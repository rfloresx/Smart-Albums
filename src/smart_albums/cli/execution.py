"""Pipeline execution — client lifecycle management and pipeline invocation."""

from __future__ import annotations

import contextlib
import logging
import os
from typing import Any, Optional

from rich.console import Console
from rich.table import Table

from protocols_system import ProtocolsRegistry

logger = logging.getLogger(__name__)
console = Console()

# Environment variables documented in the Dockerfile and README as the way
# to configure the CLI's default providers in a container (e.g. `docker run
# -e IMMICH_URL=... -e IMMICH_API_KEY=... -e OLLAMA_URL=... -e
# OLLAMA_MODEL=...`). Nothing in the CLI actually read these; provider
# configuration only ever came from CLI flags or dotted config-file keys, so
# every documented `docker run` invocation silently ran with unset/default
# connection settings. Each entry maps an env var name to the
# (protocol_name, provider_name, param_name) triple it should fill — but
# only when the provider actually chosen for that protocol matches
# provider_name, and only as a fallback below CLI flags and config file
# values (which already take priority over each other above).
_ENV_VAR_PROVIDER_DEFAULTS: dict[str, tuple[str, str, str]] = {
    "IMMICH_URL": ("image", "immich", "base_url"),
    "IMMICH_API_KEY": ("image", "immich", "api_key"),
    "OLLAMA_URL": ("llm", "ollama", "base_url"),
    "OLLAMA_MODEL": ("llm", "ollama", "vision_model"),
}


def _default_provider_for(protocol_name: str, providers: dict[str, Any]) -> str:
    """Pick the default provider for a protocol when none was chosen
    explicitly via a CLI flag or config file key.

    Previously this was always ``next(iter(providers.keys()))`` — whichever
    provider happened to be registered first, which depends on Python's
    dict insertion order, which in turn depends on module import order
    (``smart_albums.clients`` imports ``core.cache`` before ``core.no_cache``,
    see ``cli/imports.import_clients``). For ``cachemanager`` specifically,
    that meant caching was silently *on* by default, using
    ``CacheManager``'s own default directory — surprising for anyone who
    didn't explicitly configure a cache, and broken in the CLI Docker image
    (see ``core.cache._default_cache_dir``'s docstring). Caching should be
    an opt-in choice, not an accident of import order, so "cachemanager"
    defaults to the no-op "disabled" provider unless the user configured
    "cache" (or another provider) explicitly.
    """
    if protocol_name == "cachemanager" and "disabled" in providers:
        return "disabled"
    return next(iter(providers.keys()))


def _apply_env_var_defaults(
    protocol_name: str, chosen: str, provider_kwargs: dict[str, Any]
) -> None:
    """Fill in provider_kwargs from documented env vars, without overriding
    values already supplied via CLI flags or the config file."""
    for env_name, (target_protocol, target_provider, param_name) in _ENV_VAR_PROVIDER_DEFAULTS.items():
        if target_protocol != protocol_name or target_provider != chosen:
            continue
        if param_name in provider_kwargs:
            continue
        value = os.environ.get(env_name)
        if value:
            provider_kwargs[param_name] = value


# Provider kwargs whose names contain any of these substrings are treated as
# secrets and redacted before being written to the debug log. Debug logs are
# routinely captured wholesale (e.g. the web GUI persists CLI subprocess
# output into job records/log files, which can be exported or viewed by
# anyone with access to those records), so API keys must never appear there
# in the clear.
_SECRET_KEY_MARKERS = ("key", "token", "secret", "password", "credential")


def _redact_secrets(provider_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of provider kwargs with secret-looking values redacted."""
    redacted: dict[str, Any] = {}
    for key, value in provider_kwargs.items():
        if any(marker in key.lower() for marker in _SECRET_KEY_MARKERS):
            redacted[key] = "***REDACTED***"
        else:
            redacted[key] = value
    return redacted


async def execute_pipeline(
    pipeline: list[Any],
    kwargs: dict[str, Any],
    cfg: dict[str, Any],
    *,
    progress_json: bool = False,
) -> None:
    """Set up clients from kwargs/config, build context, and run the pipeline.

    Resolves which provider to use per protocol from CLI kwargs and config
    file values, enters async context managers for all clients, runs the
    pipeline, and prints a summary on completion.

    Uses AsyncExitStack to guarantee proper cleanup of all clients even
    if one fails during setup.

    Args:
        pipeline: The resolved pipeline definition (list of steps).
        kwargs: Merged CLI + config key-value pairs.
        cfg: Raw config file dict (for fallback lookups).
        progress_json: If True, emit JSON progress events to stdout instead
            of using the Rich terminal reporter.
    """
    from smart_albums.cli.imports import import_all_stages
    from smart_albums.core.cache import CacheManager
    from smart_albums.core.context import PipelineContext
    from smart_albums.core.runner import run_pipeline
    from smart_albums.progress import JsonProgressReporter, RichProgressReporter

    global_cfg: dict[str, Any] = {}
    for key, value in kwargs.items():
        if value is not None:
            global_cfg[key] = value
        elif key in cfg:
            global_cfg[key] = cfg[key]
    kwargs = global_cfg
    import_all_stages()

    progress: Any
    # In JSON progress mode stdout is a machine-readable event stream parsed
    # by the parent (web GUI) process; the human-oriented banner/summary and
    # the Rich live display must not share it or they corrupt that stream
    # (CO-11). Route all Rich output to stderr in that mode.
    run_console = Console(stderr=True) if progress_json else console
    if progress_json:
        progress = JsonProgressReporter()
    else:
        progress = RichProgressReporter(console=run_console)

    from protocols_system.protocols import (
        ICacheManager,
        IEmbeddingClient,
        IImageClient,
        ILLMClient,
        IProgressReporter,
    )

    # Everything from client creation through pipeline execution runs inside
    # a single AsyncExitStack, and registry/progress teardown runs in a
    # finally (CO-10). Previously clients were all constructed *before* the
    # stack existed, so if constructing or entering client #3 failed, the
    # already-created/entered #1 and #2 leaked (their __aexit__ never ran);
    # non-core clients like the geo client were never entered into the stack
    # at all; and clear_instances() sat after the stack with no finally, so
    # an exception during the run left stale instances registered globally.
    async with contextlib.AsyncExitStack() as stack:
        try:
            ProtocolsRegistry.clear_instances()
            ProtocolsRegistry.set_instance(IProgressReporter, progress)
            # Ensure the Rich live display is stopped even if the run raises
            # — otherwise its Live render loop keeps running over the summary
            # output and can leave the terminal cursor hidden (CO-10).
            stack.callback(lambda: getattr(progress, "stop", lambda: None)())

            clients: dict[str, Any] = {}
            # Resolve which provider to use per protocol, constructing each
            # client *inside* the stack and immediately entering it (if it's
            # an async context manager) so a later failure unwinds every
            # client created so far.
            for protocol, providers in ProtocolsRegistry.get_registry().items():
                protocol_name = protocol.__name__.removeprefix("I").removesuffix("Client").lower()

                # Skip internal protocols not meant for pipeline execution
                if protocol_name in ("healthcheck",):
                    continue

                chosen = kwargs.get(protocol_name) or cfg.get(protocol_name) or _default_provider_for(
                    protocol_name, providers
                )

                # Gather config for that provider
                prefix = f"{protocol_name}_{chosen}_"
                provider_kwargs: dict[str, Any] = {}
                for key, value in kwargs.items():
                    if key.startswith(prefix):
                        param_name = key[len(prefix):]
                        if value is not None:
                            provider_kwargs[param_name] = value

                # Fallback to config file values
                config_prefix = f"{protocol_name}.{chosen}."
                for key, value in cfg.items():
                    if key.startswith(config_prefix):
                        param_name = key[len(config_prefix):]
                        if param_name not in provider_kwargs:
                            provider_kwargs[param_name] = value

                # Fallback to documented env vars, lowest priority.
                _apply_env_var_defaults(protocol_name, chosen, provider_kwargs)

                logger.debug(
                    "Protocol %s: chosen=%s, provider_kwargs=%s",
                    protocol_name, chosen, _redact_secrets(provider_kwargs),
                )
                client: Any = ProtocolsRegistry.create(protocol, chosen, provider_kwargs)
                # Enter async-context clients immediately so cleanup is
                # registered before the next client is constructed. This
                # covers *every* protocol (geo included), not just the four
                # core ones — the old code only entered image/llm/embedding/
                # cache, leaking anything else that held resources.
                if client is not None and hasattr(client, "__aenter__"):
                    client = await stack.enter_async_context(client)
                clients[protocol_name] = client

            # Determine client roles (post-enter references)
            image_client = clients.get("image")
            llm_client = clients.get("llm")
            embedding_client = clients.get("embedding")
            cache_manager: Optional[CacheManager] = clients.get("cachemanager")

            # Register all client instances in the ProtocolsRegistry.
            if image_client is not None:
                ProtocolsRegistry.set_instance(IImageClient, image_client)
            if llm_client is not None:
                ProtocolsRegistry.set_instance(ILLMClient, llm_client)
            if embedding_client is not None:
                ProtocolsRegistry.set_instance(IEmbeddingClient, embedding_client)
            if cache_manager is not None:
                ProtocolsRegistry.set_instance(ICacheManager, cache_manager)

            # Register any remaining protocol clients (e.g. IGeoClient).
            for protocol, providers in ProtocolsRegistry.get_registry().items():
                if ProtocolsRegistry.get_instance(protocol) is not None:
                    continue  # Already registered above
                protocol_name = protocol.__name__.removeprefix("I").removesuffix("Client").lower()
                if protocol_name in clients and clients[protocol_name] is not None:
                    ProtocolsRegistry.set_instance(protocol, clients[protocol_name])

            ctx = PipelineContext(
                config={},
                assets=[],
                stats={},
            )

            run_console.print("\n[bold]🎞  smart-albums[/bold]")
            cache_label = getattr(cache_manager, "_cache_dir", None) or "disabled"
            run_console.print(f"   Cache: {cache_label}")

            # Log configured providers and models
            if llm_client is not None:
                llm_name = type(llm_client).__name__
                vision_model = getattr(llm_client, "vision_model", "n/a")
                llm_url = getattr(llm_client, "_base_url", "")
                run_console.print(f"   LLM: {llm_name} — model: [cyan]{vision_model}[/cyan] @ {llm_url}")
                logger.info("LLM provider: %s, vision_model=%s, url=%s", llm_name, vision_model, llm_url)

            if embedding_client is not None:
                emb_name = type(embedding_client).__name__
                embed_model = getattr(embedding_client, "embed_model", "n/a")
                emb_url = getattr(embedding_client, "_base_url", getattr(embedding_client, "_model_name", ""))
                run_console.print(f"   Embedding: {emb_name} — model: [cyan]{embed_model}[/cyan] @ {emb_url}")
                logger.info("Embedding provider: %s, model=%s, url=%s", emb_name, embed_model, emb_url)

            if image_client is not None:
                img_name = type(image_client).__name__
                img_url = getattr(image_client, "_base_url", "")
                run_console.print(f"   Image: {img_name} @ {img_url}")
                logger.info("Image provider: %s, url=%s", img_name, img_url)

            run_console.print()

            results = await run_pipeline(pipeline, [ctx])
            _print_summary(results, console=run_console)
        finally:
            # Always clear globally-registered instances, even if the run
            # raised — otherwise stale clients (now being torn down by the
            # exit stack) stayed registered for the next invocation (CO-10).
            ProtocolsRegistry.clear_instances()


def _print_summary(results: list[Any], console: Console = console) -> None:
    """Print a summary of pipeline results.

    ``console`` defaults to the module stdout console but is passed the
    stderr console in JSON progress mode so the summary doesn't corrupt the
    stdout event stream (CO-11).
    """
    console.print("\n[bold green]✓ Pipeline complete[/bold green]\n")

    total_assets = 0
    all_stats: dict[str, Any] = {}
    for ctx in results:
        total_assets += len(ctx.assets)
        all_stats.update(ctx.stats)

    table = Table(title="Pipeline Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="white")
    table.add_row("Final assets", str(total_assets))

    for key, label in [
        ("analyze.score.computed", "Assets scored"),
        ("analyze.score.cached", "Scores from cache"),
        ("analyze.embedding.computed", "Embeddings computed"),
        ("analyze.embedding.cached", "Embeddings from cache"),
    ]:
        if key in all_stats:
            table.add_row(label, str(all_stats[key]))

    console.print(table)
