"""Pipeline execution — client lifecycle management and pipeline invocation."""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Optional

from rich.console import Console
from rich.table import Table

from smart_albums.core.protocols import ProtocolsRegistry

logger = logging.getLogger(__name__)
console = Console()


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
    if progress_json:
        progress = JsonProgressReporter()
    else:
        progress = RichProgressReporter(console=console)

    # Resolve which provider to use per protocol
    clients: dict[str, Any] = {}
    for protocol, providers in ProtocolsRegistry.get_registry().items():
        protocol_name = protocol.__name__.removeprefix("I").removesuffix("Client").lower()

        # Skip internal protocols not meant for pipeline execution
        if protocol_name in ("healthcheck",):
            continue

        chosen = kwargs.get(protocol_name) or cfg.get(protocol_name) or next(iter(providers.keys()))

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

        logger.debug(
            "Protocol %s: chosen=%s, provider_kwargs=%s",
            protocol_name, chosen, provider_kwargs,
        )
        clients[protocol_name] = ProtocolsRegistry.create(protocol, chosen, provider_kwargs)

    # Determine client roles
    image_client = clients.get("image")
    llm_client = clients.get("llm")
    embedding_client = clients.get("embedding")
    cache_manager: Optional[CacheManager] = clients.get("cachemanager")

    # Enter async context managers via AsyncExitStack for guaranteed cleanup
    async with contextlib.AsyncExitStack() as stack:
        if image_client is not None and hasattr(image_client, "__aenter__"):
            image_client = await stack.enter_async_context(image_client)
        if llm_client is not None and hasattr(llm_client, "__aenter__"):
            llm_client = await stack.enter_async_context(llm_client)
        if embedding_client is not None and hasattr(embedding_client, "__aenter__"):
            embedding_client = await stack.enter_async_context(embedding_client)
        if cache_manager is not None and hasattr(cache_manager, "__aenter__"):
            cache_manager = await stack.enter_async_context(cache_manager)  # type: ignore[arg-type]

        # Register context-managed instances in the ProtocolsRegistry
        from smart_albums.core.protocols import (
            ICacheManager,
            IEmbeddingClient,
            IImageClient,
            ILLMClient,
            IProgressReporter,
        )

        ProtocolsRegistry.clear_instances()
        if image_client is not None:
            ProtocolsRegistry.set_instance(IImageClient, image_client)
        if llm_client is not None:
            ProtocolsRegistry.set_instance(ILLMClient, llm_client)
        if embedding_client is not None:
            ProtocolsRegistry.set_instance(IEmbeddingClient, embedding_client)
        if cache_manager is not None:
            ProtocolsRegistry.set_instance(ICacheManager, cache_manager)
        ProtocolsRegistry.set_instance(IProgressReporter, progress)

        ctx = PipelineContext(
            config={},
            assets=[],
            stats={},
        )

        console.print("\n[bold]🎞  smart-albums[/bold]")
        cache_label = getattr(cache_manager, "_cache_dir", None) or "disabled"
        console.print(f"   Cache: {cache_label}")

        # Log configured providers and models
        if llm_client is not None:
            llm_name = type(llm_client).__name__
            vision_model = getattr(llm_client, "vision_model", "n/a")
            llm_url = getattr(llm_client, "_base_url", "")
            console.print(f"   LLM: {llm_name} — model: [cyan]{vision_model}[/cyan] @ {llm_url}")
            logger.info("LLM provider: %s, vision_model=%s, url=%s", llm_name, vision_model, llm_url)

        if embedding_client is not None:
            emb_name = type(embedding_client).__name__
            embed_model = getattr(embedding_client, "embed_model", "n/a")
            emb_url = getattr(embedding_client, "_base_url", getattr(embedding_client, "_model_name", ""))
            console.print(f"   Embedding: {emb_name} — model: [cyan]{embed_model}[/cyan] @ {emb_url}")
            logger.info("Embedding provider: %s, model=%s, url=%s", emb_name, embed_model, emb_url)

        if image_client is not None:
            img_name = type(image_client).__name__
            img_url = getattr(image_client, "_base_url", "")
            console.print(f"   Image: {img_name} @ {img_url}")
            logger.info("Image provider: %s, url=%s", img_name, img_url)

        console.print()

        results = await run_pipeline(pipeline, [ctx])
        _print_summary(results)

        ProtocolsRegistry.clear_instances()


def _print_summary(results: list[Any]) -> None:
    """Print a summary of pipeline results."""
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
