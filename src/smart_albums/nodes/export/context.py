"""Export stage that writes pipeline context, metadata, and assets to disk."""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import re
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from smart_albums.core.context import ContextBatch, PipelineContext
from smart_albums.core.node import ConfigParam, Stage
from protocols_system import ProtocolsRegistry
from protocols_system.protocols import IImageClient, IProgressReporter
from smart_albums.core.registry import stage

logger = logging.getLogger(__name__)

# Characters allowed verbatim in a filename stem derived from an asset id.
_SAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]")


def _safe_asset_filename(asset_id: str, mime_type: str | None) -> str:
    """Build a filesystem-safe filename for an asset.

    The raw ``asset.id`` was previously used directly as the filename stem
    with a hardcoded ``.jpg`` extension (ND-17). An id containing ``/`` or
    ``..`` could escape the assets directory (path traversal), and the
    fixed ``.jpg`` mislabeled PNG/HEIC/video assets. This sanitizes the id
    to a safe stem and derives the extension from the asset's mime type.
    """
    stem = _SAFE_FILENAME.sub("_", asset_id).strip("._") or "asset"
    ext = None
    if mime_type:
        ext = mimetypes.guess_extension(mime_type.split(";")[0].strip())
    if not ext:
        ext = ".jpg"
    return f"{stem}{ext}"


@stage("export.context")
class ExportContext(Stage):
    """Export the pipeline context (assets + metadata) to a directory."""

    _config_schema = (
        ConfigParam(
            "export_directory",
            str,
            default="./output",
            description="Base directory for exported files. Relative paths resolve from the working directory.",
        ),
        ConfigParam(
            "image_size",
            str,
            default="thumbnail",
            choices=["thumbnail", "original"],
            description="Whether to export thumbnail-sized images or the full original resolution.",
        ),
        ConfigParam(
            "concurrency",
            int,
            default=4,
            min=1,
            description="Maximum concurrent asset downloads.",
        ),
        ConfigParam(
            "include_embeddings",
            bool,
            default=False,
            description="Include full embedding vectors in metadata.json. Off by "
            "default because they bloat the file (512-1024 floats per asset).",
        ),
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        export_directory = Path(self.get("export_directory"))
        image_size: str = self.get("image_size")
        concurrency: int = self.get("concurrency")
        include_embeddings: bool = self.get("include_embeddings")

        # Root/merged/fork contexts all collapse to partition_id="" /
        # partition_name="" and previously shared one "default" directory,
        # so a run that exported more than one such context (e.g. several
        # fork branches, or a root context plus a merged one) had each
        # overwrite the previous one's context.json/metadata.json/assets
        # (ND-17). Fall back to a unique directory name instead of a shared
        # "default" so distinct contexts no longer clobber each other.
        context_id = ctx.partition_id or ctx.partition_name
        if not context_id:
            context_id = f"context-{uuid.uuid4().hex[:8]}"
        # Also sanitize the context id itself — it feeds a directory name.
        context_id = _SAFE_FILENAME.sub("_", context_id).strip("._") or "context"
        context_dir = export_directory / context_id
        context_dir.mkdir(parents=True, exist_ok=True)

        # Write context information
        context_info = self._serialize_context_info(ctx)
        (context_dir / "context.json").write_text(
            json.dumps(context_info, indent=2, default=str)
        )

        # Write assets metadata
        assets_metadata = self._serialize_assets_metadata(ctx, include_embeddings)
        (context_dir / "metadata.json").write_text(
            json.dumps(assets_metadata, indent=2, default=str)
        )

        # Download and save asset images
        image_client = ProtocolsRegistry.get_instance(IImageClient)
        if image_client is None:
            raise RuntimeError(
                "image_client is required for stage 'export.context'"
            )

        assets_dir = context_dir / "assets"
        assets_dir.mkdir(exist_ok=True)

        progress = ProtocolsRegistry.get_instance(IProgressReporter)
        if progress:
            progress.start_stage(
                f"Exporting {len(ctx.assets)} assets ({image_size})",
                total=len(ctx.assets),
            )

        semaphore = asyncio.Semaphore(concurrency)
        exported = 0
        failed = 0
        lock = asyncio.Lock()

        async def _export_one(asset: Any) -> None:
            nonlocal exported, failed
            async with semaphore:
                try:
                    if image_size == "original":
                        image_bytes = await image_client.get_asset_full(asset.id)
                    else:
                        image_bytes = await image_client.get_asset_thumbnail(asset.id)
                    filename = _safe_asset_filename(asset.id, asset.mime_type)
                    (assets_dir / filename).write_bytes(image_bytes)
                    async with lock:
                        exported += 1
                except FileNotFoundError:
                    # Asset media missing on server — skip it.
                    async with lock:
                        failed += 1
                    if progress:
                        progress.warn(
                            f"Asset {asset.id} not found on server, skipping."
                        )
                except Exception as exc:  # noqa: BLE001
                    # A transport error, a disk write error, or any other
                    # failure for one asset should skip just that asset, not
                    # abort the export of every remaining one (ND-17 only
                    # caught FileNotFoundError).
                    async with lock:
                        failed += 1
                    logger.warning(
                        "export.context: failed to export asset %s: %s",
                        asset.id, exc,
                    )
                    if progress:
                        progress.warn(f"Failed to export asset {asset.id}: {exc}")
                finally:
                    if progress:
                        progress.advance()

        # Download concurrently (bounded by `concurrency`) instead of one at
        # a time — ND-17 noted the sequential loop as a bottleneck.
        await asyncio.gather(*[_export_one(a) for a in ctx.assets])

        if progress:
            progress.finish_stage()

        ctx.stats["export.context.output_dir"] = str(context_dir)
        # Count assets actually written, not the input size — the old
        # ``len(ctx.assets)`` reported skipped/failed assets as exported
        # (ND-17).
        ctx.stats["export.context.assets_exported"] = exported
        ctx.stats["export.context.assets_failed"] = failed
        ctx.stats["export.context.image_size"] = image_size

        return [ctx]

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_context_info(ctx: PipelineContext) -> dict[str, Any]:
        """Serialize context-level information (partition, stats, config)."""
        return {
            "partition_name": ctx.partition_name,
            "partition_id": ctx.partition_id,
            "parent_partition_id": ctx.parent_partition_id,
            "partition_depth": ctx.partition_depth,
            "asset_count": len(ctx.assets),
            "stats": ctx.stats,
            "metadata": ctx.metadata,
        }

    @staticmethod
    def _serialize_assets_metadata(
        ctx: PipelineContext, include_embeddings: bool = False
    ) -> list[dict[str, Any]]:
        """Serialize per-asset metadata into a list of dicts.

        Unless ``include_embeddings`` is set, the ``embedding`` entry in each
        asset's metadata is replaced with its dimensionality rather than the
        full vector — a 512-1024-float array per asset otherwise dominates
        metadata.json and makes it unwieldy to inspect (ND-17).
        """
        results: list[dict[str, Any]] = []
        for asset in ctx.assets:
            data = asdict(asset)
            # Ensure datetime is serialized as ISO string
            if asset.captured_at is not None:
                data["captured_at"] = asset.captured_at.isoformat()
            if not include_embeddings:
                meta = data.get("metadata")
                if isinstance(meta, dict) and "embedding" in meta:
                    embedding = meta["embedding"]
                    try:
                        dim = len(embedding)
                    except TypeError:
                        dim = None
                    meta = dict(meta)
                    meta["embedding"] = f"<omitted: {dim}-dim vector>" if dim is not None else "<omitted>"
                    data["metadata"] = meta
            results.append(data)
        return results
