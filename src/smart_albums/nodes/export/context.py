"""Export stage that writes pipeline context, metadata, and assets to disk."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from smart_albums.core.context import ContextBatch, PipelineContext
from smart_albums.core.node import ConfigParam, Stage
from smart_albums.core.protocol_registry import ProtocolsRegistry
from smart_albums.core.protocols import IImageClient, IProgressReporter
from smart_albums.core.registry import stage


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
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        export_directory = Path(self.get("export_directory"))
        image_size: str = self.get("image_size")

        context_id = ctx.partition_id or ctx.partition_name or "default"
        context_dir = export_directory / context_id
        context_dir.mkdir(parents=True, exist_ok=True)

        # Write context information
        context_info = self._serialize_context_info(ctx)
        (context_dir / "context.json").write_text(
            json.dumps(context_info, indent=2, default=str)
        )

        # Write assets metadata
        assets_metadata = self._serialize_assets_metadata(ctx)
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

        for asset in ctx.assets:
            try:
                if image_size == "original":
                    image_bytes = await image_client.get_asset_full(asset.id)
                else:
                    image_bytes = await image_client.get_asset_thumbnail(asset.id)
                (assets_dir / f"{asset.id}.jpg").write_bytes(image_bytes)
            except FileNotFoundError:
                # Asset media missing on server — skip it
                if progress:
                    progress.warn(
                        f"Asset {asset.id} not found on server, skipping."
                    )
            if progress:
                progress.advance()

        if progress:
            progress.finish_stage()

        ctx.stats["export.context.output_dir"] = str(context_dir)
        ctx.stats["export.context.assets_exported"] = len(ctx.assets)
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
    def _serialize_assets_metadata(ctx: PipelineContext) -> list[dict[str, Any]]:
        """Serialize per-asset metadata into a list of dicts."""
        results: list[dict[str, Any]] = []
        for asset in ctx.assets:
            data = asdict(asset)
            # Ensure datetime is serialized as ISO string
            if asset.captured_at is not None:
                data["captured_at"] = asset.captured_at.isoformat()
            results.append(data)
        return results
