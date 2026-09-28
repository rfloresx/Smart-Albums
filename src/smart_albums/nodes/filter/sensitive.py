"""filter.sensitive — apply sensitive-content exclusion across multiple signals."""

from __future__ import annotations

from smart_albums.core.context import PipelineContext, ContextBatch
from protocols_system.protocols import Asset
from smart_albums.core.node import Stage, ConfigParam
from protocols_system import ProtocolsRegistry
from protocols_system.protocols import IImageClient, IProgressReporter
from smart_albums.core.registry import stage


@stage("filter.sensitive")
class FilterSensitive(Stage):
    """Apply sensitive-content exclusion across multiple signals."""

    _config_schema = (
        ConfigParam("enabled", bool, default=False),
        ConfigParam("fail_closed", bool, default=False),
        ConfigParam("blocklist_ids", list[str], default=None),
        ConfigParam("blocked_tags", list[str], default=None),
        ConfigParam("negative_queries", list[str], default=None),
        ConfigParam("negative_query_limit", int, default=500),
        ConfigParam(
            "vision_threshold",
            float,
            default=0.7,
            min=0.0,
            max=1.0,
        ),
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        if not self.get("enabled"):
            return [ctx]

        # 1. Fetch negative-query matches
        image_client = ProtocolsRegistry.get_instance(IImageClient)
        if image_client is None:
            raise RuntimeError("image_client is required for stage 'filter.sensitive'")

        excluded_ids: dict[str, list[str]] = {}  # asset_id -> [signals]
        for query in self.get("negative_queries") or []:
            limit = self.get("negative_query_limit")
            for hit in await image_client.search_smart(query, limit):
                excluded_ids.setdefault(hit.id, []).append("negative_query")

        # 2. Apply blocklist / blocked tags / vision flags / fail_closed
        blocklist = set(self.get("blocklist_ids") or [])
        blocked_tags = set(self.get("blocked_tags") or [])
        threshold = self.get("vision_threshold")
        fail_closed = self.get("fail_closed")

        kept: list[Asset] = []
        tally = {
            "manual_blocklist": 0,
            "blocked_tag": 0,
            "negative_query": 0,
            "vision_flag": 0,
            "fail_closed": 0,
        }
        for asset in ctx.assets:
            signals = list(excluded_ids.get(asset.id, []))
            if asset.id in blocklist:
                signals.append("manual_blocklist")
            # Tags are stored under metadata["smart_info"]["tags"] by ImmichClient
            asset_tags: list[str] = (
                asset.metadata.get("smart_info", {}).get("tags", [])
            )
            if blocked_tags & set(asset_tags):
                signals.append("blocked_tag")
            flags = asset.metadata.get("vision_flags")
            if flags is None or asset.metadata.get("error"):
                # Only apply fail_closed when vision analysis was attempted but
                # produced an error — not when flags simply haven't been computed
                if fail_closed and asset.metadata.get("error"):
                    signals.append("fail_closed")
            elif any(v >= threshold for v in flags.values()):
                signals.append("vision_flag")

            if signals:
                for s in signals:
                    tally[s] = tally.get(s, 0) + 1
            else:
                kept.append(asset)

        excluded = len(ctx.assets) - len(kept)
        ctx.assets = kept
        ctx.stats["filter.sensitive.excluded"] = excluded
        for signal, count in tally.items():
            ctx.stats[f"filter.sensitive.excluded.{signal}"] = count
        if excluded:
            progress = ProtocolsRegistry.get_instance(IProgressReporter)
            if progress:
                progress.log(f"Excluded {excluded} sensitive asset(s).")
        return [ctx]
