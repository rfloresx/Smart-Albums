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
        ConfigParam("negative_query_limit", int, default=500, min=1),
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
            # Tags are stored under metadata["smart_info"]["tags"] by
            # ImmichClient. Guard against smart_info/tags being explicitly
            # set to None (rather than simply absent), which would
            # otherwise raise AttributeError/TypeError and abort the whole
            # filter for every remaining asset.
            smart_info = asset.metadata.get("smart_info") or {}
            asset_tags: list[str] = smart_info.get("tags") or []
            if blocked_tags & set(asset_tags):
                signals.append("blocked_tag")
            flags = asset.metadata.get("vision_flags")
            # Check the vision flags independently of any error condition.
            # Previously this was an if/elif: `asset.metadata.get("error")`
            # (a generic key that ANY earlier stage — embedding, score,
            # etc. — can set for unrelated reasons) short-circuited the
            # flag check entirely. That meant:
            #   - with fail_closed=False: an asset with genuinely dangerous
            #     vision_flags was still let through, as long as some
            #     unrelated stage also failed on it.
            #   - with fail_closed=True: any unrelated error (e.g. a
            #     transient embedding fetch failure) excluded an otherwise
            #     clean asset as "fail_closed", even though vision analysis
            #     for this filter was never attempted or was fine.
            # `vision_error` below is a dedicated key a vision-analysis
            # stage can set to say "we tried to compute vision_flags for
            # this asset and failed" — distinct from the generic `error`
            # key other stages use for their own purposes.
            if flags and any(v is not None and v >= threshold for v in flags.values()):
                signals.append("vision_flag")
            if flags is None and fail_closed and asset.metadata.get("vision_error"):
                signals.append("fail_closed")

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
