"""avoid_repeat — partition assets into fresh and backfilled pools."""

from __future__ import annotations

from datetime import date, timedelta

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.models import Asset
from smart_albums.core.node import Stage, ConfigParam
from smart_albums.core.registry import stage


def _parse_last_shown(value: object) -> date | None:
    """Coerce a last_shown_date metadata value to a date object.

    Handles:
    - None → None
    - date instance → returned as-is
    - ISO-format string (YYYY-MM-DD) → parsed to date

    Returns None if the value cannot be parsed.
    """
    if value is None:
        return None
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            return None
    return None


@stage("avoid_repeat")
class AvoidRepeat(Stage):
    """Partition into fresh + backfilled; oldest-shown backfill ordering."""

    _config_schema = (
        ConfigParam(
            "days",
            int,
            default=30,
            min=0,
            description="Cooldown window in days. Assets shown within this window are ineligible.",
        ),
        ConfigParam(
            "photo_count",
            int,
            default=500,
            min=1,
            description="Target photo count for backfill calculation.",
        ),
        ConfigParam(
            "run_date",
            date,
            required=True,
            description="The current run date used to compute the freshness cutoff.",
        ),
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        days = self.get("days")
        if days <= 0:
            ctx.stats["avoid_repeat.fresh_count"] = len(ctx.assets)
            ctx.stats["avoid_repeat.backfilled_count"] = 0
            return [ctx]

        photo_count = self.get("photo_count")
        run_date = self.get("run_date")
        cutoff = run_date - timedelta(days=days)

        fresh: list[Asset] = []
        ineligible: list[Asset] = []
        for asset in ctx.assets:
            last_shown = _parse_last_shown(asset.metadata.get("last_shown_date"))
            if last_shown is None or last_shown <= cutoff:
                fresh.append(asset)
            else:
                ineligible.append(asset)

        backfill: list[Asset] = []
        if len(fresh) < photo_count:
            # Oldest-shown first, ties by lex-smallest id
            ineligible.sort(
                key=lambda a: (
                    _parse_last_shown(a.metadata.get("last_shown_date")) or date.min,
                    a.id,
                )
            )
            deficit = photo_count - len(fresh)
            backfill = ineligible[:deficit]
            for asset in backfill:
                asset.metadata["fallback_backfilled"] = True

        eligible_ids = {a.id for a in fresh + backfill}
        ctx.assets = [a for a in ctx.assets if a.id in eligible_ids]
        ctx.stats["avoid_repeat.fresh_count"] = len(fresh)
        ctx.stats["avoid_repeat.backfilled_count"] = len(backfill)
        return [ctx]
