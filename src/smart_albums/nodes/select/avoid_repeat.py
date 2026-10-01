"""avoid_repeat — partition assets into fresh and backfilled pools."""

from __future__ import annotations

from datetime import date, datetime, timedelta

from smart_albums.core.context import PipelineContext, ContextBatch
from protocols_system.protocols import Asset
from smart_albums.core.node import Stage, ConfigParam
from smart_albums.core.registry import stage
from smart_albums.utils.history import get_last_shown


def _parse_last_shown(value: object) -> date | None:
    """Coerce a last_shown_date metadata value to a date object.

    Handles:
    - None → None
    - datetime instance → truncated to its date component
    - date instance → returned as-is
    - ISO-format date string (YYYY-MM-DD) → parsed to date
    - ISO-format datetime string (e.g. with a "T...Z" time component) →
      parsed to a date

    Returns None if the value cannot be parsed.

    Note (ND-11): ``datetime`` is a subclass of ``date``, so the previous
    ``isinstance(value, date)`` check matched a ``datetime`` and returned
    it unchanged. Comparing that returned ``datetime`` against a plain
    ``date`` cutoff elsewhere in this module (``last_shown <= cutoff``)
    then raised ``TypeError: can't compare datetime.datetime to
    datetime.date``, so any caller writing a full timestamp instead of a
    bare date crashed the whole stage. Similarly, ``date.fromisoformat``
    rejects an ISO *datetime* string (one with a time component), so
    those were silently treated as "never shown" instead of being parsed.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value)
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(value).date()
        except ValueError:
            return None
    return None


def _parse_run_date(value: object) -> date:
    """Coerce the ``run_date`` config value to a ``date``.

    ``ConfigParam`` type coercion (CO-09) is not guaranteed to run before
    a stage receives its config — e.g. a value supplied directly via a
    JSON pipeline config or a scheduled job's ``pipeline_settings`` arrives
    as a plain string. Accepting a string here (ND-11) means ``run_date``
    behaves the same way as ``last_shown_date`` instead of raising
    ``TypeError`` from the ``run_date - timedelta(...)`` subtraction as
    soon as a string slips through.

    Raises:
        TypeError: if ``value`` is not a ``date``/``datetime`` and not a
            parseable ISO date/datetime string.
    """
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        parsed = _parse_last_shown(value)
        if parsed is not None:
            return parsed
    raise TypeError(
        f"'run_date' must be a date, datetime, or ISO date/datetime string; got {value!r}"
    )


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
        run_date = _parse_run_date(self.get("run_date"))
        cutoff = run_date - timedelta(days=days)

        def _last_shown_for(asset: Asset) -> date | None:
            """Resolve an asset's last-shown date.

            Prefers an explicit ``metadata["last_shown_date"]`` (set by a
            caller that tracks history itself), falling back to the shared
            publish-history cache written by the publish stages (ND-11).
            Without this fallback, nothing in the pipeline ever populated
            ``last_shown_date``, so this stage filtered nothing and was
            effectively a no-op.
            """
            raw = asset.metadata.get("last_shown_date")
            if raw is None:
                raw = get_last_shown(asset.id)
            return _parse_last_shown(raw)

        fresh: list[Asset] = []
        ineligible: list[Asset] = []
        for asset in ctx.assets:
            last_shown = _last_shown_for(asset)
            if last_shown is None or last_shown <= cutoff:
                fresh.append(asset)
            else:
                ineligible.append(asset)

        backfill: list[Asset] = []
        if len(fresh) < photo_count:
            # Oldest-shown first, ties by lex-smallest id
            ineligible.sort(
                key=lambda a: (
                    _last_shown_for(a) or date.min,
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
