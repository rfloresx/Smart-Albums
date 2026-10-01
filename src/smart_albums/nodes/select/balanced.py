"""Balanced selection stage — two-phase monthly quota fill + quality top-up."""

from __future__ import annotations

import logging
import math
from collections import defaultdict

from smart_albums.core.context import PipelineContext, ContextBatch
from protocols_system.protocols import Asset
from smart_albums.core.node import Stage, ConfigParam
from smart_albums.core.registry import stage

logger = logging.getLogger(__name__)


@stage("select.balanced")
class SelectBalanced(Stage):
    """Two-phase: per-month quota fill, then quality top-up."""

    _config_schema = (
        ConfigParam("min_photos", int, default=300, min=1),
        ConfigParam("max_photos", int, default=500, min=1),
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        min_photos = self.get("min_photos")
        max_photos = self.get("max_photos")

        if min_photos > max_photos:
            logger.warning(
                "select.balanced: min_photos (%d) > max_photos (%d), clamping min to max.",
                min_photos, max_photos,
            )
            min_photos = max_photos

        selected = _two_phase_select(ctx.assets, min_photos, max_photos)
        selected_ids = {a.id for a in selected}

        for asset in ctx.assets:
            if asset.id in selected_ids:
                asset.metadata["selected"] = True

        ctx.assets = [a for a in ctx.assets if a.id in selected_ids]
        ctx.stats["select.balanced.selected"] = len(selected)
        ctx.stats["select.balanced.warning_below_min"] = len(ctx.assets) < min_photos
        return [ctx]


def _two_phase_select(
    qualifying: list[Asset], min_photos: int, max_photos: int
) -> list[Asset]:
    """Phase 1: monthly quota fill; Phase 2: quality top-up."""
    # Group by month (year-month). Untimed assets are tracked separately:
    # "_untimed" is not a real month, so counting it toward the month total
    # shrank every real month's quota and let undated photos claim a
    # balancing slot they shouldn't (ND-20). They still participate in the
    # score-based top-up via the remaining pool below.
    _UNTIMED = "_untimed"
    monthly: dict[str, list[Asset]] = defaultdict(list)
    for asset in qualifying:
        key = asset.captured_at.strftime("%Y-%m") if asset.captured_at is not None else _UNTIMED
        monthly[key].append(asset)

    # Sort each month by score descending
    for month_assets in monthly.values():
        month_assets.sort(key=lambda a: a.metadata.get("score", 0.0), reverse=True)

    real_months = [k for k in monthly if k != _UNTIMED]
    non_empty_months = len(real_months)
    if non_empty_months == 0 and _UNTIMED not in monthly:
        return []

    # Phase 1: fill monthly quotas.
    #
    # ``floor(min_photos / months)`` becomes 0 as soon as there are more
    # non-empty months than min_photos — Phase 1 then selected nothing and
    # the whole result was decided by the score-only top-up, defeating the
    # point of balancing (ND-20). Guarantee at least one photo per real
    # month so every month is represented, even if that means slightly
    # overshooting min_photos (Phase 2 is capped by max_photos regardless).
    quota = max(1, math.floor(min_photos / non_empty_months)) if non_empty_months else 0
    selected: list[Asset] = []
    remaining_pool: list[Asset] = []

    for key, month_assets in monthly.items():
        if key == _UNTIMED:
            # Untimed assets don't get a reserved per-month quota; they
            # compete purely on score in the top-up phase.
            remaining_pool.extend(month_assets)
            continue
        selected.extend(month_assets[:quota])
        remaining_pool.extend(month_assets[quota:])

    # If we haven't hit min_photos yet, fill from remaining by score
    remaining_pool.sort(key=lambda a: a.metadata.get("score", 0.0), reverse=True)
    deficit = min_photos - len(selected)
    if deficit > 0:
        selected.extend(remaining_pool[:deficit])
        remaining_pool = remaining_pool[deficit:]

    # Phase 2: quality top-up to max_photos
    top_up_count = max_photos - len(selected)
    if top_up_count > 0:
        selected.extend(remaining_pool[:top_up_count])

    # With the "at least 1 per month" guarantee above, a library spanning
    # more than max_photos months could overshoot the cap in Phase 1
    # alone. Trim to max_photos, keeping the highest-scoring picks so the
    # cap is still honored while preserving month coverage as far as the
    # budget allows (ND-20).
    if len(selected) > max_photos:
        selected.sort(key=lambda a: a.metadata.get("score", 0.0), reverse=True)
        selected = selected[:max_photos]

    return selected
