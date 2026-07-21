"""Rotation pipeline — nightly photo-frame album update.

Replicates the original RotationEngine as a declarative pipeline definition.
The pipeline retrieves a scoped candidate pool, applies content filtering,
scores for quality, deduplicates, enforces freshness via avoid-repeat, selects
a temporally balanced subset, and replaces a fixed album's contents via
add/remove diff.

Designed to be scheduled daily (e.g. via cron) to keep a digital photo frame
album fresh. The album is created on first run and updated in-place thereafter.

Pipeline stages:
    1. source          — retrieve candidates (library/year/date_range/recent/album)
    2. filter          — content filter (none/smart_search/on_this_day/people/random)
    3. filter_images   — drop videos, keep images only
    4. sensitive       — sensitive-content exclusion
    5. score           — AI quality scoring (cached)
    6. min_score       — quality gate
    7. phash           — compute perceptual hash
    8. dedup           — collapse near-duplicate groups
    9. avoid_repeat    — freshness cooldown with oldest-shown backfill
   10. balanced        — temporally balanced final selection
   11. publish         — replace album contents (diff-based)
"""

from __future__ import annotations

from smart_albums.core.spec import Pipeline, alias

from smart_albums.nodes.retrieve.source import Source
from smart_albums.nodes.filter.videos import RetainImages
from smart_albums.nodes.filter.none import FilterNone
from smart_albums.nodes.filter.sensitive import FilterSensitive
from smart_albums.nodes.filter.min_score import FilterMinScore
from smart_albums.nodes.analyze.score import AnalyzeScore
from smart_albums.nodes.analyze.phash import AnalyzePHash
from smart_albums.nodes.dedup.phash import DedupPHash
from smart_albums.nodes.select.avoid_repeat import AvoidRepeat
from smart_albums.nodes.select.balanced import SelectBalanced
from smart_albums.nodes.publish.replace_album import PublishReplaceAlbum

from smart_albums.pipelines.registry import register_pipeline

rotation: Pipeline = [
    # 1. Retrieve candidates from configured source
    alias("source", Source),
    # 2. Content filter (default: none/passthrough, overridable per recipe)
    alias("filter", FilterNone),
    # 3. Drop videos, keep images only
    alias("filter_images", RetainImages),
    # 4. Sensitive-content exclusion
    alias("sensitive", FilterSensitive),
    # 5. AI quality scoring (cache-first)
    alias("score", AnalyzeScore),
    # 6. Quality gate — exclude assets below min_score
    alias("min_score", FilterMinScore),
    # 7. Compute perceptual hash for deduplication
    alias("phash", AnalyzePHash),
    # 8. Collapse near-duplicate groups, keep best
    alias("dedup", DedupPHash),
    # 9. Freshness cooldown — skip recently shown, backfill oldest-shown
    alias("avoid_repeat", AvoidRepeat),
    # 10. Temporally balanced final selection
    alias("balanced", SelectBalanced),
    # 11. Replace the photo-frame album contents
    alias("publish", PublishReplaceAlbum),
]

@register_pipeline(
    "rotation",
    label="Rotation",
    description="Run the rotation pipeline (nightly photo-frame album update).",
)
def _rotation_factory() -> Pipeline:
    return rotation
