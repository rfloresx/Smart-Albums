"""Best-of-year pipeline definition.

Pipeline execution model
------------------------
The pipeline is a flat list executed sequentially. Composite nodes
encapsulate common partition→select→merge patterns into single reusable
steps, keeping the pipeline definition concise.

Composite nodes used:
- DedupSimilar: AnalyzeEmbedding → PartitionTimeGpsAnchor → PartitionFaiss → SelectBest → MergeConcat × 2
- DedupScenes: AnalyzeEmbedding → PartitionTime → PartitionCosine → SelectDiversePick → MergeConcat × 2

Both composites include AnalyzeEmbedding which no-ops for assets that
already have embeddings computed.
"""

from __future__ import annotations

from smart_albums.core.spec import Pipeline, alias
from smart_albums.nodes.fork.fork import ForkBySelection, Branch

from smart_albums.nodes.retrieve.by_year import RetrieveByYear
from smart_albums.nodes.filter.videos import RetainImages as FilterNonImages
from smart_albums.nodes.filter.min_score import FilterMinScore
from smart_albums.nodes.filter.screenshots import FilterScreenshots

from smart_albums.nodes.analyze.phash import AnalyzePHash
from smart_albums.nodes.analyze.score import AnalyzeScore

from smart_albums.nodes.dedup.phash_composite import DedupPHashComposite
from smart_albums.nodes.dedup.similar_composite import DedupSimilar
from smart_albums.nodes.dedup.scenes_composite import DedupScenes

from smart_albums.nodes.select.balanced import SelectBalanced
from smart_albums.nodes.publish.create_album import PublishCreateAlbum
from smart_albums.pipelines.registry import register_pipeline

# --- Sub-pipelines for the fork ---

# The main curation pipeline (produces the "best of" selection)
_curation_pipeline: Pipeline = [
    # Stage 1: pHash exact-duplicate removal
    alias("duplicate", DedupPHashComposite),
    # Stage 2: Near-duplicate removal (embedding + time/GPS + FAISS)
    alias("near_duplicate", DedupSimilar),
    # Stage 3: Scene clustering with diversity-aware selection
    alias("scenes", DedupScenes),
    # Stage 4: Temporal balancing
    alias("balanced", SelectBalanced),
    alias("publish", PublishCreateAlbum),
]

# Pipeline for images NOT selected by the curation pipeline
_remainder_pipeline: Pipeline = [
    alias("publish", PublishCreateAlbum),
]

best_of_year: Pipeline = [
    alias("retrieve", RetrieveByYear),
    alias("filter_images", FilterNonImages),
    alias("score", AnalyzeScore),
    alias("phash", AnalyzePHash),
    alias("min_score", FilterMinScore),
    alias("filter_screenshots", FilterScreenshots),
    # Fork: curation pipeline selects the best; remainder gets the rest
    alias("output", ForkBySelection(
        Branch(name="curation", pipeline=_curation_pipeline),
        Branch(name="remainder", pipeline=_remainder_pipeline),
    )),
]

@register_pipeline(
    "best-of-year",
    label="Best of Year",
    description="Creates a curated 'Best of [Year]' album with AI scoring, deduplication, and temporal balancing.",
)
def _best_of_year_factory() -> Pipeline:
    return best_of_year
