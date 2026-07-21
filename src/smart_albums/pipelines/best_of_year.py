"""Best-of-year pipeline definition.

Pipeline execution model
------------------------
The pipeline is a flat list executed sequentially. Partition stages (e.g.
``PartitionTimeGpsAnchor``) split contexts into multiple sub-contexts.
Subsequent stages operate on ALL sub-contexts independently until a
``MergeConcat`` step collapses them back into their parent level.

The indentation in the pipeline list below is purely cosmetic — it indicates
the *logical* nesting:

    partition_stage      ← splits into N sub-contexts
        inner_stage      ← runs on each sub-context
        merge_stage      ← collapses sub-contexts back to parent

Each ``MergeConcat`` reduces partition depth by one level by grouping
sub-contexts that share the same ``parent_partition_id``.

IMPORTANT: Every partition stage MUST be followed (at the correct depth)
by a matching ``MergeConcat``. Forgetting a merge will leave contexts
fragmented for all downstream stages.
"""

from __future__ import annotations

from smart_albums.core.spec import Pipeline, alias
from smart_albums.nodes.fork.fork import ForkBySelection, Branch

from smart_albums.nodes.retrieve.by_year import RetrieveByYear
from smart_albums.nodes.filter.videos import RetainImages as FilterNonImages
from smart_albums.nodes.filter.min_score import FilterMinScore
from smart_albums.nodes.filter.screenshots import FilterScreenshots

from smart_albums.nodes.partition.time_gps_anchor import PartitionTimeGpsAnchor
from smart_albums.nodes.partition.time import PartitionTime
from smart_albums.nodes.partition.phash import PartitionPHash
from smart_albums.nodes.partition.faiss import PartitionFaiss
from smart_albums.nodes.partition.cosine import PartitionCosine
from smart_albums.nodes.select.best import SelectBest
from smart_albums.nodes.select.diverse import SelectDiversePick
from smart_albums.nodes.select.balanced import SelectBalanced
from smart_albums.nodes.analyze.embedding import AnalyzeEmbedding
from smart_albums.nodes.analyze.phash import AnalyzePHash
from smart_albums.nodes.analyze.score import AnalyzeScore
from smart_albums.nodes.merge.merge import MergeConcat

from smart_albums.nodes.publish.create_album import PublishCreateAlbum
from smart_albums.pipelines.registry import register_pipeline

# --- Sub-pipelines for the fork ---

# The main curation pipeline (produces the "best of" selection)
_curation_pipeline: Pipeline = [
    # Partitioning by time + GPS (two-stage: time clusters then GPS anchor)
    alias("events", PartitionTimeGpsAnchor),
        # Stage 2: pHash exact dup
        alias("duplicate", PartitionPHash),
            alias("duplicate.pick_best", SelectBest),
            alias("duplicate.merge", MergeConcat),
        # Stage 3: Generate Embedding
        alias("compute_embedding", AnalyzeEmbedding),
        # Stage 4: Near-Dup Search (FAISS cosine similarity)
        alias("near_duplicate", PartitionFaiss),
            alias("near_duplicate.pick_best", SelectBest),
            alias("near_duplicate.merge", MergeConcat),
        alias("events.merge", MergeConcat),

    # Stage 5: Scene Clustering (cosine similarity)
    alias("scenes", PartitionTime),
        alias("scene_cluster", PartitionCosine),
            # Diversity-aware selection
            alias("scene_cluster.pick", SelectDiversePick),
            alias("scene_cluster.merge", MergeConcat),
        alias("scenes.merge", MergeConcat),

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

