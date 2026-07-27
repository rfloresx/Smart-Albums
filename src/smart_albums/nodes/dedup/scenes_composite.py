"""dedup.scenes — scene-based deduplication with diversity selection (composite node).

Wraps the scene clustering pattern into a single reusable node:

    AnalyzeEmbedding → PartitionTime → PartitionCosine → SelectDiversePick → MergeConcat → MergeConcat

Exposed config (all keys are unique across children):
  - concurrency (int): from AnalyzeEmbedding — max concurrent fetches
  - batch_size (int): from AnalyzeEmbedding — images per embed batch
  - time_window_minutes (float): from PartitionTime — temporal clustering window
  - threshold (float): from PartitionCosine — cosine similarity threshold
  - max_picks (int): from SelectDiversePick — max assets per scene
  - pick_percentage (float): from SelectDiversePick — fraction of group to keep
  - quality_weight (float): from SelectDiversePick — MMR quality weight
  - diversity_weight (float): from SelectDiversePick — MMR diversity weight
"""

from __future__ import annotations

from smart_albums.core.node import CompositeNode
from smart_albums.core.registry import stage
from smart_albums.core.spec import alias

from smart_albums.nodes.analyze.embedding import AnalyzeEmbedding
from smart_albums.nodes.partition.time import PartitionTime
from smart_albums.nodes.partition.cosine import PartitionCosine
from smart_albums.nodes.select.diverse import SelectDiversePick
from smart_albums.nodes.merge.merge import MergeConcat


@stage("dedup.scenes")
class DedupScenes(CompositeNode):
    """Scene-based deduplication with diversity-aware selection.

    Internally runs:
      1. AnalyzeEmbedding — compute vector embeddings
      2. PartitionTime — group by temporal proximity
      3. PartitionCosine — sub-group by embedding similarity (scenes)
      4. SelectDiversePick — keep diverse picks per scene (MMR)
      5. MergeConcat — collapse cosine partitions
      6. MergeConcat — collapse time partitions

    Config:
      - concurrency (int, default=4): Max concurrent thumbnail/embed operations.
      - batch_size (int, default=16): Images per embedding batch.
      - time_window_minutes (float, default=5.0): Time clustering window.
      - threshold (float, default=0.85): Cosine similarity threshold for scenes.
      - max_picks (int, default=3): Max assets to pick per scene.
      - pick_percentage (float, default=0.33): Fraction of scene assets to pick.
      - quality_weight (float, default=0.3): MMR quality weight.
      - diversity_weight (float, default=0.7): MMR diversity weight.
    """

    _sub_pipeline = [
        alias("embedding", AnalyzeEmbedding),
        alias("time", PartitionTime),
            alias("cosine", PartitionCosine),
                alias("select", SelectDiversePick),
                alias("cosine_merge", MergeConcat),
            alias("time_merge", MergeConcat),
    ]
