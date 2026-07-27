"""dedup.similar — deduplicate by embedding similarity (composite node).

Wraps the near-duplicate-by-embedding pattern into a single reusable node:

    AnalyzeEmbedding → PartitionTimeGpsAnchor → PartitionFaiss → SelectBest → MergeConcat → MergeConcat

The two merges correspond to collapsing the PartitionFaiss level first,
then the PartitionTimeGpsAnchor level.

Exposed config (auto-derived from children):
  - concurrency (int): from AnalyzeEmbedding — max concurrent fetches
  - batch_size (int): from AnalyzeEmbedding — images per embed batch
  - time_window_minutes (float): from PartitionTimeGpsAnchor
  - gps_window_meters (float): from PartitionTimeGpsAnchor
  - threshold (float): from PartitionFaiss — cosine similarity threshold
"""

from __future__ import annotations

from smart_albums.core.node import CompositeNode
from smart_albums.core.registry import stage
from smart_albums.core.spec import alias

from smart_albums.nodes.analyze.embedding import AnalyzeEmbedding
from smart_albums.nodes.partition.time_gps_anchor import PartitionTimeGpsAnchor
from smart_albums.nodes.partition.faiss import PartitionFaiss
from smart_albums.nodes.select.best import SelectBest
from smart_albums.nodes.merge.merge import MergeConcat


@stage("dedup.similar")
class DedupSimilar(CompositeNode):
    """Deduplicate near-similar images by embedding cosine similarity.

    Internally runs:
      1. AnalyzeEmbedding — compute vector embeddings
      2. PartitionTimeGpsAnchor — group by time+location to limit comparisons
      3. PartitionFaiss — sub-group by embedding similarity
      4. SelectBest — keep best representative per group
      5. MergeConcat — collapse faiss partitions
      6. MergeConcat — collapse time/gps partitions

    Config:
      - concurrency (int, default=4): Max concurrent thumbnail/embed operations.
      - batch_size (int, default=16): Images per embedding batch.
      - time_window_minutes (float, default=5.0): Time clustering window.
      - gps_window_meters (float, default=100.0): GPS anchor distance.
      - threshold (float, default=0.95): FAISS cosine similarity threshold.
    """

    _sub_pipeline = [
        alias("embedding", AnalyzeEmbedding),
        alias("time_gps", PartitionTimeGpsAnchor),
            alias("faiss", PartitionFaiss),
                alias("select", SelectBest),
                alias("faiss_merge", MergeConcat),
            alias("time_gps_merge", MergeConcat),
    ]
