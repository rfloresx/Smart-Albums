"""dedup.phash_composite — deduplicate by perceptual hash (composite node).

Wraps the common dedup-by-pHash pattern into a single reusable node:

    AnalyzePHash → PartitionPHash → SelectBest → MergeConcat

Exposed config:
  - concurrency (int): max concurrent thumbnail fetches (from AnalyzePHash)
  - threshold (float): pHash similarity threshold (from PartitionPHash)
"""

from __future__ import annotations

from smart_albums.core.node import CompositeNode
from smart_albums.core.registry import stage
from smart_albums.core.spec import alias

from smart_albums.nodes.analyze.phash import AnalyzePHash
from smart_albums.nodes.partition.phash import PartitionPHash
from smart_albums.nodes.select.best import SelectBest
from smart_albums.nodes.merge.merge import MergeConcat


@stage("dedup.phash_composite")
class DedupPHashComposite(CompositeNode):
    """Deduplication by perceptual hash — composite node.

    Internally runs: AnalyzePHash → PartitionPHash → SelectBest → MergeConcat

    This is equivalent to using the four nodes individually in a pipeline but
    packaged as a single step for convenience and reusability.

    Config:
      - concurrency (int, default=4): Maximum concurrent thumbnail fetches.
      - threshold (float, default=0.90): pHash Hamming similarity threshold.
    """

    _sub_pipeline = [
        alias("analyze", AnalyzePHash),
        alias("partition", PartitionPHash),
        alias("select", SelectBest),
        alias("merge", MergeConcat),
    ]
