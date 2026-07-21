"""FAISS-based cosine similarity grouping using Union-Find.

Provides a shared utility for building a FAISS IndexFlatIP from L2-normalized
embedding vectors and using Union-Find to form transitive groups of assets
whose cosine similarity exceeds a given threshold.

Used by both ``partition.faiss`` and ``partition.cosine`` stages.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import faiss
import numpy as np

from smart_albums.utils.union_find import UnionFind


def group_by_embedding_similarity(
    assets: list[Any],
    threshold: float,
) -> dict[int, list[Any]]:
    """Build FAISS index and group assets by cosine similarity.

    Extracts embeddings from ``asset.metadata["embedding"]``, L2-normalizes
    them so that inner-product equals cosine similarity, builds a FAISS
    IndexFlatIP, and uses Union-Find to form transitive groups at the given
    threshold.

    Args:
        assets: Assets that have valid embeddings in ``metadata["embedding"]``.
            Must have at least 2 elements.
        threshold: Minimum cosine similarity to consider two assets as
            belonging to the same group.

    Returns:
        A dict mapping Union-Find root indices to lists of assets in that group.
    """
    n = len(assets)
    embeddings = np.array(
        [a.metadata["embedding"] for a in assets], dtype=np.float32
    )

    # L2-normalize so inner product == cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Search: for each vector find up to k neighbors, filter by threshold
    k = min(n, 128)
    D, I = index.search(embeddings, k)

    uf = UnionFind(n)
    for i in range(n):
        for j_idx in range(k):
            j = int(I[i][j_idx])
            if j == i or j < 0:
                continue
            if D[i][j_idx] >= threshold:
                uf.union(i, j)

    # Group by root
    groups: dict[int, list[Any]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(assets[i])

    return groups
