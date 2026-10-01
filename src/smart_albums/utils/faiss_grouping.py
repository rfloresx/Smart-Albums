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

from smart_albums.utils.union_find import AnchoredUnionFind


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
        A dict mapping Union-Find root indices to lists of assets in that
        group. For 0 assets this is empty; for 1 asset it's a single
        one-member group.
    """
    n = len(assets)
    # Guard the degenerate sizes explicitly (CO-13). With n == 0 the old
    # ``embeddings.shape[1]`` raised IndexError (an empty array has no
    # second axis), and n == 1 did a pointless FAISS build. Callers
    # (partition.faiss/cosine) already special-case <2, but this makes the
    # shared helper safe to call directly.
    if n == 0:
        return {}
    if n == 1:
        return {0: [assets[0]]}

    raw_embeddings = [a.metadata["embedding"] for a in assets]
    # Validate that all embeddings share one dimensionality before building
    # the array. A ragged list (mismatched dims, e.g. embeddings from two
    # different models) otherwise produced a NumPy object-array or a
    # ValueError deep inside faiss with an opaque message (CO-13).
    dims = {len(e) for e in raw_embeddings}
    if len(dims) != 1:
        raise ValueError(
            f"group_by_embedding_similarity: embeddings have inconsistent "
            f"dimensions {sorted(dims)}; all assets must use the same embedding model"
        )
    if dims == {0}:
        raise ValueError(
            "group_by_embedding_similarity: embeddings are empty (dimension 0)"
        )

    embeddings = np.array(raw_embeddings, dtype=np.float32)

    # L2-normalize so inner product == cosine similarity
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    # Search: for each vector find up to k neighbors, filter by threshold
    k = min(n, 128)
    D, I = index.search(embeddings, k)

    # Plain single-linkage union-find lets A~B and B~C merge A and C into
    # one group however dissimilar A and C actually are — long chains of
    # gradually-drifting photos then collapse into one giant group, which
    # a downstream select.* stage reduces to a single survivor (ND-10).
    # AnchoredUnionFind requires a candidate to also match each group's
    # fixed anchor (not just the specific neighbor FAISS happened to pair
    # it with), bounding how far a group can drift — the same guarantee
    # partition.time_gps_anchor already provides for GPS clustering.
    def _similarity(i: int, j: int) -> bool:
        return bool(np.dot(embeddings[i], embeddings[j]) >= threshold)

    uf = AnchoredUnionFind(n)
    for i in range(n):
        for j_idx in range(k):
            j = int(I[i][j_idx])
            if j == i or j < 0:
                continue
            if D[i][j_idx] >= threshold:
                uf.try_union(i, j, _similarity)

    # Group by root
    groups: dict[int, list[Any]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(assets[i])

    return groups
