"""Shared Union-Find (disjoint-set) data structure.

Used by partition and deduplication stages to transitively group assets
by similarity metrics (pHash Hamming distance, cosine similarity, etc.).
"""

from __future__ import annotations


class UnionFind:
    """Disjoint-set with path compression and union by rank.

    Provides near-constant amortized time for find and union operations.

    Args:
        n: Number of elements (indexed 0..n-1).
    """

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        """Find the root representative of element x with path compression."""
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        """Merge the sets containing x and y (union by rank)."""
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
