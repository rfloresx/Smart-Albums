"""Shared Union-Find (disjoint-set) data structure.

Used by partition and deduplication stages to transitively group assets
by similarity metrics (pHash Hamming distance, cosine similarity, etc.).
"""

from __future__ import annotations

from collections.abc import Callable


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


class AnchoredUnionFind(UnionFind):
    """Union-Find variant that bounds chaining by comparing against a
    group *anchor* rather than only the immediate neighbor pair (ND-10).

    Plain single-linkage union-find lets A~B and B~C merge A and C into one
    group even when A and C are nowhere near each other — at a typical
    scene-clustering cosine threshold of 0.85, this lets long chains of
    gradually-drifting photos collapse into one giant group, which a
    downstream ``select.*`` stage then reduces to a single survivor,
    discarding everything else in the chain.

    Each group tracks a fixed anchor element (the first element added to
    it). A candidate is only merged into a group if it is within threshold
    of *that anchor* — not just of whichever member happened to be compared
    last — so a group's total "diameter" is bounded by the similarity
    threshold itself, the same guarantee ``time_gps_anchor`` already
    provides for its own GPS clustering.

    Args:
        n: Number of elements (indexed 0..n-1).
    """

    def __init__(self, n: int) -> None:
        super().__init__(n)
        self.anchor = list(range(n))

    def try_union(self, x: int, y: int, similarity_fn: Callable[[int, int], bool]) -> bool:
        """Merge x and y's groups only if each anchor accepts the other.

        ``similarity_fn(i, j)`` must return True when elements i and j are
        considered the same group (already thresholded by the caller).

        Returns True if a merge happened (or x and y were already in the
        same group), False if the anchor check rejected the merge.
        """
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return True

        anchor_x = self.anchor[rx]
        anchor_y = self.anchor[ry]
        # Require the new member to also match the *other* group's anchor,
        # not just the specific neighbor that triggered the comparison.
        if not similarity_fn(anchor_x, y) or not similarity_fn(anchor_y, x):
            return False

        # Keep the smaller-index anchor so results stay deterministic
        # regardless of which side `union` promotes to root.
        kept_anchor = min(anchor_x, anchor_y)
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        self.anchor[rx] = kept_anchor
        return True
