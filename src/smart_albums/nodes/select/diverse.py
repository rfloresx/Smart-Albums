"""select.diverse_pick — select a diverse subset of assets using MMR.

Uses Maximal Marginal Relevance to pick assets that balance quality (score)
with diversity (embedding distance from already-selected picks). The first
pick is always the highest-quality asset; subsequent picks maximize a
weighted combination of normalized quality and embedding dissimilarity.
"""

from __future__ import annotations

import math

import numpy as np

from smart_albums.core.context import PipelineContext, ContextBatch
from smart_albums.core.models import Asset
from smart_albums.core.node import Stage, ConfigParam
from smart_albums.core.registry import stage


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors via L2-normalized dot product."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a / norm_a, b / norm_b))


def _mmr_select(
    assets: list[Asset],
    pick_limit: int,
    quality_weight: float,
    diversity_weight: float,
) -> list[Asset]:
    """Select assets using Maximal Marginal Relevance.

    Args:
        assets: Assets to select from (must have len >= 2).
        pick_limit: Maximum number of assets to pick.
        quality_weight: Weight for the normalized quality score component.
        diversity_weight: Weight for the diversity (1 - max_similarity) component.

    Returns:
        Ordered list of selected assets. First element is highest quality.
    """
    # Extract scores
    scores = [a.metadata.get("score", 0.0) for a in assets]

    # Normalize scores to [0, 1] via min-max
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score
    if score_range > 0:
        normalized_scores = [(s - min_score) / score_range for s in scores]
    else:
        normalized_scores = [1.0] * len(scores)

    # Check if we have embeddings for diversity-based selection
    embeddings: list[np.ndarray | None] = []
    has_embeddings = False
    for a in assets:
        emb = a.metadata.get("embedding")
        if emb is not None:
            embeddings.append(np.array(emb, dtype=np.float32))
            has_embeddings = True
        else:
            embeddings.append(None)

    # First pick: highest quality score, tie-break by smallest id
    best_idx = max(
        range(len(assets)),
        key=lambda i: (scores[i], assets[i].id),
    )
    # For tie-breaking: we want highest score, then smallest id
    max_score_val = scores[best_idx]
    candidates_for_first = [
        i for i in range(len(assets)) if scores[i] == max_score_val
    ]
    best_idx = min(candidates_for_first, key=lambda i: assets[i].id)

    picks: list[int] = [best_idx]
    remaining: set[int] = set(range(len(assets))) - {best_idx}

    # If no embeddings, fall back to quality-only selection
    if not has_embeddings:
        # Sort remaining by quality descending, tie-break smallest id
        sorted_remaining = sorted(
            remaining,
            key=lambda i: (-normalized_scores[i], assets[i].id),
        )
        for idx in sorted_remaining[: pick_limit - 1]:
            picks.append(idx)
        return [assets[i] for i in picks]

    # Precompute normalized embedding vectors
    norm_embeddings: list[np.ndarray | None] = []
    for emb in embeddings:
        if emb is not None:
            norm = np.linalg.norm(emb)
            if norm > 0:
                norm_embeddings.append(emb / norm)
            else:
                norm_embeddings.append(None)
        else:
            norm_embeddings.append(None)

    # MMR iteration
    while len(picks) < pick_limit and remaining:
        best_candidate = None
        best_mmr_score = -float("inf")

        for cand_idx in remaining:
            # Quality component
            q_score = normalized_scores[cand_idx]

            # Diversity component: 1 - max similarity to already picked
            cand_emb = norm_embeddings[cand_idx]
            if cand_emb is not None:
                max_sim = 0.0
                for pick_idx in picks:
                    pick_emb = norm_embeddings[pick_idx]
                    if pick_emb is not None:
                        sim = float(np.dot(cand_emb, pick_emb))
                        if sim > max_sim:
                            max_sim = sim
                diversity = 1.0 - max_sim
            else:
                # No embedding for candidate: treat as maximally diverse
                diversity = 1.0

            mmr_score = quality_weight * q_score + diversity_weight * diversity

            # Tie-break: higher mmr_score wins, then smallest id
            if (mmr_score > best_mmr_score) or (
                mmr_score == best_mmr_score
                and (
                    best_candidate is None
                    or assets[cand_idx].id < assets[best_candidate].id
                )
            ):
                best_mmr_score = mmr_score
                best_candidate = cand_idx

        if best_candidate is not None:
            picks.append(best_candidate)
            remaining.discard(best_candidate)
        else:
            break  # pragma: no cover

    return [assets[i] for i in picks]


@stage("select.diverse_pick")
class SelectDiversePick(Stage):
    """Select a diverse subset of assets using Maximal Marginal Relevance.

    Operates on a single partition context containing a group of assets.
    The first pick is the highest-quality asset (by score). Subsequent picks
    maximize a weighted combination of normalized quality and embedding
    dissimilarity from already-selected picks.

    Sets ``metadata["group_pick"] = True`` on each selected asset and
    ``metadata["group_best"] = True`` on the first (highest quality) pick.
    Narrows the context assets to just the selected picks.
    """

    _config_schema = (
        ConfigParam(
            key="max_picks",
            type=int,
            default=3,
            description="Maximum number of assets to pick from each group.",
        ),
        ConfigParam(
            key="pick_percentage",
            type=float,
            default=0.33,
            description="Fraction of group assets to pick (capped by max_picks).",
        ),
        ConfigParam(
            key="quality_weight",
            type=float,
            default=0.3,
            description="Weight for quality score in MMR combined score.",
        ),
        ConfigParam(
            key="diversity_weight",
            type=float,
            default=0.7,
            description="Weight for diversity (embedding distance) in MMR combined score.",
        ),
    )

    async def run(self, ctx: PipelineContext) -> ContextBatch:
        if not ctx.assets:
            ctx.stats["groups_processed"] = 0
            ctx.stats["picks_selected"] = 0
            return [ctx]

        if len(ctx.assets) < 2:
            ctx.assets[0].metadata["group_pick"] = True
            ctx.assets[0].metadata["group_best"] = True
            ctx.stats["groups_processed"] = 1
            ctx.stats["picks_selected"] = 1
            return [ctx]

        max_picks = self.get("max_picks")
        pick_pct = self.get("pick_percentage")
        quality_w = self.get("quality_weight")
        diversity_w = self.get("diversity_weight")

        pick_limit = min(max_picks, max(1, math.ceil(pick_pct * len(ctx.assets))))

        picks = _mmr_select(ctx.assets, pick_limit, quality_w, diversity_w)

        for pick in picks:
            pick.metadata["group_pick"] = True
        picks[0].metadata["group_best"] = True

        ctx.assets = picks
        ctx.stats["groups_processed"] = 1
        ctx.stats["picks_selected"] = len(picks)
        return [ctx]
