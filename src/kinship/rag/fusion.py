"""Reciprocal rank fusion: merge several ranked lists into one. Pure functions."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

RRF_K = 60
"""The usual constant: it flattens the gap between neighbouring top ranks."""


def reciprocal_rank_fusion[K: Hashable](
    rankings: Sequence[Sequence[K]], k: int = RRF_K
) -> list[tuple[K, float]]:
    """Fuse `rankings` (each best-first) into `(key, score)` pairs, best first.

    A key scores `1 / (k + rank)` in every ranking that contains it (rank is
    1-based) and the scores are summed, so a key that several rankings agree on
    beats a key only one of them found. A key repeated inside one ranking
    counts once, at its best rank.

    Ties are broken deterministically without comparing keys: by the key's best
    rank in any ranking, then by the earliest ranking that contains it.
    """
    if k < 0:
        raise ValueError("k must not be negative")
    scores: dict[K, float] = {}
    tie_break: dict[K, tuple[int, int]] = {}
    for index, ranking in enumerate(rankings):
        seen: set[K] = set()
        for rank, key in enumerate(ranking, start=1):
            if key in seen:
                continue
            seen.add(key)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            tie_break[key] = min(tie_break.get(key, (rank, index)), (rank, index))
    return sorted(scores.items(), key=lambda item: (-item[1], tie_break[item[0]]))
