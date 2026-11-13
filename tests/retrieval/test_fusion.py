"""Reciprocal rank fusion (pure)."""

import pytest

from kinship.rag.fusion import RRF_K, reciprocal_rank_fusion


def test_scores_are_summed_reciprocal_ranks() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]])

    assert [key for key, _ in fused] == ["a", "b", "c", "d"]
    scores = dict(fused)
    assert scores["a"] == pytest.approx(1 / (RRF_K + 1) + 1 / (RRF_K + 2))
    assert scores["b"] == pytest.approx(scores["a"])
    assert scores["c"] == pytest.approx(1 / (RRF_K + 3))
    assert scores["d"] == pytest.approx(scores["c"])


def test_a_key_in_both_rankings_beats_the_top_of_one() -> None:
    both_but_low = ["x1", "x2", "x3", "shared"]
    fused = reciprocal_rank_fusion([["only-fts", "shared"], both_but_low])
    assert fused[0][0] == "shared"


def test_ties_break_by_best_rank_then_earliest_ranking() -> None:
    # "a" and "b" both score 1/(k+1); "a" is in the earlier ranking.
    assert [k for k, _ in reciprocal_rank_fusion([["a"], ["b"]])] == ["a", "b"]
    assert [k for k, _ in reciprocal_rank_fusion([["b"], ["a"]])] == ["b", "a"]
    # Equal totals: (1, 4) vs (2, 3) is not a tie, but (1, 2) vs (2, 1) is.
    fused = reciprocal_rank_fusion([["p", "q"], ["q", "p"]])
    assert [k for k, _ in fused] == ["p", "q"]
    assert fused[0][1] == pytest.approx(fused[1][1])


def test_is_deterministic_and_does_not_compare_keys() -> None:
    rankings = [[("t", 1), ("t", 2)], [object(), ("t", 2)]]
    assert reciprocal_rank_fusion(rankings) == reciprocal_rank_fusion(rankings)


def test_duplicates_in_one_ranking_count_once_at_the_best_rank() -> None:
    assert reciprocal_rank_fusion([["a", "b", "a"]]) == reciprocal_rank_fusion([["a", "b"]])


def test_k_changes_the_scores_and_empty_input_is_fine() -> None:
    assert reciprocal_rank_fusion([["a"]], k=0) == [("a", 1.0)]
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], k=-1)
