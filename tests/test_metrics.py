import math

import pytest
from scipy.stats import binomtest

from evaluation.metrics import (
    aggregate,
    first_relevant_rank,
    hit_at_k,
    latency_summary,
    mcnemar_exact,
    ndcg_at_k,
    paired_bootstrap_ci,
    question_metrics,
    reciprocal_rank,
    win_loss,
)


def test_first_relevant_rank():
    assert first_relevant_rank([5, 7, 9], {7}) == 2
    assert first_relevant_rank([5, 7, 9], {9, 5}) == 1
    assert first_relevant_rank([5, 7, 9], {4}) is None
    assert first_relevant_rank(list(range(20)), {12}) is None
    assert first_relevant_rank(list(range(20)), {12}, k=13) == 13


def test_hits_and_reciprocal_ranks_hand_computed():
    ranks = [1, 2, None, 4]

    assert sum(hit_at_k(r, 1) for r in ranks) / 4 == 0.25
    assert sum(hit_at_k(r, 3) for r in ranks) / 4 == 0.5
    assert sum(hit_at_k(r, 5) for r in ranks) / 4 == 0.75
    assert sum(reciprocal_rank(r) for r in ranks) / 4 == pytest.approx(0.4375)
    assert reciprocal_rank(11) == 0.0


def test_ndcg_single_relevant_at_rank_two():
    assert ndcg_at_k([3, 8, 1], {8}) == pytest.approx(1 / math.log2(3))
    assert ndcg_at_k([3, 8, 1], {8}) == pytest.approx(0.6309, abs=1e-4)


def test_ndcg_two_relevant_at_ranks_one_and_three():
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))

    assert ndcg_at_k([4, 6, 2], {4, 2}) == pytest.approx(expected)
    assert ndcg_at_k([4, 6, 2], {4, 2}) == pytest.approx(0.9197, abs=1e-4)


def test_ndcg_edge_cases():
    assert ndcg_at_k([1, 2], {1, 2}) == pytest.approx(1.0)
    assert ndcg_at_k([1, 1, 2], {1}) == pytest.approx(1.0)
    assert ndcg_at_k([1, 2], {3}) == 0.0
    assert ndcg_at_k(list(range(12)), {11}) == 0.0


def test_question_metrics_and_aggregate():
    rows = [
        question_metrics([1, 2, 3], {1}),
        question_metrics([1, 2, 3], {2}),
        question_metrics([1, 2, 3], {9}),
        question_metrics([1, 2, 3, 4], {4}),
    ]

    assert [r["first_relevant_rank"] for r in rows] == [1, 2, None, 4]
    result = aggregate(rows)
    assert result["hit@1"] == 0.25
    assert result["hit@3"] == 0.5
    assert result["hit@5"] == 0.75
    assert result["hit@10"] == 0.75
    assert result["mrr@10"] == pytest.approx(0.4375)
    expected_ndcg = (1 + 1 / math.log2(3) + 0 + 1 / math.log2(5)) / 4
    assert result["ndcg@10"] == pytest.approx(expected_ndcg)


def test_win_loss_treats_misses_as_rank_eleven():
    ranks_a = [1, 3, None, 2, None, 5]
    ranks_b = [1, 1, 4, None, None, 7]

    assert win_loss(ranks_a, ranks_b) == {"improved": 2, "worsened": 2, "same": 2}


def test_mcnemar_exact_hand_computed():
    hits_a = [1, 0, 0, 0, 0, 0, 1, 1]
    hits_b = [0, 1, 1, 1, 1, 1, 1, 0]

    result = mcnemar_exact(hits_a, hits_b)

    assert (result["only_a"], result["only_b"]) == (2, 5)
    assert result["p_value"] == pytest.approx(2 * (1 + 7 + 21) / 128)


def test_mcnemar_exact_one_versus_five():
    result = mcnemar_exact([1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1])

    assert result["p_value"] == pytest.approx(0.21875)
    assert result["p_value"] == pytest.approx(binomtest(1, 6, 0.5).pvalue)


def test_mcnemar_exact_without_discordant_pairs():
    assert mcnemar_exact([1, 0, 1], [1, 0, 1]) == {"only_a": 0, "only_b": 0, "p_value": 1.0}


def test_mcnemar_exact_matches_scipy_on_balanced_counts():
    result = mcnemar_exact([1] * 4 + [0] * 4, [0] * 4 + [1] * 4)

    assert result["p_value"] == pytest.approx(min(1.0, binomtest(4, 8, 0.5).pvalue))
    assert result["p_value"] == 1.0


def test_paired_bootstrap_constant_difference_has_degenerate_ci():
    result = paired_bootstrap_ci([0.0, 0.5, 1.0], [0.25, 0.75, 1.25])

    assert result["diff"] == pytest.approx(0.25)
    assert result["ci_low"] == pytest.approx(0.25)
    assert result["ci_high"] == pytest.approx(0.25)


def test_paired_bootstrap_is_reproducible_and_brackets_the_difference():
    a = [0, 1, 0, 1, 1, 0, 0, 1, 0, 0]
    b = [1, 1, 0, 1, 1, 1, 0, 1, 1, 0]

    first = paired_bootstrap_ci(a, b, n_resamples=2000, seed=0)
    second = paired_bootstrap_ci(a, b, n_resamples=2000, seed=0)

    assert first == second
    assert first["diff"] == pytest.approx(0.3)
    assert first["ci_low"] <= first["diff"] <= first["ci_high"]
    assert first["ci_low"] >= 0.0


def test_latency_summary():
    result = latency_summary([0.1, 0.2, 0.3, 0.4, 1.0])

    assert result["median"] == pytest.approx(0.3)
    assert result["p95"] == pytest.approx(0.88)
