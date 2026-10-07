"""Document-level retrieval metrics, and paired statistics for comparing two systems.

A question's ranking is the list of retrieved document ids, best first. Ranks are 1-based;
``None`` means no relevant document within the cutoff. Relevance is binary.
"""

import math
from collections.abc import Collection, Sequence

import numpy as np

CUTOFF = 10
HIT_KS = (1, 3, 5, 10)
METRICS = ("hit@1", "hit@3", "hit@5", "hit@10", "mrr@10", "ndcg@10")
PER_QUESTION_FIELD = {"mrr@10": "rr@10"}


def first_relevant_rank(
    ranked: Sequence[int], gold: Collection[int], k: int = CUTOFF
) -> int | None:
    """Return the 1-based rank of the first relevant document within the top ``k``.

    Args:
        ranked (Sequence[int]): Retrieved document ids, best first.
        gold (Collection[int]): Relevant document ids.
        k (int): Cutoff.

    Returns:
        int | None: The rank, or None if no relevant document is in the top ``k``.
    """
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in gold:
            return rank
    return None


def hit_at_k(rank: int | None, k: int) -> float:
    """Return 1.0 if a relevant document is within the top ``k``, else 0.0.

    Args:
        rank (int | None): First relevant rank.
        k (int): Cutoff.

    Returns:
        float: 1.0 or 0.0.
    """
    return 1.0 if rank is not None and rank <= k else 0.0


def reciprocal_rank(rank: int | None, k: int = CUTOFF) -> float:
    """Return the reciprocal of the first relevant rank, or 0.0 beyond the cutoff.

    Args:
        rank (int | None): First relevant rank.
        k (int): Cutoff.

    Returns:
        float: Reciprocal rank.
    """
    return 1.0 / rank if rank is not None and rank <= k else 0.0


def ndcg_at_k(ranked: Sequence[int], gold: Collection[int], k: int = CUTOFF) -> float:
    """Return binary nDCG@k with the standard log2 discount.

    The ideal ranking places min(|gold|, k) relevant documents first. A document counts
    once even if it is retrieved twice.

    Args:
        ranked (Sequence[int]): Retrieved document ids, best first.
        gold (Collection[int]): Relevant document ids.
        k (int): Cutoff.

    Returns:
        float: nDCG@k in [0, 1]; 0.0 when there is no relevant document.
    """
    seen = set()
    dcg = 0.0
    for position, doc_id in enumerate(ranked[:k]):
        if doc_id in gold and doc_id not in seen:
            seen.add(doc_id)
            dcg += 1.0 / math.log2(position + 2)
    ideal = sum(1.0 / math.log2(position + 2) for position in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def question_metrics(ranked: Sequence[int], gold: Collection[int], k: int = CUTOFF) -> dict:
    """Compute every per-question metric.

    Args:
        ranked (Sequence[int]): Retrieved document ids, best first.
        gold (Collection[int]): Relevant document ids.
        k (int): Cutoff for the rank, reciprocal rank and nDCG.

    Returns:
        dict: ``first_relevant_rank``, ``hit@1/3/5/10``, ``rr@10`` and ``ndcg@10``.
    """
    rank = first_relevant_rank(ranked, gold, k)
    row = {"first_relevant_rank": rank}
    row.update({f"hit@{h}": hit_at_k(rank, h) for h in HIT_KS})
    row["rr@10"] = reciprocal_rank(rank, k)
    row["ndcg@10"] = ndcg_at_k(ranked, gold, k)
    return row


def metric_values(rows: Sequence[dict], metric: str) -> list[float]:
    """Return one system's per-question values of an aggregate metric.

    Args:
        rows (Sequence[dict]): Per-question metric rows.
        metric (str): One of ``METRICS``.

    Returns:
        list[float]: Values in question order.
    """
    field = PER_QUESTION_FIELD.get(metric, metric)
    return [row[field] for row in rows]


def aggregate(rows: Sequence[dict]) -> dict:
    """Average the per-question metrics.

    Args:
        rows (Sequence[dict]): Per-question metric rows.

    Returns:
        dict: Mean of each metric in ``METRICS``.
    """
    return {metric: float(np.mean(metric_values(rows, metric))) for metric in METRICS}


def win_loss(
    ranks_a: Sequence[int | None], ranks_b: Sequence[int | None], miss_rank: int = CUTOFF + 1
) -> dict:
    """Count questions whose first relevant rank improved, worsened or stayed the same.

    Args:
        ranks_a (Sequence[int | None]): First relevant ranks of the reference system.
        ranks_b (Sequence[int | None]): First relevant ranks of the compared system.
        miss_rank (int): Rank assigned when nothing relevant is within the cutoff.

    Returns:
        dict: ``improved``, ``worsened`` and ``same`` counts (b relative to a).
    """
    counts = {"improved": 0, "worsened": 0, "same": 0}
    for a, b in zip(ranks_a, ranks_b, strict=True):
        a = miss_rank if a is None else a
        b = miss_rank if b is None else b
        counts["improved" if b < a else "worsened" if b > a else "same"] += 1
    return counts


def paired_bootstrap_ci(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_resamples: int = 10_000,
    seed: int = 0,
    alpha: float = 0.05,
) -> dict:
    """Percentile bootstrap CI for the mean paired difference (b - a) over questions.

    Args:
        values_a (Sequence[float]): Per-question values of the reference system.
        values_b (Sequence[float]): Per-question values of the compared system, same order.
        n_resamples (int): Bootstrap resamples.
        seed (int): Seed of the resampling.
        alpha (float): 1 - confidence level.

    Returns:
        dict: ``diff`` (observed mean difference), ``ci_low`` and ``ci_high``.
    """
    diffs = np.asarray(values_b, dtype=float) - np.asarray(values_a, dtype=float)
    rng = np.random.default_rng(seed)
    resampled = diffs[rng.integers(0, len(diffs), size=(n_resamples, len(diffs)))].mean(axis=1)
    low, high = np.percentile(resampled, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": float(diffs.mean()), "ci_low": float(low), "ci_high": float(high)}


def mcnemar_exact(hits_a: Sequence[float], hits_b: Sequence[float]) -> dict:
    """Exact (binomial) McNemar test on paired binary outcomes.

    Args:
        hits_a (Sequence[float]): Per-question 0/1 outcomes of the reference system.
        hits_b (Sequence[float]): Per-question 0/1 outcomes of the compared system.

    Returns:
        dict: ``only_a`` and ``only_b`` (discordant counts) and the two-sided ``p_value``.
    """
    only_a = sum(1 for a, b in zip(hits_a, hits_b, strict=True) if a and not b)
    only_b = sum(1 for a, b in zip(hits_a, hits_b, strict=True) if b and not a)
    n = only_a + only_b
    if n == 0:
        return {"only_a": 0, "only_b": 0, "p_value": 1.0}
    tail = sum(math.comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return {"only_a": only_a, "only_b": only_b, "p_value": min(1.0, 2 * tail)}


def latency_summary(seconds: Sequence[float]) -> dict:
    """Summarize per-query latencies.

    Args:
        seconds (Sequence[float]): Latencies in seconds.

    Returns:
        dict: ``median`` and ``p95`` in seconds.
    """
    values = np.asarray(seconds, dtype=float)
    return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95))}
