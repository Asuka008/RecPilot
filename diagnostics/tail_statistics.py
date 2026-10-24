"""Deterministic full-catalog rank diagnostics without neural dependencies."""

import numpy as np

from src.evaluation import rank_candidates

GROUPS = ("all", "0", "1-5", "6-20", "21+")


def bucket(count):
    """Use training-history frequency, not held-out target counts."""
    return "0" if count == 0 else "1-5" if count <= 5 else "6-20" if count <= 20 else "21+"


def title_permutation(frequency, seed):
    """Shuffle only within frequency buckets; preserve padding and row identities."""
    frequency = np.asarray(frequency)
    if frequency.ndim != 1 or np.any(frequency < 0):
        raise ValueError("Expected nonnegative catalog frequencies")
    permutation = np.arange(len(frequency), dtype=np.int64)
    rng = np.random.default_rng(seed)
    for group in GROUPS[1:]:
        ids = np.array([i for i in range(1, len(frequency)) if bucket(frequency[i]) == group], dtype=np.int64)
        permutation[ids] = rng.permutation(ids)
    return permutation


def target_position(scores, history, target, k=20):
    """Return exact one-based rank and signed target-minus-boundary score.

    Equal-score candidates with smaller IDs precede the target. A zero margin
    therefore does not establish a hit. If fewer than K candidates are eligible,
    the boundary is the last eligible item and its effective K is recorded.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Expected finite catalog scores")
    if not 0 < target < len(values) or target in history:
        raise ValueError("Target must be eligible")
    eligible = np.ones(len(values), dtype=bool)
    eligible[0] = False
    eligible[history] = False
    ids = np.flatnonzero(eligible)
    rank = 1 + np.count_nonzero((values[ids] > values[target]) |
                               ((values[ids] == values[target]) & (ids < target)))
    top = rank_candidates(values, history, k)
    return int(rank), float(values[target] - values[top[-1]]), len(top)


def describe(values):
    """Use explicit linear quantiles; empty groups have null summaries."""
    if not len(values):
        return None
    values = np.asarray(values)
    q = np.quantile(values, [0, .25, .5, .75, .9, .99, 1], method="linear")
    return {"mean": float(values.mean()), **dict(zip(
        ("min", "p25", "p50", "p75", "p90", "p99", "max"), map(float, q)))}


def summarize_ranks(ranks, margins, effective_k, target_frequency, cutoffs):
    """Aggregate aligned per-user observations without hiding empty slices."""
    arrays = [np.asarray(x) for x in (ranks, margins, effective_k, target_frequency)]
    ranks, margins, effective_k, target_frequency = arrays
    if any(x.ndim != 1 or len(x) != len(ranks) for x in arrays):
        raise ValueError("Per-user arrays must be aligned vectors")
    if np.any(ranks < 1) or not all(np.isfinite(x).all() for x in arrays):
        raise ValueError("Invalid rank observations")
    labels = np.array([bucket(n) for n in target_frequency])
    result = {}
    for group in GROUPS:
        mask = np.ones(len(ranks), dtype=bool) if group == "all" else labels == group
        r, m = ranks[mask], margins[mask]
        result[group] = {"users": int(mask.sum()), "rank": describe(r),
                         "target_minus_top20_boundary": describe(m),
                         "effective_boundary_k": describe(effective_k[mask]),
                         "hits": {f"Recall@{k}": int((r <= k).sum()) for k in cutoffs},
                         "recall": {f"Recall@{k}": float((r <= k).mean()) if len(r) else None for k in cutoffs}}
    return result


def complementarity(left, right, target_frequency, k=20):
    """Count same-K hits; the union is an oracle bound, not a merged ranking."""
    left, right, freq = map(np.asarray, (left, right, target_frequency))
    if left.ndim != 1 or left.shape != right.shape or left.shape != freq.shape:
        raise ValueError("Complementarity requires aligned per-user vectors")
    labels = np.array([bucket(n) for n in freq])
    output = {}
    for group in GROUPS:
        mask = np.ones(len(freq), dtype=bool) if group == "all" else labels == group
        a, b = left[mask] <= k, right[mask] <= k
        counts = {"both": int((a & b).sum()), "hstu_only": int((a & ~b).sum()),
                  "itemcf_only": int((~a & b).sum()), "neither": int((~a & ~b).sum())}
        output[group] = {"users": int(mask.sum()), **counts,
                         "oracle_union_hits": int((a | b).sum()),
                         "oracle_union_recall": float((a | b).mean()) if len(a) else None}
    return {"k": k, "scope": "same-user hit union upper bound; not deployable Top-K metrics", "groups": output}
