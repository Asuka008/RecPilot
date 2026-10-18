"""Full-catalog, single-target ranking evaluation shared by all scorers."""

from collections import Counter
import math

import numpy as np


def rank_candidates(scores: np.ndarray, history: list[int], k: int) -> list[int]:
    """Exclude padding/history and break score ties by ascending mapped item ID."""
    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("scores must be a finite one-dimensional vector")
    if k < 1:
        raise ValueError("k must be positive")
    eligible = np.ones(len(scores), dtype=bool)
    eligible[0] = False
    eligible[history] = False
    candidates = np.flatnonzero(eligible)
    take = min(k, len(candidates))
    if take == 0:
        return []
    values = scores[candidates]
    # Partition finds the boundary; explicitly select ID-ordered boundary ties.
    threshold = np.partition(values, len(values) - take)[len(values) - take]
    higher = candidates[values > threshold]
    tied = candidates[values == threshold][: take - len(higher)]
    selected = np.concatenate((higher, tied))
    order = np.lexsort((selected, -scores[selected]))
    return selected[order].tolist()


def single_target_metrics(ranking: list[int], target: int, ks: list[int]) -> dict:
    """Compute Recall and normalized DCG for one binary relevant item."""
    if len(set(ranking)) != len(ranking):
        raise ValueError("ranking must contain unique items")
    rank = ranking.index(target) + 1 if target in ranking else math.inf
    metrics = {}
    for k in ks:
        if k < 1:
            raise ValueError("cutoffs must be positive")
        hit = rank <= k
        metrics[f"Recall@{k}"] = float(hit)
        metrics[f"NDCG@{k}"] = 1 / math.log2(rank + 1) if hit else 0.0
    return metrics


def evaluate(model, records: list[dict], split: str, ks: list[int],
             train_frequency: Counter) -> dict:
    """Macro-average users, including targets with zero training occurrences."""
    if split not in {"val", "test"} or not ks or min(ks) < 1:
        raise ValueError("invalid split or ranking cutoffs")
    keys = [f"{metric}@{k}" for k in ks for metric in ("Recall", "NDCG")]
    groups = {name: {"users": 0, "sums": dict.fromkeys(keys, 0.0)}
              for name in ("all", "0", "1-5", "6-20", "21+")}
    for record in records:
        history = record["train"] + ([record["val"]] if split == "test" else [])
        target = record[split]
        if target in history:
            raise ValueError(
                f"User {record['user_id']}: target is in filtered history; "
                "repeat-target evaluation requires a separate protocol"
            )
        scores = model.score(history)
        if not 0 < target < len(scores):
            raise ValueError("target is outside the candidate catalog")
        ranking = rank_candidates(scores, history, max(ks))
        metrics = single_target_metrics(ranking, target, ks)
        frequency = train_frequency[target]
        bucket = "0" if frequency == 0 else "1-5" if frequency <= 5 else "6-20" if frequency <= 20 else "21+"
        for name in ("all", bucket):
            groups[name]["users"] += 1
            for key, value in metrics.items():
                groups[name]["sums"][key] += value
    result = {}
    for name, group in groups.items():
        count = group["users"]
        result[name] = {"users": count, "metrics": {
            key: value / count if count else None
            for key, value in group["sums"].items()
        }}
    return {"split": split, "groups_by_target_train_frequency": result}
