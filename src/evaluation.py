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


class RankingAccumulator:
    """Share ranking rules and group denominators across CPU and batched scorers."""

    def __init__(self, ks, train_frequency):
        if not ks or any(type(k) is not int or k < 1 for k in ks):
            raise ValueError("invalid ranking cutoffs")
        self.ks = ks
        self.train_frequency = train_frequency
        keys = [f"{metric}@{k}" for k in ks for metric in ("Recall", "NDCG")]
        self.groups = {
            axis: {name: {"users": 0, "sums": dict.fromkeys(keys, 0.0)} for name in names}
            for axis, names in {
                "target_train_frequency": ("all", "0", "1-5", "6-20", "21+"),
                "history_length": ("all", "1-5", "6-10", "11-20", "21+"),
            }.items()
        }

    def add(self, scores, history, target):
        """Filter the complete visible history, even when model input is truncated."""
        if target in history:
            raise ValueError("target is in filtered history; repeat targets require a separate protocol")
        if not 0 < target < len(scores):
            raise ValueError("target is outside the candidate catalog")
        metrics = single_target_metrics(rank_candidates(scores, history, max(self.ks)), target, self.ks)
        frequency = self.train_frequency[target]
        frequency_bucket = ("0" if frequency == 0 else "1-5" if frequency <= 5 else
                            "6-20" if frequency <= 20 else "21+")
        length = len(history)
        length_bucket = "1-5" if length <= 5 else "6-10" if length <= 10 else "11-20" if length <= 20 else "21+"
        for axis, bucket in (("target_train_frequency", frequency_bucket),
                             ("history_length", length_bucket)):
            for name in ("all", bucket):
                group = self.groups[axis][name]
                group["users"] += 1
                for key, value in metrics.items():
                    group["sums"][key] += value

    def result(self, split):
        """Return macro averages; empty groups have null metrics, not zero scores."""
        return {"split": split, **{
            f"groups_by_{axis}": {
                name: {"users": group["users"], "metrics": {
                    key: value / group["users"] if group["users"] else None
                    for key, value in group["sums"].items()
                }} for name, group in groups.items()
            } for axis, groups in self.groups.items()
        }}


def evaluate(model, records: list[dict], split: str, ks: list[int],
             train_frequency: Counter) -> dict:
    """Macro-average users, including targets with zero training occurrences."""
    if split not in {"val", "test"} or not ks or min(ks) < 1:
        raise ValueError("invalid split or ranking cutoffs")
    accumulator = RankingAccumulator(ks, train_frequency)
    for record in records:
        history = record["train"] + ([record["val"]] if split == "test" else [])
        target = record[split]
        scores = model.score(history)
        accumulator.add(scores, history, target)
    return accumulator.result(split)
