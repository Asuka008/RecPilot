"""Training-only Popularity and cosine ItemCF with sparse neighbor storage."""

from collections import Counter, defaultdict
import heapq
import math

import numpy as np


class Popularity:
    """Score items by training interaction count, including repeated events."""

    def __init__(self, records: list[dict], num_items: int) -> None:
        self.counts = np.zeros(num_items + 1, dtype=np.int64)
        for record in records:
            for item in record["train"]:
                self.counts[item] += 1

    def score(self, history: list[int]) -> np.ndarray:
        """Return one full-catalog score vector; history filtering is external."""
        return self.counts.astype(np.float64)


class ItemCF:
    """Use binary user-item cosine similarity and uniform history aggregation."""

    def __init__(self, records: list[dict], num_items: int, neighbors: int) -> None:
        if neighbors < 1:
            raise ValueError("neighbors must be positive")
        self.num_items = num_items
        user_items = [sorted(set(record["train"])) for record in records]
        item_users: dict[int, list[int]] = defaultdict(list)
        for user, items in enumerate(user_items):
            for item in items:
                item_users[item].append(user)

        self.neighbors: dict[int, list[tuple[int, float]]] = {}
        for item in sorted(item_users):
            cooccurrences: Counter = Counter()
            for user in item_users[item]:
                cooccurrences.update(user_items[user])
            cooccurrences.pop(item, None)
            scored = (
                (other, count / math.sqrt(len(item_users[item]) * len(item_users[other])))
                for other, count in cooccurrences.items()
            )
            self.neighbors[item] = heapq.nsmallest(
                neighbors, scored, key=lambda pair: (-pair[1], pair[0])
            )

    def score(self, history: list[int]) -> np.ndarray:
        """Sum neighbor similarities once per distinct visible history item."""
        scores = np.zeros(self.num_items + 1, dtype=np.float64)
        for item in sorted(set(history)):
            for candidate, similarity in self.neighbors.get(item, []):
                scores[candidate] += similarity
        return scores
