"""Request validation and deterministic ranking without neural dependencies."""
import numpy as np

from scripts.evaluate_hybrid_validation import combine
from src.evaluation import rank_candidates


def validate_request(request, num_items, max_history=4096, max_k=100):
    """Accept only chronological mapped IDs; never silently discard unknowns."""
    if not isinstance(request, dict) or set(request) != {'history', 'k'}:
        raise ValueError('Expected exactly history and k')
    history, k = request['history'], request['k']
    if not isinstance(history, list) or len(history) > max_history:
        raise ValueError(f'history must be a list of at most {max_history} IDs')
    if any(type(i) is not int or not 1 <= i <= num_items for i in history):
        raise ValueError('history must contain known positive mapped integer IDs')
    if type(k) is not int or not 1 <= k <= max_k:
        raise ValueError(f'k must be an integer between 1 and {max_k}')
    return list(history), k


class Recommender:
    """Keep full history for filtering/CF and let the scorer truncate its input."""

    def __init__(self, scorer, cf, popularity, num_items, identity, max_history=4096, max_k=100):
        self.scorer, self.cf, self.popularity = scorer, cf, popularity
        self.num_items, self.identity = num_items, identity
        self.max_history, self.max_k = max_history, max_k

    def scores(self, history, hybrid=True):
        """Return scores and an explicit fallback reason; scores are not probabilities."""
        if not history:
            return self.popularity.score(history), 'empty_history_popularity'
        neural = np.concatenate(([0.], self.scorer(history)))
        if not hybrid:
            return neural, 'none'
        cf = self.cf.score(history)
        cf[0] = 0
        cf[history] = 0
        return combine(neural, cf, .25), 'no_cf_signal_raw_hstu' if cf.max() == 0 else 'none'

    def recommend(self, request, hybrid=True):
        """Return fewer than k items when the eligible catalog is exhausted."""
        history, k = validate_request(request, self.num_items, self.max_history, self.max_k)
        scores, fallback = self.scores(history, hybrid)
        ids = rank_candidates(scores, history, k)
        return {'items': [{'item_id': i, 'score': float(scores[i])} for i in ids],
                'fallback': fallback, 'eligible_items': self.num_items - len(set(history)),
                'model_id': self.identity, 'mode': 'hybrid' if hybrid else 'raw_hstu'}
