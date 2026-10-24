"""Small request/ranking/HTTP tests without PyTorch or real data."""
import io
import json
import unittest

import numpy as np

from serving.core import Recommender, validate_request
from serving.http import make_handler
from src.baselines import ItemCF, Popularity


class ServingTest(unittest.TestCase):
    def setUp(self):
        self.records = [{'train': [1, 2]}, {'train': [1, 3]}]
        self.calls = []
        def scorer(history):
            self.calls.append(list(history))
            return np.array([.2, .5, .5, .1])
        self.engine = Recommender(scorer, ItemCF(self.records, 4, 2),
                                  Popularity(self.records, 4), 4, 'fixture', max_history=10)

    def test_invalid_requests_rejected(self):
        for request in [None, {}, {'history': [0], 'k': 2}, {'history': [5], 'k': 2},
                        {'history': [True], 'k': 2}, {'history': [1.0], 'k': 2},
                        {'history': [], 'k': True}, {'history': [], 'k': 0},
                        {'history': [], 'k': 101}, {'history': [1]*11, 'k': 1},
                        {'history': [], 'k': 1, 'alpha': .5}]:
            with self.subTest(request=request), self.assertRaises(ValueError):
                validate_request(request, 4, max_history=10)

    def test_empty_history_popularity_and_ties(self):
        response = self.engine.recommend({'history': [], 'k': 4})
        self.assertEqual([x['item_id'] for x in response['items']], [1, 2, 3, 4])
        self.assertEqual(response['fallback'], 'empty_history_popularity')
        self.assertEqual(self.calls, [])

    def test_no_cf_uses_raw_and_filters_history(self):
        response = self.engine.recommend({'history': [4], 'k': 4})
        self.assertEqual([x['item_id'] for x in response['items']], [2, 3, 1])
        self.assertEqual(response['fallback'], 'no_cf_signal_raw_hstu')
        self.assertEqual(response['items'][0]['score'], .5)

    def test_cf_normalization_excludes_seen_items(self):
        # Item 1 has the largest aggregate CF score but is already seen.
        history = [2, 3, 1]
        scores, fallback = self.engine.scores(history)
        self.assertEqual(fallback, 'no_cf_signal_raw_hstu')
        self.assertTrue(np.array_equal(scores[1:], [.2, .5, .5, .1]))
        self.assertEqual(self.calls[-1], history)

    def test_hybrid_formula_and_exhausted_catalog(self):
        scores, fallback = self.engine.scores([1])
        self.assertEqual(fallback, 'none')
        self.assertAlmostEqual(scores[2], .75 * (.5 + 1) / 2 + .25)
        response = self.engine.recommend({'history': [1, 2, 3, 4, 1], 'k': 20})
        self.assertEqual(response['items'], [])
        self.assertEqual(response['eligible_items'], 0)

    def test_http_adapter_success_bad_request_and_health(self):
        # Exercise the actual handler through an in-memory socket contract.
        class Connection:
            def __init__(self, request):
                self.input = io.BytesIO(request)
                self.output = bytearray()
            def makefile(self, *args):
                return self.input
            def sendall(self, data):
                self.output.extend(data)
            def settimeout(self, value):
                pass
        for method, path, body, status in [
            ('POST', '/recommend', {'history': [1], 'k': 2}, 200),
            ('POST', '/recommend', {'history': [0], 'k': 2}, 400),
            ('POST', '/unknown', {}, 404), ('GET', '/health', {}, 200)]:
            encoded = json.dumps(body).encode()
            raw = f'{method} {path} HTTP/1.0\r\nContent-Length: {len(encoded)}\r\n\r\n'.encode() + encoded
            connection = Connection(raw)
            make_handler(self.engine)(connection, ('127.0.0.1', 0), None)
            self.assertIn(f' {status} '.encode(), connection.output.split(b'\r\n')[0])
            response = json.loads(connection.output.split(b'\r\n\r\n')[1])
            if path == '/health':
                self.assertEqual(response['model_id'], 'fixture')


if __name__ == '__main__':
    unittest.main()
