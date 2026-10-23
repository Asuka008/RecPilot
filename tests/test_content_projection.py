"""Tiny NumPy checks of projection fitting scope and numerical contracts."""

import unittest

import numpy as np

from content_hstu.projection import fit_projection


class ContentProjectionTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.features = rng.normal(size=(9, 6)).astype(np.float32)
        self.features /= np.linalg.norm(self.features, axis=1, keepdims=True)
        self.features[0] = 0
        self.records = [{"train": [1, 2, 3], "val": 7, "test": 8}, {"train": [3, 4, 5], "val": 8, "test": 6}]

    def test_repeat_scale_padding_and_fit_membership(self):
        table, arrays, report = fit_projection(self.features, self.records, 3)
        again, _, second = fit_projection(self.features, self.records, 3)
        np.testing.assert_array_equal(table, again)
        self.assertEqual(report, second)
        self.assertEqual(table.dtype, np.float32)
        np.testing.assert_array_equal(table[0], np.zeros(3))
        np.testing.assert_allclose(np.linalg.norm(table[1:], axis=1), 0.02 * np.sqrt(3), atol=1e-8)
        np.testing.assert_array_equal(arrays["training_item_ids"], [1, 2, 3, 4, 5])

    def test_heldout_features_and_labels_do_not_fit_basis(self):
        _, arrays, _ = fit_projection(self.features, self.records, 3)
        changed = self.features.copy()
        changed[6:] *= -1
        records = [{**row, "val": 1, "test": 2} for row in self.records]
        table, other, _ = fit_projection(changed, records, 3)
        np.testing.assert_array_equal(arrays["mean"], other["mean"])
        np.testing.assert_array_equal(arrays["components"], other["components"])
        np.testing.assert_array_equal(arrays["table"][:6], table[:6])
        self.assertFalse(np.array_equal(arrays["table"][6:], table[6:]))

    def test_frequency_and_record_order_do_not_reweight_pca(self):
        _, arrays, _ = fit_projection(self.features, self.records, 3)
        records = [{"train": [5, 4, 3, 3, 3, 2, 1]}]
        _, other, _ = fit_projection(self.features, records, 3)
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], other[key])

    def test_degenerate_inputs_fail(self):
        for features, records, dim in ((self.features, [{"train": [0, 1]}], 3),
                                       (self.features, [{"train": [1]}], 3),
                                       (self.features, self.records, 7),
                                       (np.ones_like(self.features), self.records, 3)):
            with self.assertRaises(ValueError):
                fit_projection(features, records, dim)


if __name__ == "__main__":
    unittest.main()
