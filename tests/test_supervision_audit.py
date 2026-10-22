"""Tiny standard-library checks of shifted supervision accounting."""

import unittest

from scripts.audit_hstu_supervision import audit_supervision


class SupervisionAuditTest(unittest.TestCase):
    def test_first_event_and_truncation_are_not_positive_targets(self):
        records = [{"train": [1, 2, 3, 4], "val": 5}, {"train": [1, 6], "val": 7}]
        report = audit_supervision(records, 8, 2)
        self.assertEqual(report["train_history_events"], 6)
        self.assertEqual(report["supervised_positions"], 3)
        self.assertEqual(report["train_history_items"], 5)
        self.assertEqual(report["positive_target_items"], 3)
        self.assertEqual(report["history_observed_items_without_positive_target"], 2)
        self.assertEqual(report["catalog_items_without_positive_target"], 5)
        self.assertEqual(report["validation_by_target_history_frequency"]["0"],
                         {"validation_users": 2, "zero_supervised_target_users": 2})

    def test_held_out_labels_do_not_change_training_counts(self):
        records = [{"train": [1, 2, 3, 4], "val": 5}]
        before = audit_supervision(records, 8, 2)
        records[0].update(val=1, test="intentionally unread")
        after = audit_supervision(records, 8, 2)
        for key in before.keys() - {"validation_by_target_history_frequency"}:
            self.assertEqual(before[key], after[key])
        self.assertEqual(after["validation_by_target_history_frequency"]["1-5"],
                         {"validation_users": 1, "zero_supervised_target_users": 1})


if __name__ == "__main__":
    unittest.main()
