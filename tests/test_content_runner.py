"""Server CPU tests for the isolated title HSTU controls' full training/resume interface."""

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from content_hstu.run import run
from tests.content_fixture import write_cache
from tests.test_microlens_training import ROOT, write_data


class ContentRunnerTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data_dir = self.root / "data"
        self.data_dir.mkdir()
        write_data(self.data_dir)
        self.reference = json.loads((ROOT / "configs/hstu-microlens-debug.json").read_text())
        self.reference.update(mode="full", data_dir=str(self.data_dir), output_dir=str(self.root / "hstu"),
                              train_users=None, eval_users=None, batch_size=2, eval_batch_size=2,
                              ks=[1, 2], selection_metric="NDCG@1")
        self.reference["model"].update(sequence_length=2, dropout=0.2)
        self.reference_path = self.root / "reference.json"
        self.reference_path.write_text(json.dumps(self.reference))
        self.config = copy.deepcopy(self.reference)
        cache, digest, _ = write_cache(self.root, data_dir=self.data_dir)
        self.config.update(representation="title-fusion", fusion_weight=0.5, projection="train-item-pca-v1",
                           feature_cache=str(cache), features_sha256=digest,
                           reference_config=str(self.reference_path), output_dir=str(self.root / "content"))

    def launch(self, resume=None, stop=None, action="train", checkpoint=None, split="val"):
        path = self.root / "content.json"
        path.write_text(json.dumps(self.config))
        output = io.StringIO()
        with redirect_stdout(output):
            try:
                run(SimpleNamespace(config=path, action=action, resume=resume, stop_after_epoch=stop,
                                    checkpoint=checkpoint, split=split))
            except SystemExit:
                self.fail(output.getvalue())
        result_path = sorted((self.root / "content").glob("*/results.json"))[-1]
        return result_path.parent, json.loads(result_path.read_text())

    def assert_same(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.assert_same(actual[key], expected[key])
        elif isinstance(expected, (tuple, list)):
            self.assertEqual(len(actual), len(expected))
            for a, b in zip(actual, expected):
                self.assert_same(a, b)
        else:
            self.assertEqual(actual, expected)

    def test_epoch_boundary_replay_and_best_validation_evaluation(self):
        full_dir, full = self.launch()
        partial_dir, partial = self.launch(stop=1)
        self.assertEqual(partial["status"], "PARTIAL")
        resumed_dir, resumed = self.launch(resume=partial_dir / "last.pt")
        self.assertEqual(resumed["status"], "PASS")
        self.assert_same(resumed["selection"], full["selection"])
        self.assertEqual(resumed["initial_item_embedding_sha256"], full["initial_item_embedding_sha256"])
        self.assertEqual(resumed["parameter_count"], resumed["reference_parameter_count"])
        self.assertTrue(resumed["checks"]["checkpoint_output_exact"])
        self.assertTrue(resumed["checksums_unchanged"])
        expected = torch.load(full_dir / "last.pt", weights_only=True)
        actual = torch.load(resumed_dir / "last.pt", weights_only=True)
        for key in ("model_state_dict", "optimizer_state_dict", "rng_state", "epoch", "step"):
            self.assert_same(actual[key], expected[key])
        for a, b in zip(resumed["history"], full["history"]):
            for key in ("training", "validation", "probe_loss"):
                self.assert_same(a[key], b[key])
        _, evaluated = self.launch(action="evaluate", checkpoint=resumed_dir / "best.pt")
        best = next(row for row in resumed["history"] if row["epoch"] == resumed["selection"]["best_epoch"])
        self.assert_same(evaluated["evaluation"], best["validation"])
        self.assertEqual(actual["provenance"]["reference_config"], self.reference)
        self.assertIn("item_embedding.semantic", actual["model_state_dict"])
        self.assertTrue(resumed["frozen_content_unchanged"])
        self.assertTrue(resumed["checks"]["checkpoint_scores_exact"])
        self.assertEqual(full["content"]["content_id"], resumed["content"]["content_id"])

    def test_title_initialization_train_and_evaluate(self):
        self.config.update(representation="title-init", fusion_weight=0)
        directory, report = self.launch()
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["frozen_content_unchanged"])
        _, evaluated = self.launch(action="evaluate", checkpoint=directory / "best.pt")
        best = next(row for row in report["history"] if row["epoch"] == report["selection"]["best_epoch"])
        self.assert_same(evaluated["evaluation"], best["validation"])

    def test_smoke_mode_rejects_test_split_before_loading_a_checkpoint(self):
        self.reference.update(mode="smoke", train_users=2, eval_users=2)
        self.reference_path.write_text(json.dumps(self.reference))
        self.config.update(mode="smoke", train_users=2, eval_users=2)
        with self.assertRaisesRegex(ValueError, "cannot evaluate test"):
            self.launch(action="evaluate", checkpoint=self.root / "absent.pt", split="test")


if __name__ == "__main__":
    unittest.main()
