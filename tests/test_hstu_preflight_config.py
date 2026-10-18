"""Pure-Python safeguards; these tests never import torch or run a model."""

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_hstu_reference import ROOT, outside_project, validate_report
from src.check_hstu_reference import validate_config, verify_upstream


class PreflightConfigTest(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/hstu-reference-check-server.json").read_text())

    def test_committed_config_and_revision_are_explicit(self):
        validate_config(self.config)
        spec = json.loads((ROOT / "configs/hstu-reference-upstream.json").read_text())
        self.assertEqual(len(spec["revision"]), 40)
        self.assertIn(spec["hstu_source"], spec["files_sha256"])
        self.assertIn(spec["recipe"], spec["files_sha256"])
        self.assertEqual(spec["expected_packages"]["torch"], "2.10.0+cu128")

    def test_rejects_invalid_lengths_and_dimensions(self):
        for key, value in (("history_lengths", [0]), ("history_lengths", [9]),
                           ("embedding_dim", 0), ("num_heads", True), ("atol", -1)):
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_config(config)

    def test_environment_path_cannot_be_inside_project(self):
        for path in (ROOT, ROOT / "conda-envs/reference", ROOT / "src/../reference"):
            with self.assertRaises(ValueError):
                outside_project(path)

    def test_modified_upstream_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "model.py"
            path.write_text("original source")
            spec = {"revision": "a" * 40, "files_sha256": {
                "model.py": hashlib.sha256(path.read_bytes()).hexdigest()}}
            with patch("src.check_hstu_reference.command", side_effect=["a" * 40, ""]):
                self.assertEqual(verify_upstream(root, spec), "a" * 40)
            path.write_text("changed source")
            with patch("src.check_hstu_reference.command", side_effect=["a" * 40, ""]):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    verify_upstream(root, spec)

    def test_wrong_revision_and_dirty_checkout_are_rejected(self):
        spec = {"revision": "a" * 40, "files_sha256": {}}
        with patch("src.check_hstu_reference.command", return_value="b" * 40):
            with self.assertRaisesRegex(ValueError, "revision mismatch"):
                verify_upstream(Path("/unused"), spec)
        with patch("src.check_hstu_reference.command", side_effect=["a" * 40, " M model.py"]):
            with self.assertRaisesRegex(ValueError, "modified"):
                verify_upstream(Path("/unused"), spec)

    def test_stale_or_edited_dependency_plan_is_rejected(self):
        from scripts.prepare_hstu_reference import SPEC
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory)
            prefix = report / "environment"
            state = {"prefix": str(prefix),
                     "requirements_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest()}
            (report / "state.json").write_text(json.dumps(state))
            (report / "pip-plan.json").write_text(json.dumps({"install": [
                {"metadata": {"name": "example", "version": "1.0"}}]}))
            (report / "resolved-constraints.txt").write_text("example==1.0\n")
            validate_report(report, prefix)
            with self.assertRaisesRegex(RuntimeError, "different prefix"):
                validate_report(report, report / "another-environment")
            (report / "resolved-constraints.txt").write_text("example==2.0\n")
            with self.assertRaisesRegex(RuntimeError, "constraints"):
                validate_report(report, prefix)


if __name__ == "__main__":
    unittest.main()
