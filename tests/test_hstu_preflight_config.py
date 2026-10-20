"""Pure-Python safeguards; these tests never import torch or run a model."""

import copy
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts.prepare_hstu_reference import ROOT, outside_project, prepare, source, validate_report
from src.check_hstu_reference import validate_config, verify_upstream
from src.check_hstu_dense import validate_comparison_config


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

    def test_dense_comparison_config_accepts_empty_rows_and_rejects_bad_cases(self):
        config = json.loads((ROOT / "configs/hstu-dense-comparison-server.json").read_text())
        validate_comparison_config(config)
        for key, value in (("comparison_cases", []), ("comparison_seeds", [-1]),
                           ("comparison_cases", [{"name": "empty", "history_lengths": [0, 0]}]),
                           ("comparison_cases", [{"name": "too_long", "history_lengths": [999]}])):
            invalid = {**config, key: value}
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                validate_comparison_config(invalid)

    def test_main_reaches_package_validation_before_loading_model_libraries(self):
        import src.check_hstu_reference as preflight
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {**self.config, "output_dir": str(root / "results")}
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            failure = importlib.metadata.PackageNotFoundError("intentional-package-probe")
            with (patch("sys.argv", ["check_hstu_reference", "--config", str(config_path)]),
                  patch.object(preflight, "command", side_effect=["test-commit", ""]),
                  patch.object(preflight, "verify_upstream", return_value="test-upstream"),
                  patch.object(preflight.importlib.metadata, "distributions", return_value=[]),
                  patch.object(preflight.importlib.metadata, "version", side_effect=failure) as version,
                  redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO())):
                with self.assertRaises(SystemExit) as exited:
                    preflight.main()
            self.assertEqual(exited.exception.code, 1)
            version.assert_called_once()
            result = json.loads(next((root / "results").glob("*/results.json")).read_text())
            self.assertEqual(result["runtime"]["packages"], {})
            self.assertEqual(result["error"]["type"], "PackageNotFoundError")
            self.assertIn("intentional-package-probe", result["error"]["message"])

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

    def test_source_resumes_interrupted_checkout_without_resetting_edits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            origin, checkout = root / "origin", root / "checkout"

            def git(*args):
                return subprocess.check_output(["git", *map(str, args)], text=True,
                                               stderr=subprocess.DEVNULL).strip()

            git("init", "-q", origin)
            (origin / "model.py").write_text("reference source\n")
            git("-C", origin, "add", "model.py")
            git("-C", origin, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                "-c", "commit.gpgsign=false", "commit", "-qm", "Reference")
            revision = git("-C", origin, "rev-parse", "HEAD")
            git("clone", "--no-checkout", origin, checkout)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"repository": str(origin), "revision": revision,
                "files_sha256": {"model.py": hashlib.sha256((origin / "model.py").read_bytes()).hexdigest()}}))
            with patch("scripts.prepare_hstu_reference.MANIFEST", manifest):
                source(checkout)
                self.assertEqual((checkout / "model.py").read_text(), "reference source\n")
                self.assertEqual(git("-C", checkout, "rev-parse", "HEAD"), revision)
                (checkout / "model.py").write_text("user changes\n")
                with self.assertRaisesRegex(RuntimeError, "not overwriting"):
                    source(checkout)
                self.assertEqual((checkout / "model.py").read_text(), "user changes\n")

    def test_install_replaces_only_incompatible_fbgemm_build(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def fake_run(args, **kwargs):
                if "-c" in args and "sys.version_info" in args[-1]:
                    output = "3.11.16\n"
                elif "-c" in args and "m.version('fbgemm-gpu')" in args[-1]:
                    output = "1.5.0+cu128\n"
                else:
                    output = ""
                return subprocess.CompletedProcess(args, 0, stdout=output)

            with (patch("scripts.prepare_hstu_reference.check_platform"),
                  patch("scripts.prepare_hstu_reference.validate_report"),
                  patch("scripts.prepare_hstu_reference.run", side_effect=fake_run) as executed,
                  redirect_stdout(io.StringIO())):
                prepare("install", root / "environment", root)
            forced = [call.args[0] for call in executed.call_args_list
                      if "--force-reinstall" in call.args[0]]
            self.assertEqual(len(forced), 1)
            self.assertIn("--no-deps", forced[0])
            self.assertTrue(forced[0][-1].startswith("fbgemm-gpu @ https://files.pythonhosted.org/"))
            self.assertFalse(any(argument.startswith("torch==") for argument in forced[0]))


if __name__ == "__main__":
    unittest.main()
