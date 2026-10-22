"""Standard-library checks that inspection works before title transfer or setup."""

import argparse
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts import prepare_title_features


ROOT = Path(__file__).resolve().parents[1]


class TitlePreflightTest(unittest.TestCase):
    def test_preflight_reports_missing_inputs_without_importing_an_encoder(self):
        config = json.loads((ROOT / "configs/title-features-minilm-server.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            args = argparse.Namespace(action="preflight", config=config_path)
            with patch.object(prepare_title_features, "ROOT", root), patch.object(
                    prepare_title_features, "inspect_runtime", return_value={"dependencies_ready": False}) as inspect:
                with redirect_stdout(io.StringIO()):
                    prepare_title_features.run(args)
            inspect.assert_called_once_with(config)
            report = json.loads(next((root / config["output_dir"]).glob("*/results.json")).read_text())
            self.assertEqual(report["status"], "NEEDS_SETUP")
            self.assertFalse(any(report["input_files_present"].values()))
            self.assertFalse((root / "data").exists())

    def test_output_cannot_mutate_raw_or_processed_inputs(self):
        config = json.loads((ROOT / "configs/title-features-minilm-server.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config_path = root / "config.json"
            for output in ("data/raw/features", "data/processed/features"):
                config["output_dir"] = output
                config_path.write_text(json.dumps(config))
                with patch.object(prepare_title_features, "ROOT", root), self.assertRaisesRegex(ValueError, "raw or processed"):
                    prepare_title_features.run(argparse.Namespace(action="audit", config=config_path))
            self.assertFalse((root / "data").exists())


if __name__ == "__main__":
    unittest.main()
