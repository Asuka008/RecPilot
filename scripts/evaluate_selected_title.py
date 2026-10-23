"""Admit one validation-selected title checkpoint to frozen test evaluation."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.compare_title_validation import compare_reports


def file_digest(path):
    """Stream artifact hashes without loading model weights."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def admit(root, recipe):
    """Verify the declared winner, current inputs and exact recorded best file."""
    if set(recipe) != {"reports", "selected_model", "selected_epoch", "training_config"}:
        raise ValueError("Unexpected frozen test recipe fields")
    paths = {name: root / path for name, path in recipe["reports"].items()}
    reports = {name: json.loads(path.read_text()) for name, path in paths.items()}
    compare_reports(reports)
    order = ("id", "title-init", "title-fusion")
    winner = max(order, key=lambda name: reports[name]["selection"]["best_score"])
    if winner != recipe["selected_model"] or winner == "id":
        raise ValueError("Frozen title candidate differs from validation winner; exact ties favor ID")
    report = reports[winner]
    if report["selection"]["best_epoch"] != recipe["selected_epoch"]:
        raise ValueError("Frozen epoch differs from the selected epoch")
    config_path = root / recipe["training_config"]
    if json.loads(config_path.read_text()) != report["config"]:
        raise ValueError("Current training configuration differs")
    provenance = report["provenance"]
    reference_path = root / report["config"]["reference_config"]
    if json.loads(reference_path.read_text()) != provenance["reference_config"]:
        raise ValueError("Current reference configuration differs")
    for name, digest in {**provenance["source_sha256"], **provenance["content_source_sha256"]}.items():
        if file_digest(root / name) != digest:
            raise ValueError(f"Current model source differs: {name}")
    for name, digest in provenance["data_sha256"].items():
        if file_digest(root / report["config"]["data_dir"] / name) != digest:
            raise ValueError(f"Current processed data differs: {name}")
    checkpoint = paths[winner].parent / "best.pt"
    if file_digest(checkpoint) != report["checkpoints"]["best.pt"]["sha256"]:
        raise ValueError("Selected best checkpoint checksum differs")
    return config_path, checkpoint, report["selection"]


def main():
    """Keep checkpoint, epoch and score overrides out of the test entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hstu-title-selected-test-server.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    recipe = json.loads(args.config.read_text())
    config, checkpoint, selection = admit(ROOT, recipe)
    print("Frozen title report/checkpoint preflight: PASS", flush=True)
    print(json.dumps({"model": recipe["selected_model"], "selection": selection,
                      "checkpoint": str(checkpoint)}, sort_keys=True), flush=True)
    if not args.check_only:
        subprocess.run([sys.executable, "-m", "content_hstu.run", "evaluate", "--config", str(config),
                        "--checkpoint", str(checkpoint), "--split", "test"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
