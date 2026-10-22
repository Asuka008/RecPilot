"""Check paired validation artifacts before one frozen mean-control test evaluation."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from controls.config import validate_control_config


def digest(value):
    """Match the canonical JSON identity used by training reports."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_digest(path):
    """Hash large artifacts without loading them into memory."""
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def validate_inputs(root, mean_path, hstu_path, config_path):
    """Reject mismatched data, recipes, sources or validation-selected checkpoints."""
    mean = json.loads(mean_path.read_text())
    hstu = json.loads(hstu_path.read_text())
    config = json.loads(config_path.read_text())
    reference_path = Path(config["reference_config"])
    reference = json.loads((root / reference_path).read_text())
    validate_control_config(config, reference)
    if config["mode"] != "full" or mean["config"] != config or hstu["config"] != reference:
        raise ValueError("Full training reports must match both frozen configurations")
    mp, hp = mean["provenance"], hstu["provenance"]
    if mp["reference_config"] != reference or mp["reference_config_sha256"] != digest(reference):
        raise ValueError("Reference configuration identity differs")
    for report in (mean, hstu):
        if report["status"] != "PASS" or report["action"] != "train" or not report["checksums_unchanged"]:
            raise ValueError("Both inputs must be successful, unchanged-data training reports")
        if report["provenance"]["config_sha256"] != digest(report["config"]):
            raise ValueError("Training configuration identity differs")
        if [row["epoch"] for row in report["history"]] != list(range(1, config["epochs"] + 1)):
            raise ValueError("Training history must contain all configured epochs in order")
    for key in ("data_id", "data_sha256", "train_users_sha256", "eval_users_sha256"):
        if mp[key] != hp[key]:
            raise ValueError(f"Paired report identity differs: {key}")
    if mp["source_sha256"] != hp["source_sha256"] or mp["shared_source_id"] != hp["source_id"]:
        raise ValueError("Shared model source identities differ")
    shared = mp["source_sha256"]
    control = mp["control_source_sha256"]
    if digest(shared) != mp["shared_source_id"] or digest({"shared": shared, "control": control}) != mp["source_id"]:
        raise ValueError("Source manifest identity differs")
    for relative, expected in {**shared, **control}.items():
        if file_digest(root / relative) != expected:
            raise ValueError(f"Current source differs: {relative}")
    if {str(p.relative_to(root)) for p in (root / "src").rglob("*.py")} != set(shared):
        raise ValueError("Shared source file inventory differs")
    for relative, expected in mp["data_sha256"].items():
        if file_digest(root / config["data_dir"] / relative) != expected:
            raise ValueError(f"Current processed data differs: {relative}")
    selection = mean["selection"]
    metric = config["selection_metric"]
    if selection["split"] != "val" or selection["metric"] != metric:
        raise ValueError("Checkpoint selection must use the configured validation metric")
    for row in mean["history"]:
        if row["validation"]["split"] != "val":
            raise ValueError("Selection history must contain validation evaluations only")
    winner = max(mean["history"], key=lambda row: row["validation"]["groups_by_target_train_frequency"]["all"]["metrics"][metric])
    score = winner["validation"]["groups_by_target_train_frequency"]["all"]["metrics"][metric]
    if selection["best_epoch"] != winner["epoch"] or selection["best_score"] != score:
        raise ValueError("Saved selection differs from the earliest best validation epoch")
    checkpoint = mean_path.parent / "best.pt"
    if file_digest(checkpoint) != mean["checkpoints"]["best.pt"]["sha256"]:
        raise ValueError("Selected best.pt checksum differs")
    return checkpoint, selection


def main():
    """Offer a read-only preflight without exposing scoring or selection overrides."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mean-report", type=Path, required=True)
    parser.add_argument("--hstu-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/mean-pooling-train-server.json")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    checkpoint, selection = validate_inputs(ROOT, args.mean_report.resolve(), args.hstu_report.resolve(), args.config.resolve())
    print("Frozen mean preflight: PASS", flush=True)
    print(json.dumps({"selection": selection, "checkpoint": str(checkpoint)}, sort_keys=True), flush=True)
    if not args.check_only:
        subprocess.run([sys.executable, "-m", "controls.run_mean_pooling", "evaluate",
                        "--config", str(args.config.resolve()), "--checkpoint", str(checkpoint),
                        "--split", "test"], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
