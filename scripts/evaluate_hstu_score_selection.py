"""Evaluate exactly one validation-selected correction recipe on the frozen test split."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import diagnose_hstu_scores as scoring
from src.evaluation import RankingAccumulator


def validate_test_config(config):
    """Expose no alpha override or test-time parameter search."""
    if set(config) != {"validation_report", "output_dir"} or any(
        not isinstance(value, str) or not value.strip() for value in config.values()
    ):
        raise ValueError("Expected validation_report and output_dir only; no alpha override")


def selected_recipe(audit):
    """Recompute validation selection, including deterministic ties, before testing."""
    if audit.get("status") != "PASS" or audit.get("split") != "val":
        raise ValueError("A completed validation audit with status PASS is required")
    scoring.validate_audit_config(audit["config"])
    checks = audit.get("checks", {})
    if checks.get("alpha_zero_reproduces_frozen_validation") is not True or checks.get("inputs_unchanged") is not True:
        raise ValueError("Validation anchor or input integrity checks did not pass")
    variants = audit["variants"]
    if [v["alpha"] for v in variants] != audit["config"]["alphas"]:
        raise ValueError("Recorded variants differ from the declared validation grid")
    metric = audit["provenance"]["training_config"]["selection_metric"]
    for variant in variants:
        value = variant["evaluation"]["groups_by_target_train_frequency"]["all"]["metrics"][metric]
        if variant["evaluation"]["split"] != "val" or not math.isfinite(value):
            raise ValueError("Selection requires finite validation metrics")
    winner = max(variants, key=lambda v: v["evaluation"]["groups_by_target_train_frequency"]["all"]["metrics"][metric])
    selection = audit["selection"]
    value = winner["evaluation"]["groups_by_target_train_frequency"]["all"]["metrics"][metric]
    if selection["metric"] != metric or selection["alpha"] != winner["alpha"] or selection["value"] != value:
        raise ValueError("Saved selection does not match the validation maximum and tie rule")
    return {"alpha": winner["alpha"], "smoothing": audit["config"]["smoothing"],
            "selection_metric": metric, "validation_score": value}


def run_selected_test(config_path):
    """Verify the frozen validation artifact, then score each test user once."""
    config = json.loads(Path(config_path).read_text())
    validate_test_config(config)
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace not in {":4096:8", ":16:8"}:
        raise ValueError("Invalid deterministic CUBLAS workspace")
    now = datetime.now(timezone.utc)
    run_dir = scoring.resolved(config["output_dir"]) / now.strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    output = run_dir / "results.json"
    report = {"status": "RUNNING", "created_at_utc": now.isoformat(), "config": config,
              "split": "test", "scope": "one frozen validation-selected correction; no test search or fitting"}
    torch = device = None
    start = time.perf_counter()

    def save():
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(output)

    save()
    try:
        from src.run_baselines import file_sha256, git_state, load_data
        from src.toy_training import json_digest
        from src.microlens_config import validate_config

        audit_path = scoring.resolved(config["validation_report"])
        audit_sha = file_sha256(audit_path)
        audit = json.loads(audit_path.read_text())
        recipe = selected_recipe(audit)
        report["recipe"] = recipe
        if file_sha256(Path(scoring.__file__)) != audit["provenance"]["diagnostic_script_sha256"]:
            raise ValueError("Scoring implementation differs from the validated diagnostic script")
        if json_digest(audit["config"]) != audit["provenance"]["audit_config_sha256"]:
            raise ValueError("Validation audit configuration identity differs")
        training_config = json.loads(scoring.resolved(audit["config"]["training_config"]).read_text())
        validate_config(training_config)
        if training_config != audit["provenance"]["training_config"] or training_config["mode"] != "full":
            raise ValueError("Training configuration differs from the validation artifact")
        training_dir = scoring.resolved(audit["config"]["training_run"])
        training_report_path = training_dir / "results.json"
        if file_sha256(training_report_path) != audit["provenance"]["training_report_sha256"]:
            raise ValueError("Training report differs from the validation artifact")
        reference = json.loads(training_report_path.read_text())
        checkpoint = training_dir / "best.pt"
        checkpoint_sha = file_sha256(checkpoint)
        if checkpoint_sha != audit["provenance"]["checkpoint_sha256"]:
            raise ValueError("Checkpoint differs from the validation artifact")
        records, num_items, checksums = load_data(scoring.resolved(training_config["data_dir"]))
        if checksums != audit["provenance"]["data_sha256"]:
            raise ValueError("Processed data differs from the validation artifact")
        source_hashes = {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted((ROOT / "src").rglob("*.py"))}
        source_id = json_digest(source_hashes)
        if source_id != audit["provenance"]["model_source_id"]:
            raise ValueError("Model source differs from the validation artifact")
        import torch as torch_module
        from src.data.dataset import SASRecEvalDataset
        from src.microlens_training import make_loader
        from src.models.id_hstu import IDHSTU

        torch = torch_module
        device = torch.device(training_config["device"])
        torch.manual_seed(training_config["seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        runtime = {"python": platform.python_version(), "torch": str(torch.__version__), "numpy": np.__version__,
                   "cuda_runtime": torch.version.cuda, "device": str(device), "dtype": "float32", "tf32": False,
                   "cublas_workspace_config": workspace, "deterministic_algorithms": True,
                   "packages": dict(sorted((d.metadata["Name"], d.version)
                                           for d in importlib.metadata.distributions() if d.metadata["Name"]))}
        report["runtime"] = runtime
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Configured CUDA device is unavailable")
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
            runtime.update(device=str(device), gpu=torch.cuda.get_device_name(device),
                           gpu_total_memory_bytes=torch.cuda.get_device_properties(device).total_memory)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (payload["schema_version"] != 1 or payload["config"] != training_config
                or payload["epoch"] != audit["provenance"]["checkpoint_epoch"]
                or payload["provenance"]["source_id"] != source_id
                or payload["provenance"]["data_id"] != reference["provenance"]["data_id"]):
            raise ValueError("Checkpoint epoch, configuration, data or model identity differs")
        model = IDHSTU(num_items=num_items, **training_config["model"]).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        del payload
        length = training_config["model"]["sequence_length"]
        history_counts, positive_counts = scoring.training_counts(records, num_items, length)
        frequency = Counter({i: int(c) for i, c in enumerate(history_counts) if i and c})
        offset = scoring.corrected_scores(np.zeros(num_items + 1), positive_counts,
                                          training_config["temperature"], recipe["alpha"], recipe["smoothing"])
        dataset = SASRecEvalDataset(scoring.resolved(training_config["data_dir"]) / "sequences.jsonl", length, "test")
        by_user = {record["user_id"]: record for record in records}
        accumulator = RankingAccumulator(training_config["ks"], frequency)
        loader = make_loader(dataset, list(range(len(records))), training_config["eval_batch_size"],
                             training_config["num_workers"], training_config["seed"], device)
        candidates = torch.arange(1, num_items + 1, device=device)
        with torch.no_grad():
            for batch in loader:
                ids, mask = batch["input_ids"].to(device), batch["padding_mask"].to(device)
                if not mask[:, -1].all():
                    raise ValueError("Test evaluation requires nonempty left-padded histories")
                values = model.score_candidates(model(ids, mask)[:, -1], candidates).cpu().numpy()
                for user_id, row in zip(batch["user_id"].tolist(), values):
                    record = by_user[user_id]
                    history = record["train"] + [record["val"]]
                    accumulator.add(np.concatenate(([0.0], row)) + offset, history, record["test"])
        report["evaluation"] = accumulator.result("test")
        report["data"] = {"catalog_items": num_items, "evaluation_users": len(records)}
        report["provenance"] = {"git": git_state(), "validation_report_sha256": audit_sha,
                                "validation_report": str(audit_path), "checkpoint_sha256": checkpoint_sha,
                                "checkpoint_epoch": audit["provenance"]["checkpoint_epoch"],
                                "data_sha256": checksums, "model_source_id": source_id,
                                "scoring_script_sha256": file_sha256(Path(scoring.__file__)),
                                "evaluation_script_sha256": file_sha256(Path(__file__)), "training_config": training_config}
        if (file_sha256(audit_path) != audit_sha or file_sha256(checkpoint) != checkpoint_sha
                or file_sha256(training_report_path) != audit["provenance"]["training_report_sha256"]
                or any(file_sha256(scoring.resolved(training_config["data_dir"]) / name) != sha for name, sha in checksums.items())):
            raise RuntimeError("Frozen input artifact changed during evaluation")
        report["checks"] = {"one_validation_selected_recipe": True, "inputs_unchanged": True}
        report["status"] = "PASS"
        print("Frozen recipe:", recipe, flush=True)
        print(json.dumps(report["evaluation"], sort_keys=True), flush=True)
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        print(report["error"]["traceback"], flush=True)
    finally:
        if torch is not None and device is not None and device.type == "cuda" and torch.cuda.is_initialized():
            try:
                torch.cuda.synchronize(device)
                report["memory_bytes"] = {"peak_allocated": torch.cuda.max_memory_allocated(device),
                                          "peak_reserved": torch.cuda.max_memory_reserved(device)}
            except Exception as error:
                report["status"] = "FAIL"
                report["memory_error"] = str(error)
        report["elapsed_seconds"] = time.perf_counter() - start
        save()
        print(f"Saved: {output}", flush=True)
        print(f"HSTU selected-score test: {report['status']}", flush=True)
    if report["status"] != "PASS":
        raise SystemExit(1)
    return report, run_dir


def main():
    """Evaluate the saved winner; alpha and selection metric cannot be overridden."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hstu-score-selected-test-server.json")
    run_selected_test(parser.parse_args().config)


if __name__ == "__main__":
    main()
