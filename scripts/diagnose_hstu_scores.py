"""Validation-only frequency correction and coverage audit of a frozen HSTU."""

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
# Direct script execution otherwise puts only scripts/ on the import path.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import RankingAccumulator, rank_candidates


def validate_audit_config(config):
    """Predeclare a small grid and forbid held-out test evaluation."""
    keys = {"training_config", "training_run", "output_dir", "split", "alphas", "smoothing", "anchor_atol"}
    if set(config) != keys or config["split"] != "val":
        raise ValueError("Score audit requires exactly the documented keys and split=val")
    for key in ("training_config", "training_run", "output_dir"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"Invalid path: {key}")
    alphas = config["alphas"]
    if not isinstance(alphas, list) or not alphas or any(
        type(a) not in (int, float) or not math.isfinite(a) or a < 0 for a in alphas
    ):
        raise ValueError("alphas must be finite nonnegative numbers")
    if alphas != sorted(set(alphas)) or alphas[0] != 0:
        raise ValueError("alphas must be unique, ascending, and start at zero")
    for key in ("smoothing", "anchor_atol"):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid finite nonnegative value: {key}")
    if config["smoothing"] <= 0 or config["anchor_atol"] > 1e-6:
        raise ValueError("smoothing must be positive and anchor_atol cannot exceed 1e-6")


def training_counts(records, num_items, length):
    """Return full history counts and actual retained positive-label counts."""
    history, positive = np.zeros(num_items + 1, dtype=np.int64), np.zeros(num_items + 1, dtype=np.int64)
    for record in records:
        np.add.at(history, record["train"], 1)
        np.add.at(positive, record["train"][-(length + 1):][1:], 1)
    return history, positive


def corrected_scores(scores, positive_counts, temperature, alpha, smoothing):
    """Rank as cosine/temperature - alpha*log(positive_count+smoothing).

    Return on the cosine scale to keep alpha=0 exactly identical to the old
    ranking, including ties. Candidate eligibility is handled by shared ranking.
    """
    scores = np.asarray(scores, dtype=np.float64)
    counts = np.asarray(positive_counts)
    if scores.ndim != 1 or counts.shape != scores.shape or not np.isfinite(scores).all():
        raise ValueError("Scores and counts must be aligned finite catalog vectors")
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Counts must be finite and nonnegative")
    if not all(math.isfinite(v) for v in (temperature, alpha, smoothing)) or temperature <= 0 or alpha < 0 or smoothing <= 0:
        raise ValueError("Invalid correction parameters")
    result = scores.copy() if alpha == 0 else scores - temperature * alpha * np.log(counts + smoothing)
    result[0] = scores[0]
    if not np.isfinite(result).all():
        raise ValueError("Correction produced nonfinite scores")
    return result


def verify_anchor(actual, expected, atol):
    """Require all existing groups, counts and metrics to reproduce the saved run."""
    if actual.get("split") != "val" or expected.get("split") != "val":
        raise AssertionError("Anchor must refer to validation")
    for axis in ("groups_by_target_train_frequency", "groups_by_history_length"):
        if actual[axis].keys() != expected[axis].keys():
            raise AssertionError(f"Anchor groups differ: {axis}")
        for name, group in expected[axis].items():
            observed = actual[axis][name]
            if observed["users"] != group["users"] or observed["metrics"].keys() != group["metrics"].keys():
                raise AssertionError(f"Anchor counts or metric names differ: {axis}/{name}")
            for key, value in group["metrics"].items():
                other = observed["metrics"][key]
                if (value is None) != (other is None) or (
                    value is not None and (not math.isfinite(value) or not math.isfinite(other) or abs(value - other) > atol)
                ):
                    raise AssertionError(f"Uncorrected anchor differs: {axis}/{name}/{key}")


def resolved(path):
    """Resolve repository-relative experiment paths without changing the CWD."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def run_audit(config_path):
    """Reuse each neural score batch across all correction strengths, without fitting."""
    config = json.loads(Path(config_path).read_text())
    validate_audit_config(config)
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace not in {":4096:8", ":16:8"}:
        raise ValueError("Invalid deterministic CUBLAS workspace")
    now = datetime.now(timezone.utc)
    run_dir = resolved(config["output_dir"]) / now.strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "created_at_utc": now.isoformat(), "config": config, "split": "val", "variants": [],
              "scope": "frozen-checkpoint validation scoring diagnostic; no test evaluation or training",
              "formula": "cosine - temperature * alpha * log(retained_positive_count + smoothing)",
              "frequency_groups": "full training-history counts; same as frozen baseline",
              "correction_counts": "retained shifted positive-label events from training only"}
    torch = device = None
    start = time.perf_counter()
    output = run_dir / "results.json"

    def save():
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        temporary.replace(output)

    save()
    try:
        import torch as torch_module
        from src.data.dataset import SASRecEvalDataset
        from src.microlens_config import validate_config
        from src.microlens_training import make_loader
        from src.models.id_hstu import IDHSTU
        from src.run_baselines import file_sha256, git_state, load_data
        from src.toy_training import json_digest

        torch = torch_module
        training_config = json.loads(resolved(config["training_config"]).read_text())
        validate_config(training_config)
        if training_config["mode"] != "full":
            raise ValueError("Only completed full-data runs can be audited")
        training_dir = resolved(config["training_run"])
        training_report_path = training_dir / "results.json"
        reference_sha = file_sha256(training_report_path)
        reference = json.loads(training_report_path.read_text())
        if reference["status"] != "PASS" or reference["action"] != "train" or reference["config"] != training_config:
            raise ValueError("Training report must be a completed matching training run")
        if reference["selection"]["split"] != "val":
            raise ValueError("Reference checkpoint must be selected on validation")
        records, num_items, checksums = load_data(resolved(training_config["data_dir"]))
        report["data"] = {"validation_users": len(records), "catalog_items": num_items}
        if checksums != reference["provenance"]["data_sha256"]:
            raise ValueError("Processed data differs from the frozen training run")
        selected = next(row for row in reference["history"] if row["epoch"] == reference["selection"]["best_epoch"])
        checkpoint = training_dir / "best.pt"
        checkpoint_sha = file_sha256(checkpoint)
        if checkpoint_sha != reference["checkpoints"]["best.pt"]["sha256"]:
            raise ValueError("best.pt checksum differs from the training report")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        source_hashes = {str(p.relative_to(ROOT)): file_sha256(p) for p in sorted((ROOT / "src").rglob("*.py"))}
        if (payload["schema_version"] != 1 or payload["config"] != training_config
                or payload["epoch"] != reference["selection"]["best_epoch"]
                or payload["provenance"]["data_id"] != reference["provenance"]["data_id"]
                or payload["provenance"]["source_id"] != json_digest(source_hashes)):
            raise ValueError("Checkpoint configuration, epoch, data or model source identity differs")
        device = torch.device(training_config["device"])
        torch.manual_seed(training_config["seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        runtime = {"torch": str(torch.__version__), "numpy": np.__version__, "python": platform.python_version(),
                   "cuda_runtime": torch.version.cuda, "device": str(device), "dtype": "float32", "tf32": False,
                   "platform": platform.platform(), "cublas_workspace_config": workspace,
                   "deterministic_algorithms": True,
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
        model = IDHSTU(num_items=num_items, **training_config["model"]).to(device)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        del payload
        length = training_config["model"]["sequence_length"]
        history_counts, positive_counts = training_counts(records, num_items, length)
        offsets = [corrected_scores(np.zeros(num_items + 1), positive_counts,
                                    training_config["temperature"], alpha, config["smoothing"])
                   for alpha in config["alphas"]]
        frequency = Counter({i: int(c) for i, c in enumerate(history_counts) if i and c})
        accumulators = [RankingAccumulator(training_config["ks"], frequency) for _ in config["alphas"]]
        recommendations = [Counter() for _ in config["alphas"]]
        dataset = SASRecEvalDataset(resolved(training_config["data_dir"]) / "sequences.jsonl", length, "val")
        by_user = {record["user_id"]: record for record in records}
        loader = make_loader(dataset, list(range(len(records))), training_config["eval_batch_size"],
                             training_config["num_workers"], training_config["seed"], device)
        candidates = torch.arange(1, num_items + 1, device=device)
        with torch.no_grad():
            for batch_number, batch in enumerate(loader, start=1):
                ids, mask = batch["input_ids"].to(device), batch["padding_mask"].to(device)
                if not mask[:, -1].all():
                    raise ValueError("Audit requires nonempty left-padded evaluation histories")
                values = model.score_candidates(model(ids, mask)[:, -1], candidates).cpu().numpy()
                for user_id, row in zip(batch["user_id"].tolist(), values):
                    record = by_user[user_id]
                    scores = np.concatenate(([0.0], row))
                    for index, offset in enumerate(offsets):
                        adjusted = scores + offset
                        accumulators[index].add(adjusted, record["train"], record["val"])
                        recommendations[index].update(rank_candidates(adjusted, record["train"], max(training_config["ks"])))
                if batch_number % 100 == 0 or batch_number == len(loader):
                    print(f"Scored validation batches {batch_number}/{len(loader)}; "
                          f"correction strengths={len(config['alphas'])}", flush=True)
        for alpha, accumulator, recommended in zip(config["alphas"], accumulators, recommendations):
            counts = dict.fromkeys(("0", "1-5", "6-20", "21+"), 0)
            for item, count in recommended.items():
                n = history_counts[item]
                bucket = "0" if n == 0 else "1-5" if n <= 5 else "6-20" if n <= 20 else "21+"
                counts[bucket] += count
            total = sum(counts.values())
            report["variants"].append({"alpha": alpha, "evaluation": accumulator.result("val"),
                "recommendations": {"k": max(training_config["ks"]), "unique_items": len(recommended),
                                    "catalog_coverage": len(recommended) / num_items,
                                    "slots": total, "slots_by_history_frequency": counts,
                                    "shares_by_history_frequency": {k: v/total for k, v in counts.items()}}})
        verify_anchor(report["variants"][0]["evaluation"], selected["validation"], config["anchor_atol"])
        metric = training_config["selection_metric"]
        winner = max(report["variants"], key=lambda variant: variant["evaluation"]["groups_by_target_train_frequency"]["all"]["metrics"][metric])
        report["selection"] = {"metric": metric, "alpha": winner["alpha"],
                               "value": winner["evaluation"]["groups_by_target_train_frequency"]["all"]["metrics"][metric],
                               "ties": "smaller alpha wins; diagnostic result only"}
        report["provenance"] = {"git": git_state(), "training_config": training_config,
                                "training_report_sha256": reference_sha,
                                "checkpoint_sha256": checkpoint_sha, "checkpoint_epoch": reference["selection"]["best_epoch"],
                                "data_sha256": checksums, "model_source_id": json_digest(source_hashes),
                                "diagnostic_script_sha256": file_sha256(Path(__file__)),
                                "audit_config_sha256": json_digest(config)}
        if (file_sha256(checkpoint) != checkpoint_sha
                or file_sha256(training_report_path) != report["provenance"]["training_report_sha256"]
                or any(file_sha256(resolved(training_config["data_dir"]) / name) != value for name, value in checksums.items())):
            raise RuntimeError("Input checkpoint, report or processed data changed during audit")
        report["checks"] = {"alpha_zero_reproduces_frozen_validation": True, "inputs_unchanged": True}
        report["status"] = "PASS"
        for variant in report["variants"]:
            groups = variant["evaluation"]["groups_by_target_train_frequency"]
            print(json.dumps({"alpha": variant["alpha"], "all": groups["all"],
                              "tail_1_5": groups["1-5"], "mid_6_20": groups["6-20"],
                              "recommendations": variant["recommendations"]}), flush=True)
        print("Diagnostic validation selection:", report["selection"], flush=True)
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
        print(f"HSTU score audit: {report['status']}", flush=True)
    if report["status"] != "PASS":
        raise SystemExit(1)
    return report, run_dir


def main():
    """Expose validation scoring only; there is deliberately no test switch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hstu-score-audit-server.json")
    run_audit(parser.parse_args().config)


if __name__ == "__main__":
    main()
