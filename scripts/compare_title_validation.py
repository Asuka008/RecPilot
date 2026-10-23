"""Read paired full-training reports without model execution or test evaluation."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from content_hstu.config import validate_content_config
from src.microlens_config import validate_config

AXES = {"groups_by_target_train_frequency": ("all", "0", "1-5", "6-20", "21+"),
        "groups_by_history_length": ("all", "1-5", "6-10", "11-20", "21+")}


def digest(value):
    """Match the canonical configuration digest used in training."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def best_validation(report):
    """Require a complete run and the earliest best validation epoch."""
    config = report["config"]
    if (report["status"] != "PASS" or report["action"] != "train"
            or config["mode"] != "full" or not report["checksums_unchanged"]):
        raise ValueError("Expected a successful full training report with unchanged data")
    if report["provenance"]["config_sha256"] != digest(config):
        raise ValueError("Configuration digest differs")
    if [r["epoch"] for r in report["history"]] != list(range(1, config["epochs"] + 1)):
        raise ValueError("Incomplete epoch history")
    metric = config["selection_metric"]
    scores = []
    for row in report["history"]:
        validation = row["validation"]
        if validation["split"] != "val":
            raise ValueError("Only validation history can select an epoch")
        value = validation["groups_by_target_train_frequency"]["all"]["metrics"][metric]
        if not math.isfinite(value):
            raise ValueError("Nonfinite selection metric")
        scores.append(value)
    index = max(range(len(scores)), key=scores.__getitem__)
    selection = {"metric": metric, "best_score": scores[index], "best_epoch": index + 1, "split": "val"}
    if report["selection"] != selection:
        raise ValueError("Selection differs from earliest validation winner")
    return report["history"][index]["validation"]


def compare_reports(reports):
    """Check report-level pairing, then return exact metrics and recorded resources.

    This does not authenticate checkpoints or inspect current data/source files.
    A separate frozen evaluation admission check is needed before any test run.
    """
    if set(reports) != {"id", "title-init", "title-fusion"}:
        raise ValueError("Expected ID, initialization and fusion reports")
    baseline = reports["id"]
    validate_config(baseline["config"])
    bp = baseline["provenance"]
    selected = {name: best_validation(r) for name, r in reports.items()}
    for name in ("title-init", "title-fusion"):
        report = reports[name]
        validate_content_config(report["config"], baseline["config"])
        p = report["provenance"]
        if report["config"]["representation"] != name or not report["frozen_content_unchanged"]:
            raise ValueError("Representation or frozen content check differs")
        if p["reference_config"] != baseline["config"] or p["reference_config_sha256"] != digest(baseline["config"]):
            raise ValueError("Reference configuration differs")
        for key in ("data_id", "data_sha256", "train_users_sha256", "eval_users_sha256", "source_sha256"):
            if p[key] != bp[key]:
                raise ValueError(f"Paired report identity differs: {key}")
        if p["shared_source_id"] != bp["source_id"]:
            raise ValueError("Shared source identity differs")
        if report["data"] != baseline["data"]:
            raise ValueError("Catalog or user counts differ")
    if reports["title-init"]["provenance"]["content_id"] != reports["title-fusion"]["provenance"]["content_id"]:
        raise ValueError("Title experiments use different cached features or projections")
    for axis, groups in AXES.items():
        for name, validation in selected.items():
            for group in groups:
                entry = validation[axis][group]
                if entry["users"] != selected["id"][axis][group]["users"]:
                    raise ValueError(f"Group counts differ: {name}/{axis}/{group}")
                if any(not math.isfinite(v) for v in entry["metrics"].values()):
                    raise ValueError("Nonfinite group metric")
    summary = {}
    for name, report in reports.items():
        summary[name] = {"selection": report["selection"], "validation": selected[name],
                         "checks": report.get("checks"), "elapsed_seconds": report.get("elapsed_seconds"),
                         "memory_bytes": report.get("memory_bytes"),
                         "runtime": {k: report["runtime"].get(k) for k in
                                     ("torch", "numpy", "cuda_runtime", "device", "gpu", "gpu_total_memory_bytes", "dtype", "tf32")},
                         "git": report["provenance"].get("git"),
                         "projection": report.get("content", {}).get("projection")}
    return summary


def main():
    """Print compact paired tables and exact report fields without writing artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("id", "init", "fusion"):
        parser.add_argument(f"--{name}-report", type=Path, required=True)
    args = parser.parse_args()
    paths = {"id": args.id_report, "title-init": args.init_report, "title-fusion": args.fusion_report}
    reports = {name: json.loads(path.read_text()) for name, path in paths.items()}
    summary = compare_reports(reports)
    print("Validation report pairing: PASS (report fields only; no checkpoint or test execution)")
    for axis, groups in AXES.items():
        print(f"\n{axis}: group users | ID N10/R20 | init N10/R20 | fusion N10/R20")
        for group in groups:
            values = [summary[name]["validation"][axis][group] for name in paths]
            metrics = " | ".join(f"{v['metrics']['NDCG@10']:.8f}/{v['metrics']['Recall@20']:.8f}" for v in values)
            print(f"{group} {values[0]['users']} | {metrics}")
    print("\nExact selected metrics and resources:")
    for name, result in summary.items():
        print(json.dumps({"model": name, **result}, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
