"""Authenticate paired validation artifacts before diagnostic inference."""

import json
import math
from pathlib import Path

from scripts.compare_title_validation import compare_reports, digest
from scripts.evaluate_selected_title import file_digest

MODELS = ("id", "title-init", "title-fusion")


def validate_recipe(recipe):
    """Reject test evaluation and undeclared diagnostic settings."""
    keys = {"reports", "training_configs", "baseline_report", "baseline_config", "output_dir",
            "split", "permutation_seed", "rank_cutoffs", "margin_k", "anchor_atol"}
    if set(recipe) != keys or recipe["split"] != "val":
        raise ValueError("Mechanism audit requires the documented keys and split=val")
    for key in ("reports", "training_configs"):
        if not isinstance(recipe[key], dict) or set(recipe[key]) != set(MODELS):
            raise ValueError(f"Expected all three models in {key}")
    paths = [recipe[k] for k in ("baseline_report", "baseline_config", "output_dir")]
    paths += list(recipe["reports"].values()) + list(recipe["training_configs"].values())
    if any(not isinstance(p, str) or not p.strip() for p in paths):
        raise ValueError("Paths must be nonempty strings")
    if type(recipe["permutation_seed"]) is not int or not 0 <= recipe["permutation_seed"] < 2**32:
        raise ValueError("Invalid permutation seed")
    if recipe["rank_cutoffs"] != [20, 100, 500, 1000] or type(recipe["margin_k"]) is not int or recipe["margin_k"] != 20:
        raise ValueError("Rank cutoffs and margin K must match the preregistered diagnostic")
    value = recipe["anchor_atol"]
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1e-6:
        raise ValueError("Invalid anchor tolerance")


def source_hashes(root, directories):
    """Include the full source inventory, so added model files cannot hide."""
    return {str(p.relative_to(root)): file_digest(p)
            for directory in directories for p in sorted((root / directory).rglob("*.py"))}


def admit(root, recipe):
    """Check all reports, current sources, configs, inputs and recorded best files.

    This is a file-level check. Model payload and numerical cache validation
    happen before scoring in the runner.
    """
    root = Path(root)
    validate_recipe(recipe)
    tracked = {}

    def track(path, expected=None):
        path = (root / path).resolve()
        actual = file_digest(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Artifact checksum differs: {path}")
        tracked[str(path)] = actual
        return path

    reports = {name: json.loads(track(recipe["reports"][name]).read_text()) for name in MODELS}
    compare_reports(reports)
    shared = source_hashes(root, ("src",))
    content = source_hashes(root, ("content_hstu", "title_features"))
    checkpoints = {}
    for name, report in reports.items():
        config, provenance = report["config"], report["provenance"]
        if json.loads(track(recipe["training_configs"][name]).read_text()) != config:
            raise ValueError(f"Current configuration differs: {name}")
        if config["ks"] != [10, 20] or config["train_users"] is not None or config["eval_users"] is not None:
            raise ValueError("Audit requires all users and original Recall/NDCG@10/20")
        expected_source = digest(shared)
        if provenance["source_sha256"] != shared:
            raise ValueError(f"Shared source inventory differs: {name}")
        if name != "id":
            if (provenance["content_source_sha256"] != content
                    or provenance["shared_source_id"] != digest(shared)
                    or json.loads(track(config["reference_config"]).read_text()) != reports["id"]["config"]):
                raise ValueError(f"Content source or reference configuration differs: {name}")
            expected_source = digest({"shared": shared, "content": content})
            cache = root / config["feature_cache"]
            manifest = json.loads(track(cache / "results.json", report["content"]["manifest_sha256"]).read_text())
            track(cache / "features.npy", config["features_sha256"])
            if (report["content"]["content_id"] != provenance["content_id"]
                    or provenance["content"] != report["content"]):
                raise ValueError("Content report identity differs")
            for item, sha in manifest["input_sha256"].items():
                path = (root / manifest["config"]["titles_csv"] if item == "titles_csv" else
                        root / manifest["config"]["data_dir"] / item)
                track(path, sha)
        if provenance["source_id"] != expected_source:
            raise ValueError(f"Source identity differs: {name}")
        for item, sha in provenance["data_sha256"].items():
            track(root / config["data_dir"] / item, sha)
        checkpoints[name] = track((root / recipe["reports"][name]).parent / "best.pt",
                                  report["checkpoints"]["best.pt"]["sha256"])
    for item, sha in {**shared, **content}.items():
        track(item, sha)
    baseline = json.loads(track(recipe["baseline_report"]).read_text())
    bc = json.loads(track(recipe["baseline_config"]).read_text())
    if (baseline["config"] != bc or baseline["split"] != "val" or bc["ks"] != [10, 20]
            or baseline["data_sha256"] != reports["id"]["provenance"]["data_sha256"]
            or "itemcf" not in baseline["models"] or baseline["models"]["itemcf"]["split"] != "val"
            or type(bc["itemcf_neighbors"]) is not int or bc["itemcf_neighbors"] < 1):
        raise ValueError("ItemCF validation report/config/data differs")
    return reports, baseline, checkpoints, tracked


def verify_unchanged(tracked):
    """Refuse to publish success if any admitted input changed during the run."""
    for path, expected in tracked.items():
        if file_digest(Path(path)) != expected:
            raise RuntimeError(f"Input changed during audit: {path}")


def verify_payload(payload, report):
    """Check state-dict metadata independently from the outer file hash."""
    if (payload["schema_version"] != 1 or payload["config"] != report["config"]
            or payload["epoch"] != report["selection"]["best_epoch"]):
        raise ValueError("Checkpoint configuration or selected epoch differs")
    keys = ["data_id", "source_id", "data_sha256", "train_users_sha256", "eval_users_sha256"]
    if "content_id" in report["provenance"]:
        keys += ["content_id", "reference_config_sha256"]
    if any(payload["provenance"][k] != report["provenance"][k] for k in keys):
        raise ValueError("Checkpoint provenance differs from admitted report")
