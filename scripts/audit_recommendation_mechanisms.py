"""Audit frozen validation ranks, title signal and collaborative complementarity."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics.tail_admission import (MODELS, admit, source_hashes, validate_recipe,
                                       verify_payload, verify_unchanged)
from scripts.compare_title_validation import digest, best_validation
from scripts.evaluate_selected_title import file_digest


def write_json(path, value):
    """Publish complete reports atomically, including failures."""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def title_scores(table, ids, mask):
    """Score normalized mean visible-history content against normalized candidates."""
    import torch
    from torch.nn import functional as F

    weights = mask.unsqueeze(-1)
    query = (table[ids] * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
    zero_queries = int((query.norm(dim=-1) <= 1e-12).sum().item())
    return F.normalize(query, dim=-1, eps=1e-12) @ table[1:].T, zero_queries


def run(config_path, root=ROOT):
    """Run inference only; never select an epoch, fit a neural model or score test."""
    root = Path(root)
    recipe = json.loads(Path(config_path).read_text())
    validate_recipe(recipe)
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace not in {":4096:8", ":16:8"}:
        raise ValueError("Invalid deterministic CUDA workspace")
    output_root = (root / recipe["output_dir"]).resolve()
    for folder in ("data", "src", "controls", "content_hstu", "title_features", "diagnostics", "configs"):
        protected = (root / folder).resolve()
        if output_root == protected or protected in output_root.parents:
            raise ValueError("Diagnostic output must not be inside source or data directories")
    now = datetime.now(timezone.utc)
    directory = output_root / now.strftime("%Y%m%dT%H%M%S%fZ")
    directory.mkdir(parents=True, exist_ok=False)
    output = directory / "results.json"
    result = {"status": "RUNNING", "created_at_utc": now.isoformat(), "config": recipe,
              "split": "val", "variants": {}, "checks": {},
              "scope": "mechanism diagnostic only; no training, test evaluation or model selection",
              "protocol": {"catalog": "all mapped items", "filter": "padding and complete visible history",
                           "ties": "ascending mapped item ID", "content_history": "last L training-history events",
                           "itemcf_history": "complete training history, as in the frozen ItemCF baseline",
                           "permutation": "one shared item-to-title permutation within train-frequency buckets",
                           "margins": "target score minus K=20 boundary; zero margin can lose an ID tie",
                           "zero_query": "all scores zero; ascending eligible ID fallback",
                           "score_correction": "none", "quantiles": "linear"}}
    write_json(output, result)
    start = time.perf_counter()
    torch = device = None
    try:
        reports, baseline, checkpoints, tracked = admit(root, recipe)
        tracked[str(Path(config_path).resolve())] = file_digest(Path(config_path))
        diagnostic_sources = source_hashes(root, ("diagnostics",))
        for relative in ("scripts/audit_recommendation_mechanisms.py", "scripts/compare_title_validation.py",
                         "scripts/evaluate_selected_title.py", "scripts/diagnose_hstu_scores.py"):
            diagnostic_sources[relative] = file_digest(root / relative)
        tracked.update({str(root / name): sha for name, sha in diagnostic_sources.items()})
        print("Frozen diagnostic file admission: PASS", flush=True)

        import numpy as np
        import torch as torch_module
        from torch.nn import functional as F
        from collections import Counter
        from content_hstu.artifacts import prepare_content, verify_content
        from content_hstu.model import TitleHSTU
        from diagnostics.tail_statistics import (complementarity, summarize_ranks, target_position,
                                                title_permutation, verify_diagnostic_anchor)
        from src.baselines import ItemCF
        from src.data.dataset import SASRecEvalDataset
        from src.evaluation import RankingAccumulator
        from src.microlens_training import make_loader
        from src.models.id_hstu import IDHSTU
        from src.run_baselines import git_state, load_data
        from title_features.cache import load_cache

        torch = torch_module
        config = reports["id"]["config"]
        torch.manual_seed(config["seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(config["device"])
        runtime = {"python": platform.python_version(), "platform": platform.platform(),
                   "torch": str(torch.__version__), "numpy": np.__version__, "cuda_runtime": torch.version.cuda,
                   "device": str(device), "dtype": "float32", "tf32": False,
                   "deterministic_algorithms": True, "cublas_workspace_config": workspace,
                   "eval_batch_size": config["eval_batch_size"],
                   "packages": dict(sorted((d.metadata["Name"], d.version)
                                           for d in importlib.metadata.distributions() if d.metadata["Name"]))}
        result["runtime"] = runtime
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Configured CUDA device is unavailable")
            device = torch.device("cuda", device.index if device.index is not None else torch.cuda.current_device())
            torch.cuda.set_device(device)
            torch.cuda.reset_peak_memory_stats(device)
            runtime.update(device=str(device), gpu=torch.cuda.get_device_name(device),
                           gpu_total_memory_bytes=torch.cuda.get_device_properties(device).total_memory,
                           gpu_free_memory_bytes=torch.cuda.mem_get_info(device)[0])
        print("Actual diagnostic runtime:", json.dumps(runtime | {"packages": "recorded in results.json"}), flush=True)
        records, num_items, checksums = load_data(root / config["data_dir"])
        # The shared file loader validates the schema; scorers retain no test targets.
        records = [{k: row[k] for k in ("user_id", "train", "val")} for row in records]
        users = [r["user_id"] for r in records]
        identity = {"data_sha256": checksums, "train_users_sha256": digest(users), "eval_users_sha256": digest(users)}
        if digest(identity) != reports["id"]["provenance"]["data_id"]:
            raise ValueError("Actual data and ordered users differ from the training report")
        if reports["id"]["data"] != {"catalog_items": num_items, "train_users": len(records),
                                      "evaluation_users": len(records), "full_data_users": len(records)}:
            raise ValueError("Actual catalog/user counts differ")
        frequency = Counter(item for row in records for item in row["train"])
        counts = np.array([frequency[i] for i in range(num_items + 1)], dtype=np.int64)
        target_frequency = np.array([frequency[r["val"]] for r in records], dtype=np.int64)
        content_config = reports["title-init"]["config"]
        table, _, content, feature_config = prepare_content(content_config, checksums, num_items, records)
        if any(content != reports[name]["content"] for name in MODELS[1:]):
            raise ValueError("Recomputed cache/projection identity differs from frozen title runs")
        features, _ = load_cache(root / content_config["feature_cache"], content["input_sha256"])
        permutation = title_permutation(counts, recipe["permutation_seed"])
        permutation_path = directory / "title_permutation.npy"
        np.save(permutation_path, permutation, allow_pickle=False)
        result["data"] = {"validation_users": len(records), "catalog_items": num_items}
        result["content"] = content
        result["permutation"] = {"seed": recipe["permutation_seed"], "path": permutation_path.name,
                                 "sha256": file_digest(permutation_path),
                                 "moved_items": int((permutation[1:] != np.arange(1, num_items + 1)).sum()),
                                 "scope": "one diagnostic control, not a permutation significance test"}
        result["provenance"] = {"git": git_state(), "config_sha256": digest(recipe),
                                "data_id": digest(identity), **identity,
                                "input_files_sha256": tracked, "diagnostic_sources_sha256": diagnostic_sources,
                                "checkpoints": {name: {"path": str(checkpoints[name]),
                                                        "epoch": reports[name]["selection"]["best_epoch"]} for name in MODELS}}
        dataset = SASRecEvalDataset(root / config["data_dir"] / "sequences.jsonl",
                                    config["model"]["sequence_length"], "val")
        dataset.records = records
        index_by_user = {user: index for index, user in enumerate(users)}
        observations = {"user_id": np.array(users, dtype=np.int64), "target_train_frequency": target_frequency}

        def collect(name, batches, expected=None):
            begin = time.perf_counter()
            accumulator = RankingAccumulator([10, 20], frequency)
            ranks = np.zeros(len(records), dtype=np.int64)
            margins = np.zeros(len(records), dtype=np.float64)
            effective = np.zeros(len(records), dtype=np.int64)
            seen = set()
            zero_queries = 0
            for batch_number, (batch_users, values, zeros) in enumerate(batches, 1):
                zero_queries += zeros
                if values.shape != (len(batch_users), num_items):
                    raise ValueError("Scorer returned a misaligned catalog batch")
                for user, row in zip(batch_users, values):
                    if user in seen or user not in index_by_user:
                        raise ValueError("Duplicate or unknown diagnostic user")
                    seen.add(user)
                    index = index_by_user[user]
                    record = records[index]
                    scores = np.concatenate(([0.0], row))
                    accumulator.add(scores, record["train"], record["val"])
                    ranks[index], margins[index], effective[index] = target_position(
                        scores, record["train"], record["val"], recipe["margin_k"])
                if batch_number % 100 == 0 or len(seen) == len(records):
                    print(f"{name}: scored validation users {len(seen)}/{len(records)}", flush=True)
            if len(seen) != len(records):
                raise ValueError("Diagnostic scorer omitted users")
            evaluation = accumulator.result("val")
            if expected is not None:
                coverage = verify_diagnostic_anchor(name, evaluation, expected, recipe["anchor_atol"])
                result.setdefault("anchor_coverage", {})[name] = coverage
                result["checks"][name + "_validation_reproduced"] = True
                print(f"{name}: frozen validation anchor PASS; coverage={json.dumps(coverage)}", flush=True)
            observations[name + "_rank"] = ranks
            observations[name + "_margin"] = margins
            observations[name + "_effective_k"] = effective
            summary = summarize_ranks(ranks, margins, effective, target_frequency, recipe["rank_cutoffs"])
            result["variants"][name] = {"evaluation": evaluation, "rank_diagnostics": summary,
                                        "zero_content_queries": zero_queries, "seconds": time.perf_counter() - begin}
            write_json(output, result)

        def neural_batches(model=None, content_table=None):
            loader = make_loader(dataset, list(range(len(records))), config["eval_batch_size"],
                                 config["num_workers"], config["seed"], device)
            candidates = torch.arange(1, num_items + 1, device=device)
            with torch.no_grad():
                for batch in loader:
                    ids, mask = batch["input_ids"].to(device), batch["padding_mask"].to(device)
                    if not mask[:, -1].all():
                        raise ValueError("Expected nonempty left-padded validation histories")
                    if model is not None:
                        values = model.score_candidates(model(ids, mask)[:, -1], candidates)
                        zeros = 0
                    else:
                        values, zeros = title_scores(content_table, ids, mask)
                    yield batch["user_id"].tolist(), values.cpu().numpy(), zeros

        for name in MODELS:
            model_config = reports[name]["config"]
            if name == "id":
                model = IDHSTU(num_items=num_items, **model_config["model"])
            else:
                model = TitleHSTU(num_items=num_items, semantic=torch.from_numpy(table.copy()),
                                  representation=name, fusion_weight=model_config["fusion_weight"], **model_config["model"])
            payload = torch.load(checkpoints[name], map_location="cpu", weights_only=True)
            verify_payload(payload, reports[name])
            model.load_state_dict(payload["model_state_dict"], strict=True)
            if name != "id":
                torch.testing.assert_close(model.item_embedding.semantic, torch.from_numpy(table), atol=0, rtol=0)
            del payload
            model.to(device).eval()
            collect(name, neural_batches(model=model), best_validation(reports[name]))
            del model

        fitting_start = time.perf_counter()
        itemcf = ItemCF(records, num_items, baseline["config"]["itemcf_neighbors"])
        result["itemcf_fit_seconds"] = time.perf_counter() - fitting_start

        def itemcf_batches():
            size = config["eval_batch_size"]
            for offset in range(0, len(records), size):
                rows = records[offset:offset + size]
                yield [r["user_id"] for r in rows], np.stack([itemcf.score(r["train"])[1:] for r in rows]), 0

        collect("itemcf", itemcf_batches(), baseline["models"]["itemcf"])
        del itemcf
        result["complementarity"] = complementarity(observations["id_rank"], observations["itemcf_rank"], target_frequency)
        # Use one permutation for both dimensions and both sides of content scoring.
        for dimension, raw in (("384", features), ("64", table)):
            for shuffled in (False, True):
                name = f"title-mean-{dimension}" + ("-permuted" if shuffled else "")
                array = np.array(raw[permutation] if shuffled else raw, dtype=np.float32, copy=True)
                content_table = F.normalize(torch.from_numpy(array).to(device), dim=-1, eps=1e-12)
                collect(name, neural_batches(content_table=content_table))
                del content_table, array
        observation_path = directory / "per_user_diagnostics.npz"
        np.savez_compressed(observation_path, **observations)
        result["per_user_artifact"] = {"path": observation_path.name, "sha256": file_digest(observation_path),
                                       "alignment": "all arrays follow user_id; no held-out test targets stored"}
        verify_content(content_config, content, feature_config)
        verify_unchanged(tracked)
        result["checks"].update(inputs_unchanged=True, all_validation_users_scored=True,
                                no_test_scoring=True, no_neural_training=True)
        result["status"] = "PASS"
        for name, variant in result["variants"].items():
            groups = variant["evaluation"]["groups_by_target_train_frequency"]
            print(json.dumps({"variant": name, "all": groups["all"], "zero": groups["0"],
                              "tail_1_5": groups["1-5"], "tail_ranks": variant["rank_diagnostics"]["1-5"]}), flush=True)
        print("Same-K complementarity:", json.dumps(result["complementarity"]), flush=True)
    except Exception as error:
        result["status"] = "FAIL"
        result["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        print(result["error"]["traceback"], flush=True)
    finally:
        if torch is not None and device is not None and device.type == "cuda" and torch.cuda.is_initialized():
            try:
                torch.cuda.synchronize(device)
                result["memory_bytes"] = {"peak_allocated": torch.cuda.max_memory_allocated(device),
                                          "peak_reserved": torch.cuda.max_memory_reserved(device)}
            except Exception as error:
                result.update(status="FAIL", memory_error=str(error))
        result["elapsed_seconds"] = time.perf_counter() - start
        write_json(output, result)
        print(f"Saved: {output}\nRecommendation mechanism audit: {result['status']}", flush=True)
    if result["status"] != "PASS":
        raise SystemExit(1)
    return result, directory


def main():
    """Expose only a fixed validation diagnostic and a read-only admission check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/recommendation-mechanism-audit-server.json")
    parser.add_argument("--check-only", action="store_true", help="verify artifact files without importing torch or scoring")
    args = parser.parse_args()
    if args.check_only:
        recipe = json.loads(args.config.read_text())
        admit(ROOT, recipe)
        print("Frozen diagnostic file admission: PASS (numerical checks and inference still required)")
    else:
        run(args.config)


if __name__ == "__main__":
    main()
