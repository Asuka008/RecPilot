"""Train or evaluate an uncorrected causal prefix-mean control matched to HSTU."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import time
import traceback

from src.microlens_config import select_indices
from controls.config import resolved, validate_control_config


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    """Publish complete JSON reports atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def run(args):
    """Record failures as well as successful epoch-boundary checkpoints."""
    config = json.loads(args.config.read_text())
    reference_config = json.loads(resolved(config["reference_config"]).read_text())
    validate_control_config(config, reference_config)
    if args.action == "evaluate" and args.split == "test" and config["mode"] != "full":
        raise ValueError("Smoke configurations cannot evaluate test")
    if args.action == "train" and args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= config["epochs"]:
        raise ValueError("stop-after-epoch must be between 1 and configured epochs")
    if args.action == "train" and args.resume is not None and args.resume.name != "last.pt":
        raise ValueError("Continuation requires last.pt and its original sibling best.pt")
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace not in {":4096:8", ":16:8"}:
        raise ValueError("Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    now = datetime.now(timezone.utc)
    output_root = Path(config["output_dir"])
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    run_dir = output_root / now.strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "action": args.action, "created_at_utc": now.isoformat(),
              "config": config, "history": [], "scope": config["mode"], "control": config["control"],
              "protocol": {"name": "per-user-leave-one-out-full-catalog-v1",
                           "filter": "padding and full visible history",
                           "ties": "ascending mapped item ID", "catalog_assumed_known": True,
                           "zero_frequency_targets": "retained", "repeat_targets": "fail explicitly",
                           "loss": "full-catalog cross entropy, mean over observed shifted positions",
                           "precision": "FP32; TF32 disabled", "encoder": "causal prefix mean",
                           "score_correction": "none",
                           "selection": "validation metric; earliest epoch wins exact ties"}}
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "results.json", report)
    torch = device = None
    start = time.perf_counter()
    try:
        import numpy as np
        import torch as torch_module
        from src.checkpoints import load_checkpoint, save_checkpoint
        from src.data.dataset import SASRecEvalDataset, SASRecTrainDataset
        from src.microlens_training import (evaluate_hstu, fixed_batch_loss, make_loader,
                                            move_batch, train_epoch)
        from controls.mean_pooling import PrefixMean
        from src.run_baselines import file_sha256, git_state, load_data
        from src.toy_training import json_digest

        torch = torch_module
        device = torch.device(config["device"])
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        runtime = {"python": platform.python_version(), "platform": platform.platform(),
                   "torch": str(torch.__version__), "numpy": np.__version__,
                   "cuda_runtime": torch.version.cuda, "device": str(device),
                   "cublas_workspace_config": workspace, "deterministic_algorithms": True,
                   "dtype": "float32", "tf32": False,
                   "packages": dict(sorted((d.metadata["Name"], d.version)
                                           for d in importlib.metadata.distributions() if d.metadata["Name"]))}
        report["runtime"] = runtime
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Configured CUDA device is unavailable")
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
                runtime["device"] = str(device)
            torch.cuda.set_device(device)
            torch.cuda.manual_seed_all(config["seed"])
            torch.cuda.reset_peak_memory_stats(device)
            runtime["gpu"] = torch.cuda.get_device_name(device)
            runtime["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(device).total_memory
        data_dir = Path(config["data_dir"])
        if not data_dir.is_absolute():
            data_dir = ROOT / data_dir
        records, num_items, checksums = load_data(data_dir)
        train_indices = select_indices(len(records), config["train_users"], config["seed"])
        eval_indices = select_indices(len(records), config["eval_users"], config["seed"] + 1)
        identity = {"data_sha256": checksums,
                    "train_users_sha256": json_digest([records[i]["user_id"] for i in train_indices]),
                    "eval_users_sha256": json_digest([records[i]["user_id"] for i in eval_indices])}
        source_hashes = {str(path.relative_to(ROOT)): file_sha256(path)
                         for path in sorted((ROOT / "src").rglob("*.py"))}
        control_paths = [Path(__file__).with_name(name) for name in
                         ("__init__.py", "config.py", "mean_pooling.py", "run_mean_pooling.py")]
        control_hashes = {str(path.relative_to(ROOT)): file_sha256(path) for path in control_paths}
        upstream = json.loads((ROOT / "configs/hstu-reference-upstream.json").read_text())
        provenance = {**identity, "data_id": json_digest(identity), "git": git_state(),
                      "source_sha256": source_hashes, "shared_source_id": json_digest(source_hashes),
                      "control_source_sha256": control_hashes,
                      "source_id": json_digest({"shared": source_hashes, "control": control_hashes}),
                      "reference_config": reference_config, "reference_config_sha256": json_digest(reference_config),
                      "reference_upstream_revision": upstream["revision"],
                      "config_sha256": json_digest(config), "seed": config["seed"], "runtime": runtime}
        report["provenance"] = provenance
        report["data"] = {"catalog_items": num_items, "train_users": len(train_indices),
                          "evaluation_users": len(eval_indices), "full_data_users": len(records)}
        training_records = [records[i] for i in train_indices]
        model = PrefixMean(num_items=num_items, **config["model"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                      weight_decay=config["weight_decay"], betas=(0.9, 0.98))
        report["parameter_count"] = sum(p.numel() for p in model.parameters())
        report["reference_parameter_count"] = model.reference_parameter_count
        initial_hash = hashlib.sha256(model.item_embedding.weight.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        report["initial_item_embedding_sha256"] = initial_hash
        provenance["initial_item_embedding_sha256"] = initial_hash
        print(f"Parameters: prefix mean={report['parameter_count']}; HSTU reference={model.reference_parameter_count}", flush=True)
        source = args.resume if args.action == "train" else args.checkpoint
        payload = None
        if source is not None:
            source = source.resolve()
            payload = load_checkpoint(source, model, optimizer, expected_config=config,
                                      expected_data_id=provenance["data_id"])
            report["source_checkpoint"] = {"path": str(source), "sha256": file_sha256(source),
                                           "epoch": payload["epoch"], "provenance": payload["provenance"]}
            if (payload["provenance"]["source_id"] != provenance["source_id"]
                    or payload["provenance"]["reference_config_sha256"] != provenance["reference_config_sha256"]):
                raise ValueError("Checkpoint control/shared source or reference configuration identity differs")
            if args.action == "train":
                if (payload["provenance"]["git"] != provenance["git"]
                        or payload["provenance"]["runtime"] != runtime):
                    raise ValueError("Continuation requires the same code identity and runtime")
                report["history"] = payload["metrics"]["history"]

        length = config["model"]["sequence_length"]
        split = "val" if args.action == "train" else args.split
        evaluation_dataset = SASRecEvalDataset(data_dir / "sequences.jsonl", length, split)
        if args.action == "evaluate":
            report["evaluation"] = evaluate_hstu(model, evaluation_dataset, eval_indices, config, device, training_records)
            print(json.dumps(report["evaluation"], sort_keys=True), flush=True)
            report["status"] = "PASS"
        else:
            train_dataset = SASRecTrainDataset(data_dir / "sequences.jsonl", length)
            probe_loader = make_loader(train_dataset, train_indices[:config["batch_size"]],
                                       config["batch_size"], 0, config["seed"], device)
            probe = move_batch(next(iter(probe_loader)), device)
            report["initial_probe_loss"] = fixed_batch_loss(model, probe, config["temperature"])
            watched = {name: dict(model.named_parameters())[name].detach().clone()
                       for name in ("item_embedding.weight",)}
            first_epoch = payload["epoch"] + 1 if payload else 1
            stop_epoch = args.stop_after_epoch or config["epochs"]
            if first_epoch > stop_epoch:
                raise ValueError("Checkpoint already reached the requested stopping epoch")
            step = payload["step"] if payload else 0
            best_score = payload["metrics"]["best_score"] if payload else -1.0
            best_epoch = payload["metrics"]["best_epoch"] if payload else None
            if payload:
                best_source = source.parent / "best.pt"
                expected_hash = payload["metrics"]["best_checkpoint_sha256"]
                if not best_source.is_file() or file_sha256(best_source) != expected_hash:
                    raise ValueError("Original best.pt is missing or does not match this continuation point")
                shutil.copyfile(best_source, run_dir / "best.pt")
            for epoch in range(first_epoch, stop_epoch + 1):
                epoch_start = time.perf_counter()
                loader = make_loader(train_dataset, train_indices, config["batch_size"], config["num_workers"],
                                     config["seed"] + epoch, device, shuffle=True)
                training = train_epoch(model, optimizer, loader, config, device)
                validation = evaluate_hstu(model, evaluation_dataset, eval_indices, config, device, training_records)
                step += training["steps"]
                score = validation["groups_by_target_train_frequency"]["all"]["metrics"][config["selection_metric"]]
                row = {"epoch": epoch, "training": training, "validation": validation,
                       "probe_loss": fixed_batch_loss(model, probe, config["temperature"]),
                       "seconds": time.perf_counter() - epoch_start}
                report["history"].append(row)
                improved = score > best_score
                if improved:
                    best_score, best_epoch = score, epoch
                metrics = {"history": report["history"], "best_score": best_score, "best_epoch": best_epoch}
                if improved:
                    save_checkpoint(run_dir / "best.pt", model, optimizer, config=config, epoch=epoch,
                                    step=step, metrics=metrics, provenance=provenance)
                metrics["best_checkpoint_sha256"] = file_sha256(run_dir / "best.pt")
                save_checkpoint(run_dir / "last.pt", model, optimizer, config=config, epoch=epoch,
                                step=step, metrics=metrics, provenance=provenance)
                report["selection"] = {"metric": config["selection_metric"], "best_score": best_score,
                                       "best_epoch": best_epoch, "split": "val"}
                report["checkpoints"] = {name: {"path": str(run_dir / name), "sha256": file_sha256(run_dir / name)}
                                         for name in ("best.pt", "last.pt")}
                write_json(run_dir / "results.json", report)
                print(f"Epoch {epoch}/{config['epochs']}: loss={training['loss']:.6f}; "
                      f"probe={row['probe_loss']:.6f}; val {config['selection_metric']}={score:.6f}", flush=True)
            changes = {name: (dict(model.named_parameters())[name].detach() - before).abs().max().item()
                       for name, before in watched.items()}
            if any(change == 0 for change in changes.values()):
                raise AssertionError("Control item embedding did not update")
            model.eval()
            with torch.no_grad():
                before = model(probe["input_ids"], probe["padding_mask"]).clone()
            restored = load_checkpoint(run_dir / "last.pt", model, optimizer, expected_config=config,
                                       expected_data_id=provenance["data_id"])
            with torch.no_grad():
                after = model(probe["input_ids"], probe["padding_mask"])
            torch.testing.assert_close(after, before, atol=0, rtol=0)
            report["checks"] = {"parameter_max_changes": changes, "checkpoint_output_exact": True,
                                "restored_epoch": restored["epoch"],
                                "probe_loss_reduced": report["history"][-1]["probe_loss"] < report["initial_probe_loss"]}
            if config["mode"] == "smoke" and not report["checks"]["probe_loss_reduced"]:
                raise AssertionError("Smoke run did not reduce its fixed training-only probe loss")
            report["status"] = "PASS" if stop_epoch == config["epochs"] else "PARTIAL"
        after_checksums = {name: file_sha256(data_dir / name) for name in checksums}
        if after_checksums != checksums:
            raise RuntimeError("Processed data changed during execution")
        report["checksums_unchanged"] = True
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
        write_json(run_dir / "results.json", report)
        print(f"Saved: {run_dir / 'results.json'}", flush=True)
        print(f"Mean pooling {args.action}: {report['status']}", flush=True)
    if report["status"] == "FAIL":
        raise SystemExit(1)


def main():
    """Require a separate explicit command for held-out checkpoint evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    train = commands.add_parser("train")
    train.add_argument("--resume", type=Path, help="Continue from last.pt at a completed epoch boundary")
    train.add_argument("--stop-after-epoch", type=int)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--split", choices=("val", "test"), default="val")
    for command in (train, evaluation):
        command.add_argument("--config", type=Path, default=ROOT / "configs/mean-pooling-smoke-server.json")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
