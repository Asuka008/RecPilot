"""Server-only small-sample learning and exact checkpoint continuation gate."""

import argparse
from datetime import datetime, timezone
import importlib.metadata as package_metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import traceback

from src.toy_training_config import validate_toy_config


ROOT = Path(__file__).resolve().parents[1]


def main():
    """Record a complete synthetic experiment, including failed checks."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hstu-toy-fit-server.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_toy_config(config)
    # Set before importing torch or initializing CUDA; enforce rather than assume determinism.
    workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if workspace not in {":4096:8", ":16:8"}:
        raise ValueError("Exact replay requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    now = datetime.now(timezone.utc)
    run_dir = Path(config["output_dir"]) / now.strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "created_at_utc": now.isoformat(), "config": config,
              "scope": "synthetic FP32 learning and checkpoint continuation; not recommendation quality",
              "runtime": {"python": sys.version, "platform": platform.platform(),
                          "executable": sys.executable, "cublas_workspace_config": workspace},
              "checks": {}, "loss_history": []}
    torch = None
    device = None
    try:
        import numpy as np
        import torch as torch_module
        from src.checkpoints import load_checkpoint, save_checkpoint
        from src.models.id_hstu import IDHSTU
        from src.toy_training import evaluate_loss, json_digest, make_toy_batch, train_step
        torch = torch_module
        device = torch.device(config["device"])
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Configured CUDA device is unavailable")
            torch.cuda.set_device(device)
            torch.cuda.manual_seed_all(config["seed"])
            torch.cuda.reset_peak_memory_stats(device)
            report["runtime"]["gpu"] = torch.cuda.get_device_name(device)
            report["runtime"]["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(device).total_memory
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        report["runtime"].update({"torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                                   "numpy": np.__version__, "device": str(device), "dtype": "float32",
                                   "tf32": False, "deterministic_algorithms": True,
                                   "packages": {d.metadata["Name"]: d.version for d in package_metadata.distributions()
                                                if d.metadata["Name"]}})
        commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True).strip())
        batch, identity = make_toy_batch(config, device)
        provenance = {**identity, "git_commit": commit, "git_dirty": dirty,
                      "config_sha256": json_digest(config), "seed": config["seed"],
                      "python": platform.python_version(), "torch": str(torch.__version__),
                      "numpy": np.__version__, "packages": report["runtime"]["packages"]}
        report["provenance"] = provenance
        report["batch"] = {"users": len(batch["input_ids"]),
                            "supervised_positions": batch["valid_mask"].sum().item(),
                            "input_shape": list(batch["input_ids"].shape),
                            "fixed_negatives": True, "input_layout": "left-padded Dataset contract"}

        def create_pair():
            model = IDHSTU(**config["model"]).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                          weight_decay=config["weight_decay"], betas=(0.9, 0.98))
            return model, optimizer

        model, optimizer = create_pair()
        initial = evaluate_loss(model, batch, config["temperature"])
        report["initial_loss"] = initial
        report["loss_history"].append({"step": 0, "loss": initial})
        watched = {name: dict(model.named_parameters())[name].detach().clone()
                   for name in ("item_embedding.weight", "blocks.0._uvqk")}
        print(f"Synthetic users={len(batch['input_ids'])}; initial loss={initial:.6f}", flush=True)
        for step in range(1, config["steps"] + 1):
            train_step(model, optimizer, batch, config["temperature"])
            if step % config["log_every"] == 0 or step == config["steps"]:
                current = evaluate_loss(model, batch, config["temperature"])
                report["loss_history"].append({"step": step, "loss": current})
                print(f"Step {step}: fixed-task loss={current:.6f}", flush=True)
        final = evaluate_loss(model, batch, config["temperature"])
        reduction = (initial - final) / max(initial, 1e-12)
        updates = {name: (dict(model.named_parameters())[name].detach() - value).abs().max().item()
                   for name, value in watched.items()}
        report["checks"]["fit"] = {"initial_loss": initial, "final_loss": final,
                                    "relative_reduction": reduction, "max_parameter_changes": updates}
        if reduction < config["minimum_loss_reduction"] or any(value == 0 for value in updates.values()):
            raise AssertionError("Synthetic fitting did not meet the fixed loss/update gate")
        report["checks"]["fit"]["status"] = "PASS"
        print(f"Small-sample fit: PASS (loss reduction {reduction:.1%})", flush=True)

        model.eval()
        with torch.no_grad():
            before = model(batch["input_ids"], batch["valid_mask"]).detach().clone()
        checkpoint = run_dir / "checkpoint.pt"
        save_checkpoint(checkpoint, model, optimizer, config=config, epoch=config["steps"],
                        step=config["steps"], metrics={"loss": final}, provenance=provenance)

        def rng_probe():
            return {"python": random.random(), "numpy": float(np.random.random()),
                    "torch_cpu": torch.rand(4).tolist(),
                    "torch_device": torch.rand(4, device=device).cpu().tolist()}

        probe_before = rng_probe()
        uninterrupted_loss = train_step(model, optimizer, batch, config["temperature"])
        restored, restored_optimizer = create_pair()
        payload = load_checkpoint(checkpoint, restored, restored_optimizer,
                                  expected_config=config, expected_data_id=identity["data_id"])
        restored.eval()
        with torch.no_grad():
            after = restored(batch["input_ids"], batch["valid_mask"])
        torch.testing.assert_close(after, before, atol=config["resume_atol"], rtol=config["resume_rtol"])
        if probe_before != rng_probe():
            raise AssertionError("Restored Python/NumPy/torch RNG probes differ")
        resumed_loss = train_step(restored, restored_optimizer, batch, config["temperature"])

        def compare_states(actual, expected, name):
            if isinstance(expected, torch.Tensor):
                torch.testing.assert_close(actual, expected, atol=config["resume_atol"],
                                           rtol=config["resume_rtol"], msg=name)
            elif isinstance(expected, dict):
                if actual.keys() != expected.keys():
                    raise AssertionError(f"State keys differ: {name}")
                for key in expected:
                    compare_states(actual[key], expected[key], f"{name}/{key}")
            elif isinstance(expected, (list, tuple)):
                if len(actual) != len(expected):
                    raise AssertionError(f"State length differs: {name}")
                for index, (left, right) in enumerate(zip(actual, expected)):
                    compare_states(left, right, f"{name}/{index}")
            elif actual != expected:
                raise AssertionError(f"State value differs: {name}")

        compare_states(restored.state_dict(), model.state_dict(), "model")
        compare_states(restored_optimizer.state_dict(), optimizer.state_dict(), "optimizer")
        if abs(resumed_loss - uninterrupted_loss) > config["resume_atol"] + config["resume_rtol"] * abs(uninterrupted_loss):
            raise AssertionError("Resumed next-step loss differs")
        report["checks"]["checkpoint"] = {
            "status": "PASS", "path": str(checkpoint), "saved_step": payload["step"],
            "output_max_absolute_error": (after - before).abs().max().item(),
            "uninterrupted_next_loss": uninterrupted_loss, "resumed_next_loss": resumed_loss,
            "rng_probes_match": True, "model_and_optimizer_next_states_match": True,
        }
        report["status"] = "PASS"
        print("Checkpoint output, RNG and next-step continuation: PASS", flush=True)
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
    finally:
        if torch is not None and device is not None and device.type == "cuda" and torch.cuda.is_initialized():
            try:
                torch.cuda.synchronize(device)
                report["memory_bytes"] = {"peak_allocated": torch.cuda.max_memory_allocated(device),
                                          "peak_reserved": torch.cuda.max_memory_reserved(device)}
            except Exception as error:
                report["status"] = "FAIL"
                report["memory_error"] = str(error)
        result_path = run_dir / "results.json"
        result_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"Saved: {result_path}", flush=True)
    if report["status"] != "PASS":
        print(report.get("error", report.get("memory_error")), file=sys.stderr)
        raise SystemExit(1)
    print("HSTU toy fit and checkpoint: PASS", flush=True)


if __name__ == "__main__":
    main()
