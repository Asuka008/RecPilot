"""Audit aligned titles, inspect the server, or build a frozen feature cache."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from title_features.data import align_titles, file_sha256, input_hashes, json_digest, validate_config


def write_json(path, value):
    """Publish complete JSON reports atomically."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def command_output(arguments):
    """Record short environment diagnostics without changing the environment."""
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=20)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def inspect_runtime(config):
    """Report dependencies and actual allocation; never install packages or download models."""
    packages = dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions()
                           if d.metadata["Name"]))
    versions = {}
    for name in ("pip", "numpy", "torch", "transformers", "tokenizers", "huggingface-hub", "safetensors"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    result = {"python": platform.python_version(), "executable": sys.executable,
              "platform": platform.platform(), "versions": versions, "packages": packages,
              "executables": {name: shutil.which(name) for name in ("conda", "pip", "nvcc", "sbatch")},
              "conda": command_output([os.environ.get("CONDA_EXE", "conda"), "--version"]),
              "pip_check": command_output([sys.executable, "-m", "pip", "check"]),
              "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
              "device_requested": config["device"]}
    ready = versions["transformers"] == config["transformers_version"]
    try:
        import torch

        result.update({"torch": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                       "cuda_available": torch.cuda.is_available()})
        device = torch.device(config["device"])
        if device.type == "cuda":
            properties = torch.cuda.get_device_properties(device)
            result.update({"gpu": properties.name, "gpu_total_memory_bytes": properties.total_memory})
    except Exception as error:
        result["torch_import_or_device_error"] = str(error)
        ready = False
    if ready:
        try:
            from transformers.models.bert.modeling_bert import BertModel

            result["bert_import"] = BertModel.__name__
        except Exception as error:
            result["bert_import_error"] = str(error)
            ready = False
    result["dependencies_ready"] = ready and result["pip_check"].get("returncode") == 0
    return result


def finish_report(run_dir, report, start):
    """Publish a terminal report and a concise server-facing status."""
    report["elapsed_seconds"] = time.perf_counter() - start
    write_json(run_dir / "results.json", report)
    print(f"Saved: {run_dir / 'results.json'}", flush=True)
    print(f"Title features {report['action']}: {report['status']}", flush=True)
    if report["status"] == "FAIL":
        raise SystemExit(1)


def run(args):
    """Preserve failure reports and keep small smoke artifacts separate from full caches."""
    config = json.loads(args.config.read_text())
    validate_config(config)
    if args.action in {"smoke", "build"}:
        workspace = os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if workspace not in {":4096:8", ":16:8"}:
            raise ValueError("Deterministic CUDA requires a supported CUBLAS_WORKSPACE_CONFIG")
    output_root = (ROOT / config["output_dir"]).resolve()
    for immutable in (ROOT / "data/raw", ROOT / config["data_dir"]):
        if output_root == immutable.resolve() or immutable.resolve() in output_root.parents:
            raise ValueError("Feature reports must not be written into raw or processed data")
    run_dir = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    report = {"status": "RUNNING", "action": args.action, "config": config,
              "config_sha256": json_digest(config),
              "protocol": {"name": "static-catalog-frozen-title-features-v1", "padding_row": 0,
                           "uses_interaction_labels_for_encoding": False,
                           "catalog_assumed_known": True,
                           "historical_title_availability": "not established by this dataset snapshot",
                           "recommendation_gain": "not measured; feature preparation only"}}
    start = time.perf_counter()
    write_json(run_dir / "results.json", report)
    try:
        if args.action == "preflight":
            report["runtime"] = inspect_runtime(config)
            inputs = [ROOT / config["titles_csv"]] + [ROOT / config["data_dir"] / name
                      for name in ("item_mapping.json", "metadata.json", "sequences.jsonl")]
            report["input_files_present"] = {str(path): path.is_file() for path in inputs}
            compact = {k: v for k, v in report["runtime"].items() if k != "packages"}
            print(json.dumps({"runtime": compact, "input_files_present": report["input_files_present"]}, indent=2), flush=True)
            report["status"] = ("PASS" if report["runtime"]["dependencies_ready"]
                                and all(report["input_files_present"].values()) else "NEEDS_SETUP")
            finish_report(run_dir, report, start)
            return
        report["input_sha256"] = input_hashes(ROOT, config)
        titles, audit = align_titles(ROOT / config["titles_csv"], ROOT / config["data_dir"] / "item_mapping.json")
        report["audit"] = audit
        sources = sorted((ROOT / "title_features").glob("*.py")) + [Path(__file__).resolve()]
        report["source_sha256"] = {str(p.relative_to(ROOT)): file_sha256(p) for p in sources}
        report["source_id"] = json_digest(report["source_sha256"])
        report["git"] = {"commit": command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
                         "status": command_output(["git", "-C", str(ROOT), "status", "--porcelain"])}
        if args.action == "audit":
            print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
        else:
            report["runtime"] = inspect_runtime(config)
            if not report["runtime"]["dependencies_ready"]:
                raise RuntimeError("Server dependencies are not ready; review preflight before installing or encoding")
            import numpy as np
            import torch
            from title_features.cache import validate_array
            from title_features.encoder import encode_titles

            if config["device"].startswith("cuda"):
                torch.cuda.reset_peak_memory_stats(torch.device(config["device"]))
            selected = titles if args.action == "build" else titles[:min(config["smoke_items"], len(titles) - 1) + 1]
            report["encoded_items"] = len(selected) - 1
            vectors, report["encoder"] = encode_titles(config, selected)
            validate_array(vectors, len(selected) - 1, config["feature_dim"])
            temporary = run_dir / "features.tmp.npy"
            np.save(temporary, vectors, allow_pickle=False)
            reread = np.load(temporary, allow_pickle=False)
            if not np.array_equal(vectors, reread):
                raise AssertionError("Saved features differ from inference output")
            temporary.replace(run_dir / "features.npy")
            report["features_sha256"] = file_sha256(run_dir / "features.npy")
            report["feature_shape"] = list(vectors.shape)
            report["checks"] = {"saved_output_exact": True, "unit_rows": True, "padding_zero": True}
            if config["device"].startswith("cuda"):
                device = torch.device(config["device"])
                torch.cuda.synchronize(device)
                report["memory_bytes"] = {"peak_allocated": torch.cuda.max_memory_allocated(device),
                                          "peak_reserved": torch.cuda.max_memory_reserved(device)}
        if report["input_sha256"] != input_hashes(ROOT, config):
            raise RuntimeError("Title or processed-data inputs changed during execution")
        if any(file_sha256(ROOT / name) != sha for name, sha in report["source_sha256"].items()):
            raise RuntimeError("Feature preparation code changed during execution")
        report["checksums_unchanged"] = True
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        print(report["error"]["traceback"], flush=True)
    finish_report(run_dir, report, start)


def main():
    """Keep audit, environment inspection, smoke and full build explicit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("audit", "preflight", "smoke", "build"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs/title-features-minilm-server.json")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
