"""Check pinned upstream HSTU operations on tiny server tensors, without data."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]


def command(args):
    """Capture version-control state without invoking a shell."""
    return subprocess.check_output(args, text=True).strip()


def verify_upstream(path, spec):
    """Require the exact clean upstream revision and all audited source files."""
    revision = command(["git", "-C", str(path), "rev-parse", "HEAD"])
    if revision != spec["revision"]:
        raise ValueError(f"Upstream revision mismatch: {revision}")
    if command(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"]):
        raise ValueError("Upstream tracked files are modified")
    for name, expected in spec["files_sha256"].items():
        actual = hashlib.sha256((path / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Upstream source checksum mismatch: {name}")
    return revision


def validate_config(config):
    """Validate tensor sizes and tolerances before importing model dependencies."""
    for key in ("sequence_length", "embedding_dim", "attention_dim", "linear_dim", "num_heads"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    lengths = config["history_lengths"]
    if not lengths or any(type(n) is not int or not 2 <= n <= config["sequence_length"] for n in lengths):
        raise ValueError("History lengths must be between 2 and sequence_length")
    for key in ("atol", "rtol"):
        if not isinstance(config[key], (int, float)) or not 0 <= config[key] < 1:
            raise ValueError(f"Invalid {key}")


def check_operations(torch, device, config):
    """Check empty rows, round-trip values and exact masking of input gradients."""
    n, dim = config["sequence_length"], config["embedding_dim"]
    lengths = torch.tensor([*config["history_lengths"], 0], device=device, dtype=torch.int64)
    offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    expected_offsets = torch.cat((lengths.new_zeros(1), lengths.cumsum(0)))
    torch.testing.assert_close(offsets, expected_offsets, rtol=0, atol=0)
    x = torch.randn(len(lengths), n, dim, device=device, requires_grad=True)
    valid = torch.arange(n, device=device)[None, :] < lengths[:, None]
    jagged = torch.ops.fbgemm.dense_to_jagged(x, [offsets])[0]
    padded = torch.ops.fbgemm.jagged_to_padded_dense(jagged, [offsets], [n], 0.0)
    torch.testing.assert_close(padded, x.detach() * valid.unsqueeze(-1), rtol=0, atol=0)
    padded.sum().backward()
    torch.testing.assert_close(x.grad, valid.unsqueeze(-1).expand_as(x).float(), rtol=0, atol=0)
    return {"status": "PASS", "dense_shape": list(x.shape), "jagged_shape": list(jagged.shape),
            "offsets": offsets.tolist(), "empty_row": True, "round_trip_and_gradients": True}


def check_block(torch, device, config, upstream_module, bias_mode):
    """Test an official block, preserving its fixed-length attention scaling."""
    from importlib import import_module
    hstu = import_module(upstream_module)
    n, dim = config["sequence_length"], config["embedding_dim"]
    if bias_mode == "position_only":
        bias = hstu.RelativePositionalBias(max_seq_len=n)
    else:
        bias = hstu.RelativeBucketedTimeAndPositionBasedBias(
            max_seq_len=n, num_buckets=128,
            bucketization_fn=lambda x: (torch.log(torch.abs(x).clamp(min=1)) / 0.301).long(),
        )
    block = hstu.SequentialTransductionUnitJagged(
        embedding_dim=dim, linear_hidden_dim=config["linear_dim"],
        attention_dim=config["attention_dim"], num_heads=config["num_heads"],
        dropout_ratio=0.0, attn_dropout_ratio=0.0, linear_activation="silu",
        relative_attention_bias_module=bias, normalization="rel_bias",
        linear_config="uvqk", concat_ua=False,
    ).to(device)
    block.eval()
    lengths = torch.tensor(config["history_lengths"], device=device, dtype=torch.int64)
    offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
    valid = torch.arange(n, device=device)[None, :] < lengths[:, None]
    allowed = torch.ones(n, n, device=device).tril()
    timestamps = torch.arange(n, device=device, dtype=torch.int64)[None, :].expand(len(lengths), -1) * 60
    x = torch.randn(len(lengths), n, dim, device=device, requires_grad=True)

    def forward(dense):
        values = torch.ops.fbgemm.dense_to_jagged(dense, [offsets])[0]
        output, _ = block(x=values, x_offsets=offsets, all_timestamps=timestamps,
                          invalid_attn_mask=allowed)
        return torch.ops.fbgemm.jagged_to_padded_dense(output, [offsets], [n], 0.0)

    y = forward(x)
    if y.shape != x.shape or not torch.isfinite(y).all():
        raise AssertionError("Official block output has invalid shape or nonfinite values")
    loss = y[valid].square().mean()
    loss.backward()
    if not torch.isfinite(loss) or x.grad is None or not torch.isfinite(x.grad).all():
        raise AssertionError("Input gradient or loss is nonfinite")
    for name, parameter in block.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"Missing or nonfinite parameter gradient: {name}")
    if not any(parameter.grad.abs().sum().item() > 0 for parameter in block.parameters()):
        raise AssertionError("All parameter gradients are zero")
    torch.testing.assert_close(x.grad[~valid], torch.zeros_like(x.grad[~valid]), rtol=0, atol=0)
    with torch.no_grad():
        poisoned = x.detach().clone()
        poisoned[~valid] = 12345.0
        torch.testing.assert_close(forward(poisoned), y.detach(), atol=config["atol"], rtol=config["rtol"])
        changed = x.detach().clone()
        prefix = int(lengths[0]) - 1
        changed[0, prefix] += torch.randn(dim, device=device) * 10
        changed_output = forward(changed)
        torch.testing.assert_close(changed_output[0, :prefix], y.detach()[0, :prefix],
                                   atol=config["atol"], rtol=config["rtol"])
        torch.testing.assert_close(changed_output[1:], y.detach()[1:],
                                   atol=config["atol"], rtol=config["rtol"])
    return {"status": "PASS", "bias": bias_mode, "shape": list(y.shape),
            "loss": loss.item(), "parameter_count": sum(p.numel() for p in block.parameters()),
            "finite_gradients": True, "padding_invariance": True,
            "future_embedding_invariance": True, "timestamps_held_fixed": True,
            "precision": "float32", "dropout": 0.0}


def main():
    """Write success or failure evidence, package versions and memory usage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/hstu-reference-check-server.json")
    parser.add_argument("--upstream-dir", type=Path,
                        default=Path.home() / "upstream/generative-recommenders-reference")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    spec = json.loads((ROOT / "configs/hstu-reference-upstream.json").read_text())
    upstream = args.upstream_dir.expanduser().resolve()
    now = datetime.now(timezone.utc)
    output_dir = Path(config["output_dir"]) / now.strftime("%Y%m%dT%H%M%S%fZ")
    output_dir.mkdir(parents=True, exist_ok=False)
    result = {"status": "RUNNING", "created_at_utc": now.isoformat(), "config": config,
              "scope": "tiny official FP32 block and FBGEMM checks; no dataset or training",
              "upstream": {"path": str(upstream), "expected_revision": spec["revision"]},
              "runtime": {"python": sys.version, "executable": sys.executable,
                          "platform": platform.platform()}, "checks": {}}
    torch = None
    device = None
    try:
        result["git"] = {"commit": command(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
                         "dirty": bool(command(["git", "-C", str(ROOT), "status", "--porcelain"]))}
        result["upstream"]["revision"] = verify_upstream(upstream, spec)
        packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()
                    if d.metadata["Name"]}
        result["runtime"]["packages"] = packages
        for name, expected in spec["expected_packages"].items():
            actual = importlib.metadata.version(name)
            if actual != expected:
                raise RuntimeError(f"Expected {name}=={expected}, found {actual}")
        result["pip_check"] = command([sys.executable, "-m", "pip", "check"])
        import numpy as np
        import torch as torch_module
        torch = torch_module
        result["runtime"].update({"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                                 "torch_file": str(Path(torch.__file__).resolve())})
        import fbgemm_gpu  # noqa: F401
        device = torch.device(config["device"])
        random.seed(config["seed"])
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Configured CUDA device is not available")
            torch.cuda.set_device(device)
            torch.cuda.manual_seed_all(config["seed"])
            torch.cuda.reset_peak_memory_stats(device)
            result["runtime"]["gpu"] = torch.cuda.get_device_name(device)
            result["runtime"]["gpu_total_memory_bytes"] = torch.cuda.get_device_properties(device).total_memory
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        result["runtime"].update({"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                                 "device": str(device), "tf32": False})
        sys.path.insert(0, str(upstream))
        module_name = "generative_recommenders.research.modeling.sequential.hstu"
        module = importlib.import_module(module_name)
        if Path(module.__file__).resolve() != upstream / spec["hstu_source"]:
            raise RuntimeError("Imported HSTU from a different checkout")
        result["checks"]["fbgemm"] = check_operations(torch, device, config)
        print("FBGEMM round-trip and gradients: PASS", flush=True)
        for mode in ("position_only", "time_and_position"):
            result["checks"][mode] = check_block(torch, device, config, module_name, mode)
            print(f"Official HSTU block ({mode}): PASS", flush=True)
        result["status"] = "PASS"
    except Exception as error:
        result["status"] = "FAIL"
        result["error"] = {"type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc()}
    finally:
        if torch is not None and device is not None and device.type == "cuda" and torch.cuda.is_initialized():
            try:
                torch.cuda.synchronize(device)
                result["memory_bytes"] = {
                    "allocated": torch.cuda.memory_allocated(device),
                    "reserved": torch.cuda.memory_reserved(device),
                    "peak_allocated": torch.cuda.max_memory_allocated(device),
                    "peak_reserved": torch.cuda.max_memory_reserved(device),
                }
            except Exception as error:
                result["memory_error"] = str(error)
                result["status"] = "FAIL"
        path = output_dir / "results.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"Saved: {path}", flush=True)
    if result["status"] != "PASS":
        print(result.get("error", result.get("memory_error")), file=sys.stderr)
        raise SystemExit(1)
    print("HSTU reference preflight: PASS", flush=True)


if __name__ == "__main__":
    main()
