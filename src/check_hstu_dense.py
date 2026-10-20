"""Numerical parity checks; importing this module does not import torch."""

from src.check_hstu_reference import validate_config


def validate_comparison_config(config):
    """Validate all test cases without loading model or accelerator libraries."""
    if not config.get("comparison_cases") or not config.get("comparison_seeds"):
        raise ValueError("Comparison cases and seeds must be nonempty")
    if any(type(seed) is not int or seed < 0 for seed in config["comparison_seeds"]):
        raise ValueError("Comparison seeds must be nonnegative integers")
    names = set()
    for case in config["comparison_cases"]:
        name = case["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Comparison case names must be unique nonempty strings")
        names.add(name)
        merged = {**config, **case}
        lengths = merged["history_lengths"]
        if not lengths or any(type(n) is not int or n < 0 for n in lengths) or max(lengths) < 2:
            raise ValueError("Comparison lengths need a nonempty history of at least two items")
        # The original preflight validates only positive histories. Here zero/one
        # rows are also tested, while retaining the same dimension/tolerance rules.
        validate_config({**merged, "history_lengths": [max(2, n) for n in lengths]})


def compare_dense_blocks(torch, device, config, hstu, report):
    """Compare both bias modes with strict shared state and identical probes."""
    from src.models.hstu import DenseHSTUBlock
    report.update({"status": "RUNNING", "cases": [], "dtype": "float32",
                   "timestamps_held_fixed": True, "atol": config["atol"], "rtol": config["rtol"]})

    def compare(actual, expected, label, case_result):
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise AssertionError(f"Nonfinite tensor: {label}")
        difference = actual.detach() - expected.detach()
        case_result["errors"][label] = {
            "max_absolute": difference.abs().max().item(),
            "relative_l2": (difference.norm() / expected.detach().norm().clamp_min(1e-12)).item(),
        }
        torch.testing.assert_close(actual, expected, atol=config["atol"], rtol=config["rtol"], msg=label)

    for case in config["comparison_cases"]:
        settings = {**config, **case}
        n, dim = settings["sequence_length"], settings["embedding_dim"]
        lengths = torch.tensor(settings["history_lengths"], device=device, dtype=torch.int64)
        offsets = torch.ops.fbgemm.asynchronous_complete_cumsum(lengths)
        valid = torch.arange(n, device=device)[None, :] < lengths[:, None]
        allowed = torch.ones(n, n, device=device).tril()
        for seed in config["comparison_seeds"]:
            for bias_mode in ("position_only", "time_and_position"):
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                case_result = {"name": case["name"], "seed": seed, "bias": bias_mode,
                               "status": "RUNNING", "errors": {}}
                report["cases"].append(case_result)
                native_bias = (hstu.RelativePositionalBias(max_seq_len=n) if bias_mode == "position_only" else
                    hstu.RelativeBucketedTimeAndPositionBasedBias(
                        max_seq_len=n, num_buckets=128,
                        bucketization_fn=lambda x: (torch.log(torch.abs(x).clamp(min=1)) / 0.301).long()))
                native = hstu.SequentialTransductionUnitJagged(
                    embedding_dim=dim, linear_hidden_dim=settings["linear_dim"],
                    attention_dim=settings["attention_dim"], num_heads=settings["num_heads"],
                    dropout_ratio=0.0, attn_dropout_ratio=0.0, linear_activation="silu",
                    relative_attention_bias_module=native_bias, normalization="rel_bias",
                    linear_config="uvqk", concat_ua=False,
                ).to(device).eval()
                dense = DenseHSTUBlock(
                    sequence_length=n, embedding_dim=dim, attention_dim=settings["attention_dim"],
                    linear_dim=settings["linear_dim"], num_heads=settings["num_heads"],
                    bias_mode=bias_mode, dropout=0.0,
                ).to(device).eval()
                dense.load_state_dict(native.state_dict(), strict=True)
                native_parameters = dict(native.named_parameters())
                dense_parameters = dict(dense.named_parameters())
                if native_parameters.keys() != dense_parameters.keys():
                    raise AssertionError("Parameter names differ")
                x = torch.randn(len(lengths), n, dim, device=device)
                timestamps = torch.randint(1, 10000, (len(lengths), n), device=device).cumsum(1)
                native_x, dense_x = x.clone().requires_grad_(), x.clone().requires_grad_()

                def native_forward(values):
                    packed = torch.ops.fbgemm.dense_to_jagged(values, [offsets])[0]
                    output, _ = native(x=packed, x_offsets=offsets, all_timestamps=timestamps,
                                       invalid_attn_mask=allowed)
                    return torch.ops.fbgemm.jagged_to_padded_dense(output, [offsets], [n], 0.0)

                native_output = native_forward(native_x)
                dense_output = dense(dense_x, valid, timestamps)
                case_result["shape"] = list(dense_output.shape)
                compare(dense_output, native_output, "output", case_result)
                torch.testing.assert_close(dense_output[~valid], torch.zeros_like(dense_output[~valid]),
                                           rtol=0, atol=0)
                probe = torch.randn_like(x) * valid.unsqueeze(-1)
                native_loss = (native_output * probe).sum() / valid.sum()
                dense_loss = (dense_output * probe).sum() / valid.sum()
                compare(dense_loss, native_loss, "probe_loss", case_result)
                native_gradients = torch.autograd.grad(native_loss, [native_x, *native_parameters.values()])
                dense_gradients = torch.autograd.grad(dense_loss, [dense_x, *(
                    dense_parameters[name] for name in native_parameters)])
                for name, actual, expected in zip(["input", *native_parameters], dense_gradients, native_gradients):
                    compare(actual, expected, f"gradient/{name}", case_result)
                torch.testing.assert_close(dense_gradients[0][~valid],
                                           torch.zeros_like(dense_gradients[0][~valid]), rtol=0, atol=0)
                # Recompute under no_grad: autograd.grad released the original graphs.
                with torch.no_grad():
                    poisoned = x.clone()
                    poisoned[~valid] = 12345.0
                    changed = x.clone()
                    user = next(i for i, length in enumerate(settings["history_lengths"]) if length >= 2)
                    prefix = settings["history_lengths"][user] - 1
                    changed[user, prefix] += torch.randn(dim, device=device) * 10
                    for name, forward, original in (
                        ("native", native_forward, native_output.detach()),
                        ("dense", lambda values: dense(values, valid, timestamps), dense_output.detach()),
                    ):
                        compare(forward(poisoned), original, f"{name}/padding", case_result)
                        future_output = forward(changed)
                        compare(future_output[user, :prefix], original[user, :prefix],
                                f"{name}/causality", case_result)
                        other_users = torch.arange(len(lengths), device=device) != user
                        if other_users.any():
                            compare(future_output[other_users], original[other_users],
                                    f"{name}/batch_isolation", case_result)
                case_result["status"] = "PASS"
                print(f"Dense/reference {case['name']} seed={seed} {bias_mode}: PASS", flush=True)
    report["status"] = "PASS"
