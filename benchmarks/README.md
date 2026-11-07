# Performance baseline

This module measures the existing recipe without modifying checkpoint-bound source
files, training configurations, serving behavior, or frozen quality results.
GPU execution is pending server verification. Local checks cover standard-library
statistics, instrumentation restoration and synthetic loopback HTTP load only.

## Server procedure

Use the verified project environment and inspect the actual allocation first:

```bash
git pull --ff-only
conda activate "$HOME/conda-envs/agenticrec-hstu-reference"
nvidia-smi
python -c 'import torch; print(torch.__version__, torch.version.cuda); print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)'
python -m benchmarks.run serving --config configs/hybrid-serving-server.json
python -m benchmarks.run training --config configs/hybrid-serving-server.json
```

Both commands require the original admitted experiment artifacts, exactly as the
existing service does. Expected completion: `Performance benchmark: PASS` and a
new directory under `runs/performance/`. Failure also writes `results.json`.
No model is saved and no test quality evaluation is performed. All measurements
use existing FP32/TF32-off settings, actual device and CPU thread counts recorded
in the runtime manifest. Never run core profiling and HTTP load simultaneously.

Each command writes:

- `results.json`: provenance, runtime/config/data/checkpoint identities, workload,
  raw timing samples, percentiles, units/second and allocator peaks.
- `trace.json`: CPU/CUDA operator trace, readable with a local trace viewer.
- `operators.txt`: CPU-self-time sorted operator summary; use the trace for CUDA.
- `stages.svg`: synchronized exclusive stage wall time, **diagnostic only**.
- `requests.json`: mapped-ID training histories, without user IDs or held-out
  labels; remains an ignored runtime artifact. Both reports identify its hash.

Three separate passes use identical seeds and warmup: plain benchmark, synchronized
stage attribution, and profiler capture. Do not quote stage/trace throughput as
baseline performance. Stage synchronization perturbs overlap; exclusive stage
sums are not an estimate of the uninstrumented critical path. Plain serving
measures synchronized batch-one core latency. Plain training measures a continuous
window; per-step wall samples are not individually CUDA-synchronized latencies.

Training starts each pass from a disposable copy of admitted best weights with a
fresh AdamW optimizer, using the original training batch size/workers and 100
measured steps after 20 warmup steps. It is a workload benchmark, not a replay of
an original epoch or a new convergence experiment. It reports **effective
supervised positions/second**, data wait, transfer, encoder, candidate scoring,
loss preparation, backward, optimizer and validation timing. Validation is a
bounded first-user subset, uses only validation labels and discards quality
metrics. Loading/startup is recorded separately; checkpoint I/O is not measured.
The resident serving model contributes to process allocator peaks during training;
these peaks are not standalone training memory. CPU CF memory is not GPU memory.

Serving ranges cover request validation, input construction/transfer (combined),
encoder, candidate scoring, ItemCF, fusion, filtering/Top-K and residual overhead.
Instrumented responses must equal plain responses on shared requests. Existing
admission/source hashes are rechecked at completion. This instrumentation does not
change the existing HTTP service or its serial processing behavior.

## End-to-end HTTP load

After core measurement finishes, start the existing service in terminal A:

```bash
python -m serving.run serve --config configs/hybrid-serving-server.json
```

In terminal B, set the exact request file printed by the **serving** benchmark:

```bash
REQUESTS=runs/performance/REPLACE_WITH_SERVING_RUN/requests.json
for concurrency in 1 4 16 32; do
  python -m benchmarks.http_load --requests "$REQUESTS" --mode closed --concurrency "$concurrency" --duration 60
done
for rate in 50 100 200 400; do
  python -m benchmarks.http_load --requests "$REQUESTS" --mode open --concurrency 32 --rate "$rate" --duration 60
done
```

Closed mode waits for each response before the worker's next request. Open mode
schedules arrivals independently of responses, caps concurrent work, and counts
arrivals that cannot be admitted as **client drops**. No unbounded executor queue
is allowed. The rate grid is exploratory, not a promised service capacity.
Repeat interesting operating points at least three times; increase duration if
tail estimates are unstable. Record co-tenant GPU load and avoid cross-run traffic.

Expected: `HTTP load: COMPLETE`. This means measurement finished, **not** that a
latency SLO passed. Inspect successful QPS, errors, drops, dispatch lag and both
HTTP and scheduled-to-completion percentiles together. Failed attempts retain
status/error and timing; timeouts do not become successful latency samples.
The client verifies model identity, candidate counts, history exclusion and scores.
A new TCP connection is used per request, matching the current HTTP/1.0 server.
Results include client scheduling and connection/server waiting; server queue time
cannot be isolated without later server instrumentation. No GPU utilization or
server memory is inferred from HTTP client timings. Match the request hash and
health model ID with the core report for hardware and model provenance.

Initial completion criterion: successful server core runs plus an HTTP load curve
showing throughput, tail latency and failure/drop rates. Only then choose the next
optimization. Do not infer concurrent capacity from reciprocal batch-one latency.
