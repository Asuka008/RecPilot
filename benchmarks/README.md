# Performance baseline

This module measures the existing recipe without modifying checkpoint-bound source
files, training configurations, serving behavior, or frozen quality results.
Both core entry points returned server PASS on 2026-11-07; their JSON reports
have been inspected. All eight HTTP load reports and both operator tables were
subsequently reviewed; see [results and limits](RESULTS-2026-11-07.md). Full raw
trace timelines remain uninspected.
Local checks cover standard-library statistics, instrumentation restoration and
synthetic loopback HTTP load only.

## Server procedure

Run from the project root (for the existing server: `cd "$HOME/AgenticRec"`).
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

After core measurement finishes, start the existing service in terminal A from
the project root:

```bash
python -m serving.run serve --config configs/hybrid-serving-server.json
```

In terminal B, also enter the project root, activate the environment, and set
the exact request file printed by the **serving** benchmark:

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

## Verified core measurements: 2026-11-07

Returned JSON reports were inspected for serving run `20261107T210935152729Z`
and training run `20261107T211029879526Z`, at clean server commit `254444e`.
All five benchmark source hashes match the local implementation. Both reports
identify the same frozen model and request corpus. At the initial core review, raw traces and operator tables had not been inspected.
The later [HTTP/operator review](RESULTS-2026-11-07.md) supersedes this pending
operator status; raw trace timelines still have not been inspected. The profiler cycle-accumulation warning
alone does not invalidate this single continuous capture or independent plain pass.

Actual environment: A100-SXM4-80GB MIG 2g.20gb, 19.50 GiB visible, PyTorch
2.10.0+cu128, CUDA runtime 12.8, FP32, TF32 off, 24 CPU threads. These are not
H200 results. Each plain pass measured 100 steps after 20 warmup steps.

| Plain baseline | Result |
|---|---:|
| Serving mean / p50 / p95 / p99 | 3.749 / 3.646 / 3.909 / 4.310 ms |
| Serial in-process serving throughput | 266.21 requests/s |
| Serving peak allocated / reserved | 47.47 / 54.00 MiB |
| Training window, 100 steps | 1.32224 s |
| Training effective prediction positions | 13,438 |
| Training effective positions/s | 10,163.04 |
| Training peak allocated / reserved | 151.83 / 194.00 MiB |
| Bounded validation, 6,400 users | 2.83962 s |

Serving excludes HTTP and queueing. Training uses a disposable copy with a fresh
optimizer, not an original epoch. Training memory includes the resident admitted
model; GPU allocator peaks exclude CPU CF and external allocations. Lifetime peak
process RSS was 1.418 GiB (serving) and 2.050 GiB (training), including startup and
diagnostic passes, not an isolated steady-state allocation.

Synchronized diagnostic exclusive means (shares of summed exclusive stage time):

| Stage | Serving ms / share | Training ms / share |
|---|---:|---:|
| User encoder | 2.634 / 62.14% | 3.398 / 23.51% |
| Backward, all model/loss components | — | 5.139 / 35.56% |
| Checks and step overhead | — | 3.043 / 21.06% |
| Data wait | — | 1.093 / 7.57% |
| Candidate scoring forward | 0.299 / 7.05% | 0.393 / 2.72% |
| Input preparation and transfer | 0.356 / 8.40% | — |
| ItemCF | 0.272 / 6.43% | — |
| Filtering and Top-K | 0.293 / 6.91% | — |

The diagnostic passes took 14.77% (serving) and 9.60% (training) longer than
plain passes. Their percentages are not additive estimates of plain runtime or
promised optimization savings. Encoder includes input compaction, embeddings,
normalization and attention; backward includes the scoring/loss path as well.
Checks overhead includes finite checks, clipping, scalar extraction and other
step work. CPU/GPU synchronization is a hypothesis to inspect in the trace,
not yet an operator-level conclusion. Neither an attention-only bottleneck nor a need for FSDP
has been established. The initial core review did not establish concurrent
capacity; the subsequent HTTP load evidence is recorded in the linked report.
For a future reproduction, proceed with HTTP load
using `runs/performance/20261107T210935152729Z/requests.json`; do not rerun these
core measurements just to suppress the profiler warning.


## Returning HTTP reports

From the project root, package completed reports without rerunning measurements:

```bash
find runs/http-load -mindepth 2 -maxdepth 2 -type f -name results.json -print0 |
  tar --null -czf /tmp/http-load-results.tar.gz --files-from=-
tar -tzf /tmp/http-load-results.tar.gz
mkdir -p runs/exports
cp /tmp/http-load-results.tar.gz runs/exports/http-load-results.tar.gz
```

Download the file from `AgenticRec/runs/exports/` in Jupyter's file browser.
Exports remain ignored by Git. Inspect the archive list to ensure it contains
only the intended runs. Do not commit raw reports or archives.
