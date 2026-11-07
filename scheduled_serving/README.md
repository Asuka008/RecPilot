# Bounded serving experiment

This experiment adds admission control and telemetry around the frozen HSTU +
ItemCF recommendation core. The existing `serving/`, model, training and benchmark
client implementations are unchanged. The baseline remains available on port
8765; the new service defaults to 8766. Run only one load experiment at a time.
There is **one inference worker and no microbatching**. This isolates scheduling
behavior from changes in numerical precision, ranking or batch size.

## Behavior and bounds

`configs/scheduled-serving-server.json` sets these initial engineering budgets,
not validation-selected performance claims:

| Setting | Default | Meaning |
|---|---:|---|
| Waiting queue capacity | 16 | FIFO jobs, excluding the active job |
| Active inference | 1 | The worker alone invokes the model/CF core |
| HTTP handler limit | 64 | Limits accepted handler threads and their watchdogs |
| TCP listen backlog request | 128 | OS may apply its own backlog limits |
| Server accepted-connection budget | 200 ms | Header/body reading, queue, core, response |
| Response reserve | 10 ms | Stop waiting for inference before the connection deadline |
| Shutdown grace | 5 s | Bound the worker drain wait, then handler drain wait |

The server returns 503 + Retry-After when the job queue is full or it is stopping,
504 for expired application work, 400 for malformed requests and 500 for a core
exception. A handler-limit rejection attempts a small 503 directly, with bounded
write time; transport failures are counted and not described as delivered 503s.
Invalid requests are validated before inference admission. Request bodies are
limited to 64 KiB, with existing history/K/ID validation preserved.

A watchdog closes a connection at its absolute accept-time deadline even if a
client trickles headers/body and repeatedly resets a per-operation socket timer.
This is a best-effort operating-system timer, not a hard real-time guarantee.
**Time before accept in TCP/OS queues, client scheduling and network transit are
outside this server budget.** The original client still uses a five-second socket
operation timeout; this is not an end-to-end client deadline. A late or already
broken connection may receive no HTTP error response at all.

Expired waiting jobs are removed without running the model. Active CUDA work
cannot safely be interrupted: the worker remains occupied until it completes,
its late result is discarded, and only then can the next job start. Disconnects
are counted when observed during reading/writing; a peer closing while a job is
queued is not reliably detected immediately. Pending jobs still expire within
their budget. We do not interpret a half-closed read stream as definitive proof
that the peer stopped receiving responses.

The service uses bounded standard-library handler threads plus a watchdog timer
per admitted connection. Thread creation, validation and synchronization have
costs; no speedup is claimed before server measurements. This is a loopback-only
engineering experiment, not a production web server or a public endpoint.

## Identity and observability

- `GET /health`: readiness, actual runtime, model ID, service ID, random instance
  ID and scheduling config. A live worker may still be busy: readiness is not a
  guarantee of spare capacity. Warmup runs on the inference worker before ready.
- `GET /metrics`: cumulative bounded counters/histograms, queue depth, active
  inference and live accepted connections. It includes the metrics connection.
- `POST /recommend`: identical JSON body to the old service. Response headers
  add X-Request-ID, X-Service-ID and, on success, Server-Timing for queue/core.
- `queue_started_ms`: enqueue to start, for jobs that began execution.
- `queue_cancelled_ms`: residence time of jobs cancelled before execution.
- `core_ms`: full recommendation-core wall time, including data transfer, CF and
  ranking. It is not GPU-only time and includes results later discarded.
- `accepted_connection_ms`, `response_attempt_ms`: server-side transport timing,
  including control requests. Fixed buckets are disjoint; the final null upper
  bound means overflow. Histograms are bounded, not stored per-request samples.
- `recommend_attempt_<status>` vs `recommend_written_<status>`: intended response
  vs successful server socket write. Neither proves the client consumed it.
  General `http_*` counters include control endpoints; connection-limit rejection
  has separate counters because a request may not yet have been parsed.
- `jobs_expired`, `late_results_discarded`, `response_write_failed`,
  `connection_deadline_closed`, `inference_errors` and admission counters explain
  failure paths. They are overlapping events, not disjoint request outcomes.

No user histories are written to service logs. Metrics and request IDs do not
include raw user identities. Control endpoints are also bounded; they may be
rejected if all handler slots are occupied.

The frozen model ID stays unchanged because `serving/` is untouched. A separate
service ID hashes the model ID, scheduling sources and configuration. Instance ID
changes on process restart. The load collector checks all three identities.
Startup/shutdown reports retain original artifact provenance and actual GPU info.
SIGINT/Ctrl+C and SIGTERM stop admission, cancel waiting jobs, invalidate the
active result and attempt bounded draining. A worker that cannot finish within
grace is explicitly reported as not drained, rather than claimed to be cancelled.

## Server validation

Local verification: 13 standard-library scheduler/HTTP/collector tests passed,
plus four existing benchmark tests, syntax and CLI checks. They use synthetic
engines, not PyTorch. Real model parity and performance await server execution.
Developer tests and teaching notes remain local-only under the repository policy.

In the server project root, activate the existing environment and inspect the
actual GPU. No package installation or environment change is required:

```bash
cd "$HOME/AgenticRec"
git pull --ff-only
conda activate "$HOME/conda-envs/agenticrec-hstu-reference"
nvidia-smi
python -m scheduled_serving.run check --config configs/scheduled-serving-server.json
```

Expected: `Artifact admission: PASS`, the actual runtime, and
`Scheduled serving: PASS`. Check uses a temporary loopback port, compares 100
training histories plus the empty-history fallback against direct core responses
(scores and ranking must match exactly), checks invalid inputs and telemetry,
and exits. Reports are under `runs/scheduled-serving/`. No training, checkpoint
updates or test-quality evaluations occur. Original admitted runtime artifacts
must still exist; a Git clone alone does not supply them.

After check passes, terminal A:

```bash
cd "$HOME/AgenticRec"
conda activate "$HOME/conda-envs/agenticrec-hstu-reference"
python -m scheduled_serving.run serve --config configs/scheduled-serving-server.json
```

Stop the old service first if it is still being load-tested. Terminal B:

```bash
cd "$HOME/AgenticRec"
conda activate "$HOME/conda-envs/agenticrec-hstu-reference"
python -m scheduled_serving.load \
  --requests runs/performance/20261107T210935152729Z/requests.json
```

This runs the same eight 60-second points (closed 1/4/16/32, open 50/100/200/400)
with the unchanged benchmark client, connection policy, 20 warmups per point,
32-client cap for open load and five-second socket timeout. Only the endpoint
and server scheduler change. Per-point files add cumulative before/after server
snapshots, counter/histogram differences, error categories and all-attempt
latency. Lifetime maxima are not subtracted to invent per-run maxima. COMPLETE
means measurement finished, not SLO success. Service failures remain measurements.
Failure to drain or identity drift causes FAIL, with partial reports preserved.

After all points finish, stop terminal A with Ctrl+C to write final metrics.
Return its check and final serve reports plus the new load directory. For example:

```bash
mkdir -p runs/exports
find runs/scheduled-serving runs/scheduled-load -type f -name '*.json' -print0 |
  tar --null -czf runs/exports/scheduled-serving-results.tar.gz --files-from=-
```

Download that file through Jupyter. Do not commit reports or exports. Do not run
the old/new service load tests concurrently on the same GPU. The debug config
points to the existing CPU runtime for small server-side synthetic debugging;
it is not permission to run real model benchmarks on the Mac.

## Acceptance before batching

1. Direct-vs-scheduled model responses match; validation and fallback contracts
   remain intact. Source/data/config hashes stay unchanged throughout the run.
2. Queue and connection counts stay bounded; no unbounded request/history logs.
3. Under excess load, report 503/504, socket errors and client drops separately.
   Do not claim improved availability just because slow failures become fast 503s.
4. Compare successful QPS, successful and all-attempt tails, offered-load success
   fraction, queue/core distributions and recovery. Explicit 503 is controlled
   degradation, not successful recommendation throughput.
5. Repeat important operating points before claiming stable improvements. If
   batching follows, measure it as a separate change with another parity check.
