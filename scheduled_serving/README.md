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
engines, not PyTorch. Real-model HTTP parity passed on the server. All eight load points are documented in the
[paired measurements](RESULTS-2026-11-08.md).
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


## Server parity verified

The returned report `runs/scheduled-serving/20261108T022625745414Z/results.json`
was inspected. It records PASS at commit `6bba2c8`, A100-SXM4-80GB MIG 2g.20gb,
PyTorch 2.10.0+cu128, FP32/TF32 off and 24 Torch CPU threads. All six scheduling
source hashes and the scheduling config hash match the local implementation.
The working tree had an untracked `http-load-results.tar.gz`; the source hashes
match, but this is not described as a clean Git working tree.

All 101 requests (100 training histories and the empty-history fallback) returned
exactly the same response as the direct reference, including returned item IDs,
scores and metadata. This checks returned Top-K responses, not every catalog
score on every possible input. The invalid-history request returned 400. Counters
show 101 admitted/completed jobs and 101 recommendation responses written as 200;
no timeout, overload, inference-error or disconnect events were recorded in this
sequential check. Health and metrics endpoints also passed.

Worker and handlers drained successfully. Final queue depth, active inference
and connection count are all zero. Final readiness is false because the check
has intentionally stopped the service, not because startup failed. The frozen
model ID remains `17aaae44451f3d9fe5de7e4ee9a6e2e3a3ff981bf55374c4d836227f76d9abdb`;
the scheduler service ID is
`d9a87e9b6ab2ba9753bbe42b00117a62e8cc56381d935fb4fe985051d272b649`.

The subsequent [load comparison](RESULTS-2026-11-08.md) records the overload behavior and the throughput regression.

## Isolating HTTP thread creation at 200 requests per second

The first bounded service returned 163 overload responses out of 12,000 at this rate. Its successful HTTP p99 was 108.61 ms. The serial reference served all 12,000 at 18.05 ms p99 on an A100 MIG 2g.20gb. The following experiment tests whether creating threads for every connection contributed to this gap.

| Transport | HTTP handlers | Deadline enforcement |
|---|---|---|
| `original` | New thread per admitted connection | New timer per admitted connection |
| `shared-timer` | New thread per admitted connection | One monitor per server |
| `handler-pool` | 64 reusable threads started before load | New timer per admitted connection |
| `pool-and-shared-timer` | Same pool | Same monitor |

All four use the same instrumented code revision, frozen model and request corpus, batch size 1, one inference worker, 16 waiting jobs, 64 connection slots and a 200 ms deadline from accept. The contemporary control is `original`. Every registered socket keeps its connection slot through handler cleanup. The shared monitor watches deadlines in accept order and wakes the handler by shutting down an expired socket.

On the server, stop any other load test or copy of the recommendation service using this GPU. Run from the checkout in one terminal.

```bash
cd "$HOME/AgenticRec"
git pull --ff-only
conda activate "$HOME/conda-envs/agenticrec-hstu-reference"
nvidia-smi
python -m scheduled_serving.compare
```

Each transport runs in a fresh process on a temporary loopback port. Before measurement, each instance checks 100 training histories and the empty-history fallback against direct inference, checks the invalid-input response, and warms the load path. A separate client process sends 200 arrivals/s for 60 seconds. The second round reverses transport order. Each server drains and exits automatically. The eight load windows take eight minutes in total, plus model startup, verification and report writing.

The terminal prints `Transport comparison: COMPLETE` after all measured runs and a path to `runs/transport-comparison/<timestamp>/summary.json`. Return that one file. It contains the checked runtime, identities, per-round outcomes, thread starts, CPU seconds, mean queue and core times, late discards and client tails. Raw request timings stay in the adjacent trial directories. CPU seconds cover the service process, including its warm worker during the snapshot interval and control requests around the load. The client runs in its own process.

The primary target is **12,000 valid responses out of 12,000 offered and successful HTTP p99 ≤ 20 ms in each of the two rounds**. This is an experiment target set from the historical serial observation. Also check successful QPS inside the window, p99 from scheduled arrival, and client dispatch lag. Compare `original` with each single change and then with the combined setting in both rounds. A large drop in thread starts with little change in successful QPS or tail latency is evidence to investigate another source of delay. Two observations per variant support a directional comparison on this hardware.

For a separate overload comparison, `python -m scheduled_serving.compare --settings open-400 closed-32` runs only those two load points. Compare offered success, all-attempt latency, 503/504 counts and client drops. With closed-loop load, fast refusals can increase the number of attempted requests.

### Server results at 200 requests per second

Only `handler-pool` met the predeclared target in both rounds. It kept the per-connection deadline timer, reused 64 HTTP handler threads, and served all 24,000 offered requests across its two 60-second windows. Successful HTTP p99 was 12.46 ms and 13.54 ms. Measured from scheduled arrival, successful p99 was 12.64 ms and 13.69 ms. Window throughput was 199.98 and 199.97 successful requests/s.

Each cell gives round 1 / round 2. Successes include completions during drain. Queue times are means over jobs that started inference.

| Transport | Valid responses out of 12,000 | Successful HTTP p99 (ms) | Queue mean (ms) | Server CPU (s) |
|---|---:|---:|---:|---:|
| Original | 12,000 / 11,703 | 14.47 / 110.49 | 0.61 / 39.61 | 57.50 / 62.68 |
| Shared timer | 11,999 / 11,969 | 76.73 / 92.04 | 7.02 / 10.88 | 58.70 / 57.15 |
| Handler pool | 12,000 / 12,000 | 12.46 / 13.54 | 0.28 / 0.40 | 53.55 / 54.37 |
| Pool and shared timer | 12,000 / 12,000 | 17.75 / 22.16 | 0.51 / 1.93 | 52.31 / 57.54 |

Across all eight windows, all 96,000 offered requests reached the client dispatch path. Every one of the 329 unsuccessful requests received HTTP 503 and corresponds to a full-queue rejection. Of these, 297 occurred in the original transport's second round. Its mean core time rose from 3.90 to 4.70 ms while mean queue time rose from 0.61 to 39.61 ms. The control itself varied substantially between rounds, which limits attribution of the historical 108.61 ms p99 to any single cost.

The original transport started 12,127 handler threads per server lifetime, including parity, warmup and control calls. The handler pool started 64 and still started 12,127 timer threads. Sharing the timer as well brought that count to one. The combined variant served all requests but crossed the latency limit in round 2. These results qualified `handler-pool` for overload testing.

The returned summary records eight distinct server instances at commit `d9af9db` on an A100 MIG 2g.20gb, FP32, with 24 Torch threads. Each instance matched the 101 direct-reference responses and drained its active workers at shutdown. The source/configuration digests match this experiment's Git revision. All four service IDs, eight target outcomes and the client/server outcome counts were checked from the summary. The latency percentiles come from the load client’s summary. Raw per-request samples were retained on the server. Returned summary SHA-256 is `93f5220783b2f9650f65b2b3452f40998c527eb53e474a5b6aa83cbe0d337b20`.

## Server results under overload

Each of eight fresh server instances ran 400 scheduled arrivals/s followed by 32 closed-loop clients, for 60 seconds each. Round 2 reversed transport order. All eight instances matched the 101 direct reference responses and shut down cleanly on the A100 MIG 2g.20gb. Client drops were zero across all 16 windows. Successful QPS counts valid responses completed inside the window. Success counts and percentages also include responses completed during drain.

Each cell gives round 1 / round 2.

| Transport | 400/s successful QPS | 400/s offered success | 400/s successful HTTP p99 (ms) | Closed-32 successful QPS | Closed-32 valid responses / attempts (round 1, round 2) |
|---|---:|---:|---:|---:|---:|
| original | 194.98 / 183.87 | 48.82% / 46.04% | 99.18 / 121.24 | 0.83 / 0.75 | 64 / 68,753, 61 / 68,031 |
| shared-timer | 192.25 / 203.80 | 48.13% / 51.02% | 119.30 / 105.04 | 0.13 / 0.20 | 24 / 90,130, 24 / 92,102 |
| handler-pool | 202.47 / 191.92 | 50.69% / 48.05% | 96.86 / 106.39 | 0.18 / 0.15 | 26 / 90,864, 25 / 85,774 |
| pool-and-shared-timer | 214.98 / 218.42 | 53.82% / 54.67% | 90.64 / 89.17 | 0.12 / 0.07 | 22 / 130,394, 20 / 129,059 |

At 400/s the combined transport delivered the most recommendations in both rounds. It completed 12,916 and 13,122 of 24,000 offered requests. Every unsuccessful request at this load received a full-queue 503. The handler pool alone remains the only variant that met the earlier 200/s latency and success target in both rounds.

Under closed-32, each client sends its next request immediately after any outcome. The combined transport received about 129,000–130,000 attempts per window. Across all eight closed-loop windows, 755,107 attempts produced 266 valid responses including drain, 714,071 full-queue 503s, 40,766 deadline 504s and four transport errors. Only 146 valid responses completed within the windows. The handler pool alone averaged about 177 ms waiting in the queue for jobs that started execution. Its worker discarded 1,109 and 1,030 late results. The execution cutoff was 190 ms from accept. The historical serial service recorded 237.33 successful QPS at the same client concurrency in a separate run.

A follow-up can hold the handler pool and 32 clients fixed and compare immediate resubmission with honoring Retry-After on 503. Measure attempted and successful QPS, queue residence and late results. Client pacing changes offered load, so a higher success percentage alone would be insufficient to establish improved throughput.

The model, request corpus, configuration and source hashes match the experiment at `d9af9db`. Client outcome totals and server 200/503/504 write counts reconcile in every window. Percentiles are those reported by the load client. The experiment verifies drain and shutdown. Measuring recovery time would require observing service after load returns to normal. Returned summary SHA-256 is `576e71c069ab6c827cd6eff8c57103deacd4211dad900f5e6e55498752b575d8`.
