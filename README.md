# AgenticRec

An **HSTU + ItemCF** video recommendation project on MicroLens-50K, covering data preprocessing, reference implementation checks, offline evaluation, error diagnosis, controlled experiments, and an HTTP recommendation service.

The frozen recipe combines HSTU's sequential representations with ItemCF's collaborative signals. Test **NDCG@10 improves by 18.72%** over pure-ID HSTU and **8.31%** over the fixed frequency-corrected baseline. Scoring parity, core latency, and live HTTP calls have been verified on an A100 MIG allocation.

> Updated: 2026-10-24. Results are based on server outputs returned by the project owner; full artifacts remain on the server. Evidence is limited to single-seed offline experiments on a benchmark reused across several frozen recipes. Statistical significance, online CTR/watch-time gains, and production readiness have not been established.

## What This Project Delivers

- **Shared data and evaluation contracts:** deterministic chronological ordering, reversible item mappings, per-user leave-one-out splits, and full-catalog Recall/NDCG.
- **HSTU implementation checks:** a pinned upstream revision, dense-block output/gradient comparisons, synthetic fitting, checkpoint restoration, and the official MovieLens-1M recipe.
- **Evidence-driven improvements:** Popularity, ItemCF, mean pooling, pure-ID HSTU, and title-feature controls to diagnose sparse-target failures and measure complementary scoring signals.
- **A frozen serving recipe:** validation-selected fusion weights, fixed test evaluation, exact full-catalog recommendations, deterministic fallbacks, and model/data identity tracking.

## Offline Results

### MicroLens-50K Test Set

Each recipe evaluates 50,000 user targets on the same data snapshot. Ranking covers all 19,220 mapped items, excludes padding and the user's complete visible history, and resolves score ties by ascending mapped item ID. Targets with zero training-history occurrences remain in the evaluation denominator.

| Recipe | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
|---|---:|---:|---:|---:|
| Popularity | 0.00244 | 0.001254 | 0.00378 | 0.001594 |
| ItemCF | 0.03534 | 0.017473 | 0.05760 | 0.023088 |
| Causal prefix mean pooling | 0.03634 | 0.017816 | 0.05982 | 0.023729 |
| Pure-ID HSTU | 0.04882 | 0.024729 | 0.07238 | 0.030686 |
| HSTU + fixed frequency correction | 0.05174 | 0.027107 | 0.07572 | 0.033161 |
| **HSTU + ItemCF (current serving recipe)** | **0.05610** | **0.029359** | **0.08346** | **0.036262** |

The hybrid improves NDCG@10 / Recall@20 by **18.72% / 15.31%** over pure-ID HSTU and **8.31% / 10.22%** over frequency-corrected HSTU. These are relative changes, not percentage-point differences.

### Which Targets Benefit?

The table counts test users whose target appears in the Top 20, grouped by **target-item occurrences in training histories**. These groups are diagnostic; serving never uses the unknown target's frequency to choose a recommendation strategy.

| Target training frequency | Users | Pure-ID HSTU | Frequency-corrected HSTU | HSTU + ItemCF |
|---|---:|---:|---:|---:|
| 0 | 2,226 | 0 | 0 | 0 |
| 1–5 | 11,557 | 0 | 0 | **86** |
| 6–20 | 19,270 | 153 | 250 | **580** |
| 21+ | 16,947 | 3,466 | **3,536** | 3,507 |
| All | 50,000 | 3,619 | 3,786 | **4,173** |

The hybrid improves some sparse targets while losing a small number of frequent-target hits relative to frequency correction. Recall@20 in the 1–5 group is still only **0.744%**, and zero-frequency targets still receive no hits. Long-tail recommendation and cold start remain unresolved.

See the [frozen hybrid test record](docs/learning-25-frozen-hybrid-test.md).

## Approach and Experimental Rationale

```mermaid
flowchart LR
    H[Chronological watch history] --> S[Latest 32 events: ID HSTU]
    H --> C[Complete history: ItemCF]
    S --> N[Rescale cosine scores]
    C --> F[Filter seen items and normalize]
    N --> M[Fixed weighted fusion]
    F --> M
    M --> R[Full-catalog ranking and history exclusion]
    R --> O[Top-K recommendations]
```

HSTU uses positional bias, tied input/output ID embeddings, and full-catalog cross entropy. ItemCF builds binary user-item cosine similarities from training histories, retains 100 neighbors per item, and aggregates neighbor scores over the complete visible query history.

A predeclared validation grid `{0, 0.1, 0.25, 0.5}` selected **0.25**: require no overall NDCG@10 regression against raw HSTU and at least one hit in the 1–5 frequency group, then maximize overall validation NDCG@10. The frozen recipe is:

```text
score = 0.75 * (HSTU_cosine + 1) / 2
      + 0.25 * ItemCF_score / max(eligible_ItemCF_scores)
```

When CF has no eligible signal, scoring falls back to raw HSTU. Empty histories use training popularity. The current service adds neither titles nor the earlier frequency correction. Ranking scores are not click probabilities.

| Control | Observation and decision |
|---|---|
| Causal prefix mean pooling | Same scoring head, objective, and training budget as HSTU; weaker overall but better on some sparse-target groups. Retain HSTU and investigate scoring behavior. |
| Frequency correction | Improves overall metrics through scoring alone; still no hits for frequency 1–5 targets. |
| Title initialization / fixed fusion | Uses frozen MiniLM features. Initialization reaches test NDCG@10=0.025048 but does not solve sparse-target failures; stable content gains are unproven. |
| Shared title mapping | Gains from learning the mapping mainly come from frequent targets; zero-frequency and lowest-frequency groups decline. Stop this recipe. |
| HSTU + ItemCF | Diagnostics identify exclusive hits in both channels. Bounded validation fusion and frozen test evaluation show a single-seed gain; select this recipe for serving. |

These controls support the current engineering decision. They do not establish that text features are generally ineffective or isolate attention/order as the cause of gains. See the [title controls](docs/learning-20-title-hstu-controls.md), [shared mapping experiment](docs/learning-23-shared-title-mapping.md), [mechanism audit](docs/learning-22-validation-mechanism-audit.md), and [hybrid validation](docs/learning-24-hybrid-validation.md).

### Upstream Implementation Checks

Meta's [generative-recommenders](https://github.com/meta-recsys/generative-recommenders/tree/10c4950343b98bacec9ffd5b8cd879f811fe2326) is pinned at `10c4950343b98bacec9ffd5b8cd879f811fe2326`.

- Twelve dense-block output/gradient comparison cases passed on the server.
- Synthetic fitting reduced loss by 99.3%; checkpoint output, RNG restoration, and next-step continuation checks passed.
- The official MovieLens-1M recipe completed 101 epochs at batch size 128. All six final metric differences from the pinned reference are below 0.00374, within the predeclared 0.02 absolute engineering tolerance.

This does not establish statistical equivalence. MicroLens uses positional bias and full-catalog cross entropy; the official ML-1M recipe uses temporal bias and sampled softmax. Their datasets and protocols differ, so scores are not directly comparable. See the [upstream manifest](configs/hstu-reference-upstream.json) and [ML-1M configuration](configs/hstu-ml1m-train-server.json).

## Recommendation Service and Performance

The frozen recipe exposes `GET /health` and `POST /recommend`. The service binds to `127.0.0.1:8765` on the server, authenticates data/config/source/checkpoint identities at startup, and loads HSTU and training-only ItemCF once.

**Verified environment:** Python 3.11.16, PyTorch 2.10.0+cu128, NumPy 1.26.4, A100-SXM4-80GB MIG 2g.20gb (approximately 19.50 GiB visible memory), FP32, TF32 disabled, and 24 Torch CPU threads.

Against the original offline batched scorer, 100 training-history samples produced **identical Top-20 rankings**, with maximum score error `2.12e-7`, below the `1e-6` threshold.

| Core request latency | Raw HSTU | HSTU + ItemCF |
|---|---:|---:|
| Mean | 3.354 ms | 3.705 ms |
| p50 | 3.348 ms | 3.612 ms |
| p95 | 3.438 ms | 3.932 ms |
| p99 | 3.546 ms | 4.505 ms |

Conditions: 19,220 candidates, batch size 1, concurrency 1, 20 warmup requests and 100 measured requests per mode, with CUDA synchronization around each timed request. Timing includes input validation, inference, GPU-to-CPU transfer, CF scoring, and ranking; it **excludes HTTP, queueing, and startup**. Mean hybrid overhead is approximately 0.35 ms (10.45%). Startup takes approximately 11.38 s. A 100-request sample does not establish stable production tail latency.

Both modes share the loaded model and CF. Peak PyTorch allocated/reserved memory is approximately 47.48/60 MiB; this excludes CPU ItemCF storage and some external allocations and is not total service memory. Live health and recommendation calls passed. HTTP end-to-end latency and concurrent throughput remain unmeasured.

### Run with Existing Server Artifacts

These commands assume the environment, data, and original experiment reports/checkpoints are available. Runtime artifacts are not in Git; cloning alone is insufficient to start the service.

```bash
conda activate "$HOME/conda-envs/agenticrec-hstu-reference"
python -m serving.run serve --config configs/hybrid-serving-server.json
```

From another terminal on the same server:

```bash
curl --fail http://127.0.0.1:8765/health
curl --fail http://127.0.0.1:8765/recommend \
  -H 'Content-Type: application/json' \
  -d '{"history":[1,2,3],"k":10}'
```

`history` contains **mapped IDs** from `item_mapping.json`, ordered oldest to newest. Responses include `items`, `fallback`, `eligible_items`, `model_id`, and `mode`; seen items are excluded. This synthetic request does not represent a real user's preferences. Invalid IDs return HTTP 400; empty histories return popularity recommendations.

See the [serving guide](docs/learning-26-exact-serving.md) for full commands, limits, and artifact-admission dependencies. The current implementation is a local sequential demonstration service.

## Data and Reproduction

### Data Contract

The current snapshot contains **50,000 users, 19,220 items, and 359,708 interactions**. Full sequence length has mean 7.19 and median 6. Input is `data/raw/MicroLens-50k_pairs.csv`, with columns `user,item,timestamp`.

```bash
python -m src.preprocess --min-seq-len 5
```

Preprocessing sorts by user and timestamp, retains sequences with at least five interactions, maps items to contiguous positive integers, and reserves 0 for padding. For each user, the last event is test, the penultimate event is validation, and all earlier events form training history.

| File in `data/processed/` | Contents |
|---|---|
| `sequences.jsonl` | Per-user `user_id`, `train`, `val`, and `test` |
| `item_mapping.json` | Bidirectional original/mapped item-ID conversion |
| `metadata.json` | Preprocessing configuration, statistics, and source checksum |

Training uses left-padded, shifted next-item sequences. Validation sees training history; test sees training history plus the validation item. The model retains the latest 32 events, while ranking filters the complete visible history. This preserves per-user chronology; **it is not a global-time split across users**. Title experiments require a separately supplied title CSV; the frozen service does not use title features.

### Environments and Reproduction Path

Model, DataLoader, training, and GPU validation run on the Linux server. The Mac is used for editing, preprocessing, syntax checks, and lightweight tests. Inspect the actual assigned GPU before execution; it may differ from the originally planned H200.

Environment specifications are in the [Conda file](environments/hstu-reference.yml) and [requirements file](environments/hstu-reference-requirements.txt). For a new environment, use the [preparation script](scripts/prepare_hstu_reference.py) to inspect Python/Conda/Torch/CUDA before planning installation. The driver's reported CUDA capability does not identify the installed runtime.

| Stage | Entry points and documentation |
|---|---|
| Data and shared evaluation | [Dataset](src/data/dataset.py), [metrics](src/evaluation.py), [baselines](docs/learning-04-evaluation-and-baselines.md) |
| MicroLens HSTU | [Integration and checks](docs/learning-11-microlens-hstu.md), [full training](docs/learning-12-microlens-full-training.md) |
| Diagnostics and controls | [Mean pooling](docs/learning-17-prefix-mean-control.md), [title features](docs/learning-19-frozen-title-features.md), [mechanism audit](docs/learning-22-validation-mechanism-audit.md) |
| Frozen hybrid | [Validation fusion](docs/learning-24-hybrid-validation.md), [frozen test](docs/learning-25-frozen-hybrid-test.md) |
| Serving and performance | [API, parity, and measurements](docs/learning-26-exact-serving.md) |

A fresh reproduction must generate data, train baselines/models, complete validation selection, and then run fixed evaluation and serving. Existing admission configurations reference original timestamped artifacts. New runs must explicitly configure their corresponding artifacts rather than disable hash checks to bypass dependencies. Linked teaching notes are primarily in Chinese.

With existing artifacts, run the serving checks:

```bash
python -m unittest tests.test_serving tests.test_serving_torch -v
python -m serving.run benchmark --config configs/hybrid-serving-server.json
```

All seven focused serving tests and the benchmark passed on the server. Documentation updates do not require repeating completed experiments.

### Experiment Artifacts

Code, configurations, tests, and documentation are synchronized through Git. Raw/processed data, physical environments, logs, caches, and checkpoints are excluded. Formal runs record Git state, configuration, seed, data checksums, actual hardware, and relevant versions. Serving also records model identity and source-file hashes. Checkpoints use `state_dict`; outputs live under ignored `runs/` directories.

| Completed run | Directory under `runs/` |
|---|---|
| Pure-ID HSTU training | `hstu-microlens/20261022T023420952080Z/` |
| Frozen hybrid validation | `hybrid-validation/20261024T203229829396Z/` |
| Frozen hybrid test | `hybrid-selected-test/20261024T204032578400Z/` |
| Serving parity and latency | `hybrid-serving/20261024T204850491181Z/` |

## Repository Layout

```text
src/data/         Sequential Dataset and DataLoader
src/models/       Portable dense HSTU blocks and pure-ID model
src/preprocess.py Data preprocessing
src/evaluation.py Shared full-catalog ranking and metrics
src/baselines.py  Popularity and ItemCF
controls/         Causal prefix-mean control
content_hstu/     Title initialization and fixed-fusion controls
mapped_hstu/      Shared title-mapping controls
title_features/   Title alignment, frozen encoding, and cache verification
diagnostics/      Mechanism diagnostics, statistics, and artifact admission
serving/          Request validation, model loading, HTTP, and core benchmarks
scripts/          Environment preparation, audits, and frozen evaluation
configs/          Experiment and device configurations
tests/            Focused data, metric, model, and serving checks
docs/             Teaching notes, experimental decisions, and evidence
```

## Limitations and Next Steps

The frozen recipe has completed offline evaluation and a local HTTP demonstration. Multi-seed robustness, global-time replay, HTTP concurrency, and online gains remain unverified. There is no full SASRec comparison, so no superiority over SASRec is claimed. Negative title results apply to the tested recipes. Agents are outside the current implementation scope; MiniOneRec and related models remain reading references.

Next priorities are shareable experiment artifacts and demonstrations, followed by end-to-end service measurements and robustness checks as needed. Exact scoring is already measured at the current catalog size; ANN or a new backbone would require additional evidence.
