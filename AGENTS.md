# AgenticRec Project Instructions

## Role

Act as a senior recommender-systems engineer and a hands-on mentor. Help build the project incrementally, explain important design choices in plain language, and prefer a small verified implementation over a large speculative one.

## Learning and delivery mode

Deliver complete, runnable implementations by default. Do not leave code gaps,
student TODOs, or partially implemented exercises for the user unless the user
explicitly requests an exercise for a specific topic.

For every meaningful implementation milestone, also create or update a concise
Markdown teaching note under `docs/`. The note should explain the most important
code paths, input and output shapes, recommender-system concepts, design choices,
and verification commands. Focus on the parts worth learning rather than
documenting every utility line.

All repository code comments and docstrings must be written in English.
Teaching documents and explanations to the user may be written in Chinese.

## Development and execution environments

Use one device-independent codebase with separate local and server configurations.

The authoritative training target supplied by the user is:

- GPU: NVIDIA H200 NVL
- GPU memory reported by `nvidia-smi`: 143771 MiB
- NVIDIA driver: 570.211.01
- Maximum CUDA version reported by the driver: 12.8
- MIG: disabled in the supplied snapshot
- Conda executable: `/opt/conda/bin/conda`, version 26.5.3
- Base Python: 3.13.14; do not use it directly for this project
- `nvcc`: not installed
- Slurm `sbatch`: not installed

Use a Python 3.11 Conda prefix environment for the server baseline unless a later HSTU dependency establishes a different requirement. Prefer an actual environment path such as `~/conda-envs/agenticrec-h200`, stored on persistent home storage, and activate it by prefix. Keep the reproducible environment specification in the repository, but never place or commit the physical Conda environment directory inside the repository. Register the prefix environment as a Jupyter kernel after creation.

The persistent prefix bootstrap was verified on 2026-10-04: `/home/zichenl8/conda-envs/agenticrec-h200` contains Python 3.11.16 and pip 26.2.1 and occupied approximately 232 MiB before project libraries were installed. The server's `/home/zichenl8/AgenticRec` directory was empty at verification time and is reserved as the future Git checkout.

The later 2026-10-10 inspection reported an A100 MIG allocation with 19968 MiB,
not the intended H200. Reinspect the assigned GPU and installed stack before
server execution; the environment name does not identify the actual device.
On 2026-10-18, the user reported all eight Dataset CPU tests passing on the
server at commit a129baa. All three processed-data file SHA-256 values matched the Mac on 2026-10-18,
as reported by the user. Later server outputs verified Python 3.11.16, PyTorch
2.10.0+cu128, FBGEMM-GPU 1.5.0, CUDA reference checks and official ML-1M
training on A100-SXM4-80GB MIG 2g.20gb (19.50 GiB visible). These are historical
results; inspect the actual allocation before every new experiment.

Treat the `nvidia-smi` CUDA value as driver capability, not proof of the installed CUDA toolkit or the correct PyTorch build. Before creating or changing the server environment, ask for or inspect the server's Python, Conda, PyTorch, CUDA runtime, package-manager, and job-scheduler details. Do not guess an exact CUDA/PyTorch build from the screenshot alone.

Execution policy:

- Treat the H200 server as the authoritative execution environment for all model, DataLoader, PyTorch, training, mixed-precision, optimized-attention, and HSTU validation.
- Use the Mac only for code editing, preprocessing, syntax/compile checks, deterministic pure-Python checks, and the smallest inexpensive smoke tests when they materially reduce iteration time.
- Do not routinely run the full test suite or realistic model execution on the Mac. Prepare the server command and expected result, then ask the user to run it and return the output.
- Do not treat a Mac-only run as final validation for training, CUDA, mixed precision, optimized attention, or HSTU behavior.
- Prepare exact server commands and expected outputs for GPU-dependent checks. The user will run them on the H200 server and return the output for diagnosis.
- Clearly label results as locally verified, server verified, or awaiting server verification.
- Deliver complete code even when final execution must happen on the server. It is acceptable to perform only syntax checks or lightweight CPU tests locally, but clearly mark GPU-dependent behavior as awaiting server verification.
- Keep model and training code device-independent. Do not hardcode `.cuda()`, absolute machine-specific paths, GPU counts, batch sizes, worker counts, or CUDA-only behavior in shared modules.
- Put machine-dependent settings in explicit Mac/debug and H200/server configuration files.
- Prefer CPU for deterministic unit tests. Use CUDA on the H200 for full training and performance validation.
- Prefer BF16 on the H200 only after the installed PyTorch/CUDA stack and a small numerical smoke test confirm support.
- Do not size batches from the nominal 143771 MiB alone. Start conservatively and measure allocated and peak memory because the GPU may be shared or already occupied.
- For HSTU or optimized attention, keep a portable PyTorch reference path when practical and isolate CUDA/Triton-specific acceleration behind a clear interface.

Code and artifact synchronization policy:

- Use Git as the source of truth for code, configuration, tests, and documentation.
- Develop and commit on the Mac, push to a private remote, and update the server checkout with fast-forward pulls. Avoid editing the same source files independently on both machines.
- After completing requested project changes and the appropriate local checks, commit and push task-related code, configuration, and documentation to the existing private remote automatically. The user authorized this default on 2026-10-21; do not ask them to repeat the push request. Preserve unrelated local changes and keep datasets, logs, checkpoints, environments, and secrets out of commits.
- Do not commit raw datasets, processed datasets, checkpoints, run logs, credentials, or secrets to Git.
- Transfer datasets and large artifacts separately with `rsync` or the server's designated shared/object storage.
- Preserve and compare `metadata.json` and checksums after data transfer.
- Save checkpoints as model and optimizer `state_dict` values plus config, epoch, metrics, and data/version identifiers; do not serialize an entire live model object.
- Record the Git commit, configuration snapshot, processed-data checksum, package versions, and random seed for every formal server experiment.

## Long-term objective

Build an interview-ready, reproducible HSTU-based video recommendation system
on MicroLens data. The user prioritizes industrial relevance, reliable
reproduction, and a defensible improvement over novelty by model name.

The user revised the project direction on 2026-10-16. These decisions supersede
the earlier SASRec-first and mandatory-agent progression:

1. Finish server verification of the existing sequential Dataset contract.
2. Establish shared offline ranking metrics and inexpensive Popularity/ItemCF baselines.
3. Integrate a pure-ID HSTU baseline, using a pinned official/reference implementation and a portable reference path where practical.
4. Diagnose errors and choose one improvement supported by experiments.
5. Deliver reproducible experiments and a small measured recommendation service.

Full SASRec reproduction is optional, not a prerequisite for HSTU. Reuse the
existing Dataset classes even though their names contain `SASRec`; rename only
when implementation work makes it useful. A claim of superiority over SASRec
requires an actual comparison under the same evaluation protocol.

Agents are outside the current scope. The target is automatic feed
recommendation without requiring users to state requests. Titles and other
content features are optional hypotheses, not committed improvements. ID
embeddings can already learn semantic-related collaborative structure; adding
text must earn its cost against a well-trained interaction-only baseline.

HSTU is the selected backbone. OneRec, OpenOneRec, PinRec, and MTGR are reading
references, not additional implementation milestones. Do not switch backbones
merely because a newer paper appears. See `docs/project-roadmap.md` for the
decision record, evidence, and proposed diagnostic experiments.

Do not jump to a later stage merely because it is more novel. Each stage must produce a runnable, inspectable artifact and pass its validation criteria first.

## Completed stage: MicroLens interaction preprocessing

The immediate objective is to transform the MicroLens-50K interaction log into user sequences that both SASRec and HSTU can consume.

Known local inputs:

- `data/raw/MicroLens-50k_pairs.csv` is the required interaction file.
- Its observed columns are `user`, `item`, and `timestamp`.
- `data/raw/readme.txt` documents the source format.
- `data/raw/MicroLens-50k_titles.csv` is optional metadata for future text or multimodal work.
- Raw videos, frame archives, audio, covers, and extraction/download scripts are not needed during the interaction-only stage.

Treat `data/raw/` as immutable. Write all generated artifacts to `data/processed/`.

## Required preprocessing pipeline

Implement and verify these steps in order:

1. Load the interaction CSV and validate the required columns and data types.
2. Report basic statistics: number of users, items, and interactions; per-user sequence-length statistics; missing values; and duplicate rows.
3. Sort interactions chronologically by `user` and `timestamp`. Never use a random split for sequential recommendation.
4. Remap item IDs to contiguous integers suitable for an embedding table. Reserve index `0` for padding unless the downstream model explicitly establishes another convention. Save both forward and reverse mappings.
5. Group each user's chronologically ordered interactions into one item sequence.
6. Filter sequences that are too short. Use a configurable `min_seq_len`, initially `5`, rather than burying the value in code.
7. Apply a per-user leave-one-out split: the last item is test, the penultimate item is validation, and all earlier items are training history.
8. Save deterministic processed data and item mappings in a simple documented format under `data/processed/`.
9. Manually inspect at least 5-10 sampled users, showing full sequence, train history, validation target, and test target.
10. Document the final data contract so the later PyTorch dataset and models consume the same representation.

The preprocessing milestone is complete only when all of the following are demonstrated:

- User, item, and interaction counts can be reproduced.
- At least one correctly ordered user sequence can be displayed.
- Each user's train/validation/test split preserves chronology and excludes that user's held-out target from its input history. This does not establish global-time leakage freedom across users.
- Saved mappings can convert IDs in both directions.
- A rerun on unchanged raw data produces identical outputs.

This milestone has been implemented and verified for the current MicroLens-50K
input. The sequential Dataset and DataLoader contract passed its eight server
CPU tests on 2026-10-18. All three processed-data checksums also matched the Mac
on that date. Shared evaluation and simple baselines subsequently passed server verification;
the MicroLens HSTU integration tests and three-epoch smoke run subsequently
passed on the server; the first 30-epoch full-data run also returned PASS.

The SASRec Dataset and DataLoader contract has been fully implemented and its
eight focused CPU unit tests passed locally on 2026-10-05 and on the server
at commit a129baa on 2026-10-18. The tests use synthetic records and do not
verify the transferred MicroLens files or CUDA behavior. Pure-ID HSTU and learned ID embeddings have since been implemented and verified
with synthetic tasks and the separate official ML-1M experiment. MicroLens HSTU
training/evaluation passed its 33 focused server CPU tests and three-epoch
real-data smoke run; its first full-data 30-epoch run and frozen test evaluation subsequently returned PASS. No title
embedding table or content fusion has been implemented.

## Locally verified data findings (2026-10-16)

- 50,000 users, 19,220 mapped items, and 359,708 interactions.
- Full sequence length: mean 7.19416, median 6, P99 24, maximum 218.
- 18,490 items occur in training histories; 6,970 of them occur 1-5 times.
- 730 mapped items never occur in any training history. Their targets account
  for 844 validation examples and 2,226 test examples (4.452% of test users).
  These are target-example counts, not distinct-item counts or verified new
  publication events.
- All 19,220 mapped items have one nonblank title row; semantic quality has
  not been assessed. Raw titles and item IDs are not pretrained embeddings.
- No validation/test target repeats within its own visible evaluation history
  in this dataset snapshot.

These findings favor diagnosing sparse feedback and short-history behavior
before pursuing long-sequence acceleration. They do not prove text will help.
The auditable definitions and reproduction command are in
`docs/data-audit-2026-10-16.md`.

## Stage gates

After preprocessing is verified, proceed in this order unless the user explicitly changes priorities:

### Stage 2: shared evaluation and simple baselines

- Pass the eight existing Dataset tests on the server and verify processed-data checksums.
- Implement full-catalog Recall/NDCG@10/20 with explicit padding exclusion,
  visible-history filtering, deterministic ties, and candidate eligibility.
- Fit Popularity and ItemCF using training histories only. Report evaluation
  targets with zero training-history occurrences instead of silently removing them.
- Keep per-user leave-one-out as the established benchmark. Global-time replay
  is a separate, versioned protocol requiring timestamp and item-availability checks.

### Stage 3: pure-ID HSTU baseline

- Pin the upstream revision, dependencies, configuration, and reference results.
- Verify a small official public-data recipe before interpreting MicroLens results;
  establish tolerances and document adaptations instead of expecting identical metrics.
- Validate causal/padding behavior, finite gradients, small-sample fitting,
  checkpoint restoration, and server training before adding features.
- Reuse the existing item mapping and sample contract. Add aligned timestamps
  only when enabling temporal bias, without changing held-out targets.
- Keep architecture comparisons separate from fused-kernel and serving benchmarks.

### Stage 4: evidence-driven improvement

- Diagnose metrics by user history length and target-item training frequency,
  including the zero-occurrence group. Select settings on validation data only.
- A proposed inexpensive neural control is masked history mean pooling with
  the same item embeddings, scoring head, objective, and tuning budget. This
  tests whether short histories benefit from the more complex encoder.
- If diagnostics motivate text, first compare ID-only, content initialization,
  and simple fusion; consider frequency-dependent fusion only after that.
- Any text fusion must define both history-input and candidate-output
  representations. Freeze the text encoder first and cache aligned features.
- Do not assume more epochs create independent interaction evidence, that
  sequentially adjacent items share a topic, or that semantic similarity equals preference.
- Introduce larger media assets or additional datasets only for a concrete
  hypothesis that the current data cannot test.

### Stage 5: reproducible delivery and measured serving

- Start with exact candidate scoring at the current catalog size; add ANN only
  with a measured latency/recall tradeoff against exact retrieval.
- Record warmup, batch size, concurrency, device, precision, and candidate count
  when reporting serving latency or throughput.
- Provide model/data/config identifiers and deterministic fallback behavior.
- Describe offline metrics honestly. The available interaction table does not
  establish exposure negatives, online CTR/watch-time lift, or production deployment.
- Keep hypothetical future resume copy separate from actual implementation status.

## Engineering rules

- Inspect the repository and existing outputs before proposing work; continue from the earliest incomplete milestone instead of restarting.
- Keep raw data untouched and make preprocessing deterministic.
- Prefer configuration values and explicit schemas over hidden constants.
- Guard against temporal leakage in preprocessing, negative sampling, feature construction, and evaluation.
- Make the smallest coherent change, then run an appropriate check before expanding scope.
- Keep code readable for a portfolio: clear module boundaries, concise comments, reproducible commands, and a README that explains both what was done and why.
- Do not download large media assets or add infrastructure such as Parquet/LMDB unless the current experiment requires it.
- When information is uncertain, inspect the local data or authoritative dataset documentation instead of guessing.
- When reporting progress, distinguish clearly between implemented, verified, planned, and optional work.

## Guidance style

- Lead with the next concrete action and its completion criterion.
- Explain recommendation-specific concepts when they first matter, especially chronology, masking, padding, negative sampling, leakage, and offline metrics.
- Avoid overwhelming the user with future stages while a foundational stage is incomplete.
- When offering code, make it runnable in this repository and state the exact command to verify it.

## Current implementation and verification status (2026-10-22)

This section supersedes older pending-stage descriptions in historical notes.
Server results below are based on outputs returned by the project owner; raw
server run artifacts have not been independently inspected from this Mac.

- Stage 2 is server verified: 18 Dataset/baseline tests passed, and full
  validation/test Popularity and ItemCF evaluations completed. Validation ItemCF
  Recall/NDCG@10 = 0.04804/0.024495; test = 0.03534/0.017473.
- The pinned official HSTU revision is
  `10c4950343b98bacec9ffd5b8cd879f811fe2326`. Twelve dense block output/gradient
  comparison cases passed on the server. Synthetic fitting reduced loss by
  99.3%, and checkpoint output/RNG/next-step continuation passed.
- Official ML-1M training completed 101 epochs, batch 128. All six final metric
  differences from the pinned reference are below 0.00374, within the declared
  0.02 absolute engineering band. This does not establish statistical equivalence
  or MicroLens quality. See README and `docs/learning-10-official-ml1m-training.md`.
- MicroLens HSTU integration is now implemented in `src/run_hstu_microlens.py`
  and `src/microlens_training.py`, with separate debug, server smoke, and full
  configurations. It reuses the existing Dataset, position-only IDHSTU and shared
  full-catalog ranking rules. The initial loss is full-catalog cross entropy,
  excluding padding; it is an adaptation, not the official sampled-softmax recipe.
- Sixteen tiny baseline/configuration tests and syntax checks passed locally.
  The user then reported all 33 focused server CPU tests passing (11.080s),
  including the synthetic epoch-boundary continuation check. The MicroLens
  three-epoch smoke run returned PASS at
  `runs/hstu-microlens-smoke/20261022T022602941154Z/results.json`.
  Training loss was 9.946178 -> 8.179191 -> 7.135582; fixed training-batch probe
  loss was 8.586332 -> 7.200485 -> 6.391037; validation NDCG@10 was
  0.001397 -> 0.004264 -> 0.005421. The smoke uses 1,024 training users and
  1,024 validation users, so it is not a full-data baseline comparison.
  The user subsequently returned actual runtime and check fields: PyTorch
  2.10.0+cu128, CUDA runtime 12.8, cuda:0, A100-SXM4-80GB MIG 2g.20gb,
  20,937,965,568 visible bytes. Peak PyTorch allocated/reserved memory was
  0.154/0.193 GiB; elapsed time was 17.091s. Parameters updated, checkpoint
  outputs matched exactly at epoch 3, probe loss decreased, and processed-data
  checksums were unchanged. These memory figures exclude some external GPU
  allocations and do not establish full-data memory or serving performance.
  Git/data-hash smoke-report fields were not shown.
- The user subsequently reported a complete 30-epoch full-data run and
  `MicroLens HSTU train: PASS`, saved at
  `runs/hstu-microlens/20261022T023420952080Z/results.json`.
  The printed validation NDCG@10 peaks at epoch 28, 0.037724, compared with
  fixed ItemCF validation NDCG@10=0.024494992910568 (approximately +54.0%
  relative, from rounded log output). Epoch 30 scores 0.037618; use best.pt,
  not last.pt, for the frozen first-candidate test evaluation. Training loss
  goes from 9.077149 to 8.120724. The user subsequently returned exact epoch-28
  selection NDCG@10=0.03772422803068947, validation Recall@20=0.10176 and
  matching HSTU/ItemCF data hashes and group counts. The validation 1-5 group
  Recall@20 is 0 vs ItemCF 0.04790749; 6-20 is 0.01010315 vs 0.06035440;
  21+ is 0.21303345 vs 0.09257406. The regression is established on validation,
  not just test. Full-training runtime was A100 MIG 2g.20gb, 1196.934s,
  peak allocated/reserved 171903488/243269632 bytes (0.16010/0.22656 GiB).
  The score audit later reproduced all four uncorrected validation metrics:
  Recall@10/20=0.07112/0.10176 and NDCG@10/20=
  0.03772422803068947/0.04544371132907886. Full provenance fields are not shown.
- The user returned frozen best.pt test evaluation PASS at
  `runs/hstu-microlens/20261022T030714112850Z/results.json`, all 50,000 users.
  Recall@10/20=0.04882/0.07238; NDCG@10/20=
  0.02472889109808947/0.030686079403136588. Relative gains over fixed ItemCF
  are 38.14%/25.66% for Recall and 41.53%/32.91% for NDCG. These are one-seed,
  offline protocol results, not online lift or a controlled architecture claim.
  Frequency-0 (2,226 users) and frequency-1-5 (11,557 users) targets have no
  Top-20 hits; frequency-6-20 contributes 153 and frequency-21+ contributes
  3,466 of the 3,619 Top-20 hits. Do not choose improvements from test slices.
- A local pure-Python supervision audit at L=32 found 259,708 history events,
  206,613 retained positive prediction positions, 18,490 history items and
  18,008 positive-target items. Of history-observed items, 482 never receive a
  positive target. Validation frequency-1-5 has only 74/7,264 targets with zero
  retained positive occurrences; missing supervision alone is not a sufficient
  explanation for all tail behavior. The audit reads no test labels. Its two
  synthetic tests and full local data audit passed; server replay is pending.
  See `scripts/audit_hstu_supervision.py` and teaching note 13.

- The user returned four validation score-audit variants at
  `runs/hstu-score-audit/20261022T145804815608Z/results.json`. Alpha=0 exactly
  reproduces the frozen validation metrics. Predeclared validation NDCG@10
  selects alpha=0.5: 0.040344999048154036 (+6.95% vs raw HSTU), Recall@20=0.10616.
  Alpha=1 worsens overall metrics. Every alpha still has zero 1-5-frequency
  target hits; alpha=0.5 raises 6-20 Recall@20 to 0.01523407, still below ItemCF.
  The final PASS line and diagnostic resource fields were not included in the
  pasted output; the later fixed test PASS confirms that report-status guard succeeded.

- The fixed alpha=0.5, smoothing=1 recipe returned
  `HSTU selected-score test: PASS` at
  `runs/hstu-score-selected-test/20261022T151722543561Z/results.json`.
  Test Recall@10/20=0.05174/0.07572 and NDCG@10/20=
  0.027107023028108444/0.03316138641732385, all 50,000 users.
  Relative gains over raw HSTU are 5.98%/4.61% Recall and 9.62%/8.07% NDCG.
  Top-20 adds 167 hits: frequency-6-20 gains 97 (153->250), 21+ gains 70
  (3466->3536); frequency 0 and 1-5 remain zero. This validates a single-seed
  scoring improvement, not a new encoder, cold-start solution or online lift.
  The actual corrected-test runtime/memory fields and separate CPU test logs
  have not been returned; the full real-data test entry point did return PASS.

- The causal prefix-mean control is now implemented in `controls/mean_pooling.py`,
  with `controls/run_mean_pooling.py` providing complete train/evaluate and
  epoch-boundary continuation. It reuses the existing Dataset, scoring head,
  objective, training and ranking routines. Config validation requires matching
  HSTU settings except the output directory and control/reference identifiers.
  The reference CPU initializer supplies the same untrained ID table and initial
  RNG state; only that table is trainable. No learned positions or attention
  survive into the control. Prefixes use deterministic dense triangular means,
  not future items. Existing src and diagnostic/evaluation scripts are unchanged,
  preserving historical checkpoint identities. New control/source/config hashes
  identify these separate checkpoints. Five standard-library config tests,
  syntax/CLI and unchanged-source checks passed locally; neural tests await server.

The user reported the three-epoch mean-control smoke PASS at
`runs/mean-pooling-smoke/20261022T155527051279Z/results.json`.
Mean/reference parameter counts were 1,230,144 / 1,273,406. Training loss fell
10.029359 -> 7.695507, probe 9.183881 -> 6.918766, and validation NDCG@10
0.000836 -> 0.006182. PASS enforces embedding updates, reduced probe loss,
exact restored outputs and unchanged data hashes. Full resource/runtime fields
were initially unreported; the user subsequently confirmed all 18 focused tests OK.

The user also reported the full 30-epoch mean-control train PASS at
`runs/mean-pooling/20261022T160521830353Z/results.json`.
Best validation NDCG@10 was approximately 0.029651 at epoch 19 (log-rounded),
versus raw HSTU 0.03772422803068947 at epoch 28. HSTU is approximately 27.23%
higher relative to Mean. Final Mean loss was 8.059710; final validation NDCG@10
was 0.029334. This is a single-seed matched-recipe comparison, not a pure
attention/order ablation or a significance claim. The next user report supplies full precision, Mean slices and resources below;
focused CPU tests were subsequently confirmed OK by the user.

The user subsequently returned the exact Mean validation score
0.02965074791224214 and slices: frequency 1-5 NDCG@10=0.0058661551,
Recall@20=0.0136288546 (99 hits); 6-20 NDCG@10=0.0169910603,
Recall@20=0.0494578154 (935 hits); 21+ NDCG@10=0.0486670575,
Recall@20=0.1552616696 (3569 hits). Raw HSTU is better overall and in 21+,
but worse in both observed lower-frequency groups. Zero-frequency remains zero.
This motivates investigating representation/scoring before attributing tail
failure to missing text. Mean does not beat ItemCF in the lower-frequency groups.
Mean time was 762.4217709070072 seconds, allocated/reserved peak bytes
160582144/230686720, on A100-SXM4-80GB MIG 2g.20gb. Versus the earlier HSTU
1196.934-second run, total run time was 36.3% lower; this is not serving latency.
Restoration, probe reduction and embedding update checks returned true/nonzero;
restored last epoch 30 is distinct from selected best epoch 19.

HSTU best-validation history-length slices have now been returned, with the
same group counts as Mean. NDCG@10 for lengths 1-5/6-10/11-20/21+ is
0.03338191598/0.04837605255/0.05298267643/0.06827897215.
Versus Mean this is +9.79%/+62.51%/+135.64%/+502.89%; the last group has
only 608 users and a low Mean baseline. HSTU also wins Recall@20 in all groups.
History length and target frequency are separate axes; without joint slices do
not infer that each short-history user should get Mean or that order alone
causes the gain. Retain HSTU; lower-frequency target performance remains a
specific weakness despite overall and history-length wins.

A frozen mean-test admission wrapper is now implemented in
`scripts/evaluate_mean_control.py`, outside checkpoint-bound src/controls.
It verifies paired configurations, full validation history, shared/data/user
identities, current source/data hashes, earliest validation winner and best.pt
hash before invoking the existing test evaluator. It exposes no score or epoch
selection overrides; --check-only avoids model execution. Six standard-library
corruption/admission tests passed locally, along with syntax and CLI checks.
No model, training config, scoring routine or existing source identity changed.

The user returned `Frozen mean preflight: PASS` followed by
`Mean pooling evaluate: PASS`, using epoch 19 best.pt. The test report is
`runs/mean-pooling/20261022T163358097988Z/results.json`.
Mean test Recall@10/NDCG@10/Recall@20/NDCG@20 are
0.03634/0.017815997493387952/0.05982/0.02372864608208967.
Raw HSTU test NDCG@10 is 38.80% higher than Mean; corrected HSTU is 52.15%
higher, including both encoder and correction changes. Mean frequency 1-5/6-20/21+
Recall@20 is 0.00830665398/0.03191489362/0.13453708621 (96/615/2280 hits).
Zero-frequency remains zero. Mean loses to raw HSTU in every test history-length
group, but wins in the two lower observed target-frequency groups, confirming
validation observations. Actual paired configurations, users, data, sources and
best checkpoint passed the preflight. The user subsequently confirmed all 18 focused server tests OK. This evaluation's
resource fields have not been returned.

The Mean real-data control record is complete; do not repeat training or test
without a concrete reason. Retain HSTU and keep all reported test recipes frozen.
A subsequent title-feature experiment remains optional and must be designed from
prior validation evidence: frozen text encoder, aligned cached features, explicit
history/candidate representations, ID-only/init/simple-fusion comparisons and
validation-only selection. Frozen title-feature preparation is now implemented as
described below; no actual encoded cache, fusion model or gain exists yet.
Do not treat test feedback as a new tuning signal or claim significance from one seed.


## Frozen title-feature preparation (2026-10-22)

The user authorized the next stage after confirming all 18 mean-control tests OK.
A complete, isolated title feature pipeline is implemented in `title_features/`
and `scripts/prepare_title_features.py`, with audit/preflight/smoke/build actions.
It does not change checkpoint-bound src/controls or any baseline training config.
Config `title-features-minilm-server.json` pins all-MiniLM-L6-v2 revision
1110a243fdf4706b3f48f1d95db1a4f5529b4d41, transformers 4.57.1, 384 dimensions,
max 256 tokens, batch 32, FP32/eager attention and seed 42. The model stays frozen;
attention-masked token mean pooling is normalized. Row 0 is zero; full cache
shape is [19221,384]. Smoke caches cannot be loaded for training. Source/data/
config/model snapshot/feature hashes, Git, versions, runtime and resource use
are recorded. This is feature preparation only, not title fusion or evidence of gain.

Local complete title audit PASS: 19220 CSV/catalog items, zero missing/blank/
extra mapped titles, 87 duplicate-text excess rows; character min/median/max
4/109/6293, 798 titles with non-ASCII and 88 with CJK characters. This is not
language identification. Raw CSV SHA256 is
244aad5380cbbe0fb43458cfcda5ebe820f534384602f80a64dbbcd07dd30e49;
aligned normalized title digest is
e4dee17e5a4fbdf31d79d1462ccc60411c81051e125c48ba28c7de86bbfdcdf7.
Eleven lightweight alignment/config/cache/preflight tests passed locally; three
PyTorch token-pooling checks and actual text inference await the server. No model
was downloaded or executed on the Mac. Outputs remain ignored under runs/.

Next action: run the 14 focused title tests and the read-only server preflight
in teaching note 19. Preflight works even if title CSV is not yet transferred.
Inspect package versions, Python/Conda/pip, torch/CUDA/GPU, pip check and scheduler
before installing anything. A requirements file pins transformers only; final
transitive versions await the actual environment. Do not replace verified torch
or guess missing packages. Transfer titles separately if absent (Git ignores CSV).
After environment/files are verified, audit then 64-item smoke, review it, then
full cache build; only then implement downstream initialization/fusion with
explicit history and candidate representations and validation-only selection.


## MiniOneRec prioritization (user confirmation, 2026-10-22)

The user explicitly chose to continue the current title-feature/HSTU work and
asked that MiniOneRec be considered later when appropriate. Treat it as an
optional subsequent comparison, not an authorization to start a second training
pipeline now. Reassess after the title experiment and a measured HSTU serving
path have concrete results, considering actual GPU allocation and implementation
budget. Any later comparison must preserve the same data/evaluation contract and
separate semantic features, SID construction, SFT and optional RL effects.
The immediate gate remains the 14 title tests and server preflight; no new
server output has been supplied since the feature-preparation implementation.


## Title preflight server update (2026-10-23)

The user returned all 14 title tests OK in 0.360 seconds and preflight NEEDS_SETUP
at `runs/title-features-minilm/20261023T011421721161Z/results.json`.
Python 3.11.16, pip 26.2.1, Conda 26.9.2, NumPy 1.26.4, torch 2.10.0+cu128,
CUDA runtime 12.8 and A100-SXM4-80GB MIG 2g.20gb (20937965568 bytes) were
reported; driver 570.211.01. Full-GPU memory query lacked permission, but torch
allocation properties were available. pip check passed, nvcc/sbatch absent.
All three processed files exist; raw title CSV is missing. The four HF packages
are absent. Do not ask for the same environment inspection again.

Requirements now pin transformers 4.57.1, huggingface-hub 0.36.0,
tokenizers 0.22.1 and safetensors 0.6.2, compatible with the declared dependency
ranges. Install using title-features-constraints.txt to preserve verified torch
2.10.0+cu128 and NumPy 1.26.4, without torch extras or blanket upgrades. Actual
installation/import compatibility is still pending server execution. Transfer
the existing 2.3 MB raw title CSV separately (Jupyter upload is suitable), verify
its previously recorded SHA256, then preflight -> audit -> 64-item smoke. Review
smoke output before full build. The 14 tests need not be repeated merely to
resubmit their output. Detailed commands are in teaching note 19.


## Title setup and smoke server PASS (2026-10-23)

The user supplied installation success, pip check clean, raw title SHA256 PASS,
preflight PASS at `runs/title-features-minilm/20261023T013402313329Z/results.json`,
audit PASS at `runs/title-features-minilm/20261023T013439647115Z/results.json`,
and 64-title smoke PASS at
`runs/title-features-minilm/20261023T013447156086Z/results.json`.
All four pinned HF versions match, BertModel imports successfully, and torch
2.10.0+cu128/numpy 1.26.4/CUDA runtime 12.8 remain unchanged on A100 MIG 2g.20gb.
The six pinned snapshot files including a 90.9 MB weight file downloaded, and
both 32-item batches completed. Server title audit and aligned digest match Mac.
Smoke PASS enforces frozen parameters, exact repeated inference, batch-padding
consistency, finite unit rows, zero padding, exact saved output, and unchanged
inputs/source/snapshot. Full smoke report fields (time/memory/hashes) were not
shown, so do not invent their values.

The user subsequently returned full build PASS with all 19220 titles encoded,
at `runs/title-features-minilm/20261023T013804609858Z/results.json`.
The same directory contains features.npy; full shape [19221,384], FP32, zero
padding/finite/unit-row and exact saved-output checks are enforced by the build.
Actual full report truncation count, file hash, runtime and memory were not shown.

Next action: load this exact full cache with `title_features.cache.load_cache`
and current `input_hashes`, and print the compact summary in teaching note 19.
This verifies the consumer path and captures artifact identity before downstream
initialization/fusion implementation. No re-encoding, reinstall, unit-test or
smoke repetition is needed. No HSTU content fusion or recommendation gain has
been implemented or verified yet. Keep MiniOneRec deferred as agreed.


## Full title consumer PASS and downstream implementation (2026-10-23)

The user returned `Full title cache load: PASS` for the exact full build above:
shape (19221,384), float32, features SHA256
`af5c0482e9b37294e1e4817810d5e2b0b407c22759207078fd402f7ea0525b25`.
All frozen/repeat/batch-padding/padding-zero/unit-row/saved-output checks passed;
10 titles truncated, elapsed 37.32873547798954 seconds, peak allocated/reserved
515812864/828375040 bytes. This supersedes the pending consumer gate above.
Do not repeat installation, title encoding or the already passed feature tests.

New isolated `content_hstu/` implements complete title-init and title-fusion
train/evaluate/epoch-resume paths, paired configs and teaching note 20. Fixed PCA
uses distinct items in selected training histories only; all catalog titles are
transformed by that basis and scaled to 0.02*sqrt(D). Init replaces initial ID
weights; fusion preserves random ID initialization and adds 0.5 times a frozen
title table. Both histories and candidate scores use this representation, with
unchanged trainable parameter budget and raw scoring. Original src/controls,
title_features and existing configs remain unchanged to preserve old identities.

Nine local pure-Python/tiny-NumPy tests and syntax checks passed. Eight new
PyTorch model/runner tests are prepared but NOT run locally. Next action: run
all 17 focused content tests on server CPU, then both three-epoch CUDA smokes;
review results before the two full 30-epoch candidates. No recommendation gain
or GPU behavior has yet been verified for content HSTU. Formal selection remains
validation-only, and MiniOneRec stays deferred. Exact commands and limitations
are in docs/learning-20-title-hstu-controls.md.


## Content HSTU server tests and both smokes PASS (2026-10-23)

The user confirmed all 17 focused content tests passed. Title initialization
smoke returned PASS twice at runs/title-init-smoke-server/20261023T022111405200Z
and 20261023T022133518327Z, with matching printed metrics: losses
9.650886/7.882080/6.883350, probes 8.118129/6.831610/6.121463, validation NDCG@10
0.010190/0.007294/0.007940. Fixed fusion smoke returned PASS at
runs/title-fusion-smoke-server/20261023T022236721652Z, losses
9.891118/8.195031/7.155380, probes 8.621272/7.183420/6.410565, validation NDCG@10
0.001962/0.004034/0.006858. Both parameter counts match reference at 1273406.
No full resource reports were supplied for these runs.

This supersedes the pending test/smoke gate above. Next run title-init-train-server
and title-fusion-train-server sequentially, 30 epochs each from scratch, with
unchanged configurations. No cache rebuild, repeated tests/smokes or new setup
is needed. Review best-epoch overall/frequency/history validation metrics and
resources against raw ID HSTU before deciding the frozen test recipe. Do not use
smoke scores or already observed test results to tune new configurations.
Full-data title experiments and recommendation gains remain unverified.


## Both full title HSTU training runs PASS (2026-10-23)

The user returned all 30 epochs and PASS for title-init at
`runs/title-init-train-server/20261023T162529774092Z/results.json` and title-fusion
at `runs/title-fusion-train-server/20261023T165455784364Z/results.json`.
Both report 1273406 parameters. Printed best validation NDCG@10 occurs at epoch
30 for both: init 0.037931, fusion 0.037252, approximately +0.55%/-1.25% versus
raw ID HSTU 0.03772422803068947. Init first/final losses 8.897017/8.105081;
fusion 9.014198/8.115843. Treat these as rounded single-seed observations, not
significance or established title gains. Full precision, group metrics, actual
runtime/device and memory for these runs are not yet supplied.

Next action is read-only report comparison via scripts/compare_title_validation.py
with the three exact report paths in note 20. It validates report-level pairing
and earliest validation selection and prints slices/resources. Five synthetic
standard-library tests and CLI check passed locally; real report pairing awaits
server output. It does not load checkpoints, verify current files or authorize
test evaluation. Do not repeat finished training/tests or expand epochs/weights
before reviewing validation slices. Keep historical checkpoint-bound sources
unchanged and MiniOneRec deferred.


## Full title validation comparison and frozen candidate (2026-10-23)

The user returned report pairing PASS and exact best validation slices/resources.
Init NDCG@10=0.03793132350166986 (+0.54897%), R20=.1024; fusion
NDCG@10=.03725230719301553 (-1.25098%), R20=.10182. ID/init/fusion Top-20
hits are 5088/5120/5091 overall, 191/221/206 for frequency 6-20 and
4897/4899/4885 for 21+. Frequency 0 (844) and 1-5 (7264) stay at zero.
Init adds only 32 hits and does not solve sparse-item failure; do not claim
statistical significance or that all text methods fail. History-length results
are mixed. Full evidence is recorded in teaching note 20.

Reported runtimes: ID 1196.9340426961426s, init 1688.9285601947922s,
fusion 1253.4578634558711s. Peak allocated bytes 171903488/181745152/181745664;
reserved 243269632/251658240/251658240. All report A100 MIG 2g.20gb,
torch2.10+cu128, CUDA12.8, NumPy1.26.4, FP32/TF32off. These are whole-run
measurements, not isolated overhead or serving benchmarks. Both title runs use
clean commit 921b276f7b502e9f942de2f5c18614da99a1987c and identical PCA identity:
18490 fit items, 384->64, explained variance .5789973137664619. Variance is not
semantic-information retention. The user supplied actual array hashes in output.

Freeze title-init epoch30 best.pt under the predeclared validation NDCG@10 rule,
seed42 and unchanged raw-score configuration, for one test evaluation to close
this comparison. No claim of promotion to the main model or stable gain is made.
Do not test fusion or retune epochs/weights/alpha from test feedback. New isolated
scripts/evaluate_selected_title.py and configs/hstu-title-selected-test-server.json
verify report selection, current config/source/data and recorded best.pt hash
before invoking the existing evaluator, which checks cache/projection/checkpoint
identity and actual runtime. Twelve local standard-library checks and CLI/syntax
checks passed. Actual frozen admission and title-init test are pending server
execution. Do not repeat previous training/encoding/model tests. MiniOneRec stays
deferred. Current exact command is in note 20.


## Frozen title-init test PASS and experiment closure (2026-10-23)

The user returned Frozen title report/checkpoint preflight PASS (title-init,
epoch30 best.pt) and Title HSTU evaluate PASS. Test report:
`runs/title-init-train-server/20261023T184815307170Z/results.json`.
All 50000 users: NDCG@10=.025048004444575108, NDCG@20=.03109323246867899,
Recall@10=.04878, Recall@20=.0728. Versus raw ID, NDCG@10 is +1.29045%,
NDCG@20 +1.32683%, R10 -0.08193%, R20 +0.58027%. Top-20 hits increase
3619->3640, with frequency6-20 153->163 and21+ 3466->3477. Frequency0 (2226)
and1-5 (11557) remain zero. Top-10 hits decrease2441->2439. No significance,
all-metric improvement, or cold-start/long-tail solution is established.
History-length N10 is higher in every group, but R20 falls slightly in11-20;
longest group has only686 users. Full figures are recorded in teaching note20.
Actual test time/memory/runtime fields have not been supplied.

The one-seed title comparison is complete. Do not repeat encoding, tests,
training or test evaluation without a concrete failure. Do not tune title
weights, epochs or score correction from these test results. Preserve the
existing frozen baselines and title results. Possible next work is validation
candidate-hit complementarity and simple HSTU/ItemCF fusion, motivated by earlier
validation evidence, not the new test slices. This fusion is not implemented or
verified yet; it targets sparse observed items and cannot solve zero-interaction
cold start on its own. Keep content generalization, statistical robustness,
serving measurements and MiniOneRec separate from completed evidence.

## Research-first correction after title controls (2026-10-23)

The user requested an industrial-practice review before more experiments,
expressing concern about blind iteration. This supersedes automatic progression
to HSTU/ItemCF fusion. Primary-source research and implementation review are
documented in docs/learning-21-industrial-cold-start-and-tail-review.md.
Public methods distinguish content-to-preference mapping, training with missing
preference inputs, feature-use/ranking diagnostics, and online exploration.
Do not infer industry adoption rates or transferable lift from these examples.

Code inspection establishes that fixed V=E+0.5*S with unrestricted per-item E
does not expand the fixed-catalog lookup function class. It changes initialization,
optimization and regularization; it does not prove learned cancellation of S.
There is no learned shared content projection or item-ID dropout in the current
title controls. Full-catalog cross entropy includes zero-positive items as
non-target classes; they are not untouched embeddings. These facts identify
mechanisms to examine, not an established causal explanation of all tail failures.
All catalog items are scored, so upstream retrieval omission is not the current
failure. Do not conflate sampled-negative log-Q correction with our full-softmax
objective or post-hoc popularity correction.

Next planned artifact: a validation-only diagnostic using existing frozen
artifacts, comparing raw384/PCA64 title-mean scoring with one preregistered
within-frequency item/title permutation, target-rank distributions for existing
models, and same-K HSTU/ItemCF exclusive hits. Preserve filtering and identity
checks; reproduce raw baseline validation metrics. No diagnostic implementation
or new server verification exists yet. Do not start another 30-epoch run before
reviewing this evidence. Weak zero-shot content results do not rule out learned
content usefulness. Future shared mapping/ID-dropout or fusion changes are
conditional hypotheses, not guaranteed fixes. Keep all reported test recipes
frozen and disclose reuse of this benchmark; do not call it an untouched test.


## Mechanism audit implementation (2026-10-24)

The research-first validation audit is now implemented in diagnostics/ and
scripts/audit_recommendation_mechanisms.py, configured by
configs/recommendation-mechanism-audit-server.json. It authenticates all three
paired checkpoints and the historical ItemCF validation report, reproduces four
validation anchors, and computes raw384/PCA64 title-mean controls with one fixed
within-frequency permutation, exact target ranks/margins, and same-K exclusive
hits. It does not train, select a new model, or evaluate test. Model sources and
training configurations remain unchanged. Twelve tiny standard-library/NumPy
tests passed locally; two synthetic server CPU checks and real-data inference
remain pending. Exact commands and interpretation are in teaching note22.
Do not repeat completed training or declare a tail improvement from implementation.


## Legacy ItemCF report fix (2026-10-24)

The user reported all original 14 tests and three HSTU validation anchors PASS
on A100 MIG. The audit failed on a missing history-length axis in the historical
ItemCF report, not an observed metric mismatch. Only legacy ItemCF may omit this
axis: its frequency groups remain strictly checked, and newly computed history
metrics are labeled computed_without_historical_anchor. Modern reports still
check both axes. Fifteen lightweight local tests pass; the updated server checks
and real audit remain pending (17 focused tests total). Repeat validation
inference only; no retraining, title encoding or test evaluation is needed.
The failed run did not save per-user ranks, so inference must restart.


## Full mechanism audit server PASS (2026-10-24)

User returned 17 tests PASS (7.413s) and full audit PASS at
runs/recommendation-mechanism-audit/20261024T162518831454Z/results.json.
All three neural anchors and historical ItemCF frequency anchor reproduced.
A100 MIG, torch2.10/cu128, 50000 validation users. Raw HSTU tail median rank
10428.5, hits@100/1000=3/108. Raw384 title mean tail@20=225 vs permuted11;
PCA64=202 vs permuted3. Zero-frequency hits=26/23 vs controls0/2. This supports
usable title association, not significance or proven new-item temporal lift.
PCA improves deeper tail recall despite reducing Top20; it is not the sole
explanation for neural zero tail hits. ItemCF adds1674 exclusive hits, including
348 tail and1011 mid; union6762 is an oracle upper bound, not fusion performance.
Prioritize a separately specified learned shared content mapping hypothesis;
ID dropout is a separate ablation. No new mapper implemented or gain established.
Do not rerun the completed audit. Detailed evidence is appended to note22.


## Shared mapping implementation (2026-10-24)

User authorized the most reasonable next experiment after audit evidence.
Isolated mapped_hstu/ implements content-only HSTU with shared centered linear
384-to-64 mapping, initialized from training-item PCA, normalized and used on
both history and candidate sides. No independent trainable ID table survives.
Compare map-frozen vs map-learned; do not mix ID dropout, gating, ItemCF or score
correction. Fixed mapping trains the HSTU backbone; learned mapping also trains
24576 shared weights. Trainable parameter counts expected43262/67838 vs original
1273406, so no equal-parameter claim. Both start identically. Three local config
checks plus syntax/CLI pass; four server neural checks and paired 3-epoch smokes
are pending. Exact commands and gates in note23. Review smoke before the two
predeclared30-epoch validation-only runs; test evaluation is forbidden. Historical
checkpoint-bound sources/configs remain unchanged. No gain established yet.


## Frozen projection registration fix (2026-10-24)

The user returned six passing tests and one failing runner integration test.
The fixed projection was a requires_grad=False Parameter, while the shared
training routine deliberately requires a finite gradient for every named
parameter. It is now a registered buffer; the learned projection remains a
Parameter. Both use F.linear and save the same weight key in state_dict.
The original training routine and gradient guard remain unchanged. Tests now
explicitly check buffer/parameter registration and checkpoint inclusion.
Three local configuration tests and syntax checks pass; corrected neural tests
and real-data smokes await server verification. The shell && chain stopped at
tests, so neither real-data smoke ran. Pull and repeat the seven focused tests
followed by the two smoke commands above. No prior training rerun is required.


## Paired mapping smoke server PASS (2026-10-24)

The user returned all seven focused tests PASS (13.702s) and both three-epoch
smokes PASS at commit53a191b. Runtime: Python3.11.16, torch2.10.0+cu128,
CUDA12.8, A100-SXM4-80GB MIG2g.20gb,20937965568 visible bytes.
Frozen:43262 trainable parameters; loss9.738615->9.434431,
probe9.461222->9.072540, final validation NDCG@10=.010448.
Report:runs/map-frozen-smoke-server/20261024T165622195010Z/results.json.
Learned:67838 trainable parameters; loss9.679343->9.212684,
probe9.271473->8.798348, final validation NDCG@10=.009406
(best printed .009813 at epoch1).
Report:runs/map-learned-smoke-server/20261024T165639012746Z/results.json.
These are1024-user smokes, not evidence of a full-data improvement. Lower learned
training loss does not establish better validation ranking. Actual final resource
fields were not supplied. Both smoke gates passed; next execute the predeclared
paired30-epoch full-data validation-only recipes without changing hyperparameters.
Do not repeat smoke, cache preparation or old test evaluations. Compare each
model's earliest validation-selected best epoch, not simply its last epoch.


## Full paired mapping training PASS (2026-10-24)

User returned both30-epoch full-data training runs PASS on A100 MIG2g.20gb,
Python3.11.16/torch2.10.0+cu128. Frozen43262 parameters: best printed validation
NDCG@10=.012059 at epoch28; final loss9.286922. Report:
runs/map-frozen-train-server/20261024T165826137072Z/results.json.
Learned67838 parameters: best printed validation NDCG@10=.014608 at epoch28;
final loss8.956967. Report:
runs/map-learned-train-server/20261024T171919631080Z/results.json.
From rounded logs learned is about21.14% higher than frozen, but about61.28%
lower than raw ID (.0377242280). It is only about3.92% above direct384 title
mean (.0140572575), and that comparison changes encoder/dimension/training.
No robust architecture or end-to-end improvement is established. Exact best
selection, frequency/history slices, paired identities and resource fields have
not yet been returned. Next read these saved reports only; do not add epochs,
change tuning, or evaluate test. Tail and zero-frequency improvement is unknown
until selected validation slices are inspected. Retain ID HSTU as main model.


## Shared-map mechanism experiment closed (2026-10-24)

Exact paired identities and validation selections returned by user. Both epoch28:
frozen N10=.012059027869283721, learned=.01460829926623397. Top20 counts by
frequency0/1-5/6-20/21+/all: frozen16/106/459/1411/1992; learned9/97/485/2043/2634.
Of642 net extra hits,632 are in21+, while zero/tail decline7/9. No target tail
improvement; stop this recipe, no extra epochs or test evaluation. Raw384 title
mean225tail/26zero and PCA64 mean202/23 beat both on sparse targets. Keep ID HSTU
main model. Potential next work is bounded validation-only HSTU/ItemCF score
combination justified by measured exclusive hits, with content channel separate.
Do not promise union-oracle gains, route using true target frequency, or add more
neural mechanisms automatically. Full exact metrics/resources are in note23.

## Bounded hybrid validation implemented

scripts/evaluate_hybrid_validation.py evaluates fixed alpha0/.1/.25/.5 using
raw frozen HSTU cosine mapped to[0,1] and eligible ItemCF scores normalized by
user maximum. Zero CF falls back to raw HSTU. Shared admission and strict ID/CF
anchors apply; no neural training or test evaluation. Three tiny NumPy tests
and syntax checks passed locally; real server verification pending. See note24.
A candidate must not regress raw overall N10 and must have tail hits; selection
is only for review, not significance or proof of improvement. Compare against
stronger corrected HSTU too. Do not expand search automatically on failure.


## Hybrid validation server PASS and recipe frozen (2026-10-24)

Three tests and full validation PASS at runs/hybrid-validation/20261024T203229829396Z/results.json.
Predeclared selection chose alpha.25: N10=.044361011644391685, R20=.11482;
hits5741 vs raw5088, tail72 vs0, mid730 vs191, head4939 vs4897; zero remains0/844.
N10 +17.59% vs raw and +9.95% vs corrected HSTU. Tail remains below ItemCF348.
Alpha.5 has272 tail hits but regresses overall N10=.0360072778, so do not choose it.
Freeze full .25 recipe with raw HSTU, training-only100-neighbor CF, eligible-CF-max
normalization and no-evidence fallback. No added correction/title branch. Next
implement frozen admission and one test evaluation, not more validation search.
The frozen test entry point is now implemented in scripts/evaluate_selected_hybrid.py,
with validation-selection/source/data/checkpoint admission and alpha=.25 locked.
Seven lightweight local tests passed; server inference subsequently passed as below.
See docs/learning-25-frozen-hybrid-test.md. Single-seed selected validation result,
not significance, cold-start solution or independent confirmation. See note24.

## Frozen hybrid test server PASS (2026-10-24)

User returned seven tests OK (0.051s), admission PASS and test PASS at
runs/hybrid-selected-test/20261024T204032578400Z/results.json on A100 MIG 2g.20gb,
torch2.10.0+cu128/numpy1.26.4, FP32 with TF32 off. Alpha stays .25.
Test R10/N10/R20/N20=.0561/.02935906328561316/.08346/.03626219147039771.
Top20 hits total4173: zero0, frequency1-5=86,6-20=580,21+=3507.
N10 +18.72% vs raw HSTU and +8.31% vs corrected HSTU; R20 +15.31%/+10.22%.
Against corrected HSTU, tail/mid/head hit-count differences are +86/+330/-29.
Tail R20 remains only .0074413775 and zero-frequency remains0/2226.
Full server report not independently read locally; resources/history slices not shown.
Do not rerun training/test or tune from this result. Single seed and reused benchmark;
no significance, online-lift or solved-cold-start claims. Next proposed milestone is
exact-catalog serving with measured latency/resource cost; not implemented yet.
See note25 for full result and limitations.

## Exact hybrid serving implemented; server verification pending

Isolated serving/ package keeps all historical source inventories unchanged.
Frozen alpha=.25, authenticated startup, training-only ItemCF, exact full-catalog
ranking, complete visible-history filtering; requests use mapped IDs only.
Empty history uses training popularity, absent CF evidence uses raw HSTU;
invalid IDs are rejected. Loopback-only serial HTTP demo has no new dependencies.
Six lightweight local tests passed; one tiny neural CPU test awaits server.
Benchmark first compares original offline batches with request-at-a-time scores
(atol1e-6, exact Top20) on training histories, then measures raw/hybrid batch1,
concurrency1, warmup20,100requests, FP32 synchronized core latency excluding HTTP.
No repeated test-quality evaluation or new tuning. Server correctness and measured
latency remain unverified. See docs/learning-26-exact-serving.md for commands.

## Serving parity and core latency server PASS (2026-10-24)

User returned 7 tests OK (.175s), artifact admission PASS and benchmark PASS:
runs/hybrid-serving/20261024T204850491181Z/results.json. A100 MIG2g.20gb,
torch2.10.0+cu128/numpy1.26.4, Python3.11.16, FP32/TF32off,24CPUthreads,
19220candidates,100CFneighbors,1111056edges. Startup11.377485s.
100 training-history requests: max score error2.123415470123291e-7 (atol1e-6),
exact Top20 parity. Raw mean/p50/p95/p99 ms=3.354121/3.348491/3.438029/3.546075;
hybrid=3.704634/3.612403/3.932139/4.505386. Mean overhead .350514ms (+10.45%).
Both peak allocated/reserved49789952/62914560bytes; shared loaded artifacts,
not independent raw-only deployment memory. Batch1,concurrency1,warmup20 per mode,
100measurements per mode; excludes HTTP/startup. Serial reciprocal rates298.14/269.93
are not concurrent throughput. Full report not locally inspected; RSS not returned.
Do not repeat training, quality tests or benchmark without a reason. Next verify
the actual loopback HTTP endpoint using note26 commands; live HTTP remains pending.

## Live HTTP basic calls verified by user

Server /health returned ready with19220items. /recommend history=[1,2,3],k=10
returned10 unique positive IDs excluding all history, descending scores,
eligible_items19217,fallback=none,mode=hybrid. Both responses share model_id
17aaae44451f3d9fe5de7e4ee9a6e2e3a3ff981bf55374c4d836227f76d9abdb.
Basic real HTTP path now verified; do not repeat benchmark or training.
Synthetic request does not verify relevance. HTTP latency/concurrency unmeasured;
fallback/error paths have adapter tests but live network calls not yet returned.
