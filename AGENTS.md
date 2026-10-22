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
and the 12 focused CPU test completion have not been returned.

The user also reported the full 30-epoch mean-control train PASS at
`runs/mean-pooling/20261022T160521830353Z/results.json`.
Best validation NDCG@10 was approximately 0.029651 at epoch 19 (log-rounded),
versus raw HSTU 0.03772422803068947 at epoch 28. HSTU is approximately 27.23%
higher relative to Mean. Final Mean loss was 8.059710; final validation NDCG@10
was 0.029334. This is a single-seed matched-recipe comparison, not a pure
attention/order ablation or a significance claim. Full precision, slices, current
GPU, time/memory and focused CPU test completion remain unreported.

Next action: inspect saved best-validation slices and report identities using
teaching note 17, without retraining or testing. Keep raw/corrected HSTU frozen.
No mean-control speed claim is made. Do not select settings from test. Content
fusion, multi-seed robustness and real-data interrupted continuation remain
outside the completed verification record.
