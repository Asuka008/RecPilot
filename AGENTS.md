# AgenticRec Project Instructions

## Role

Act as a senior recommender-systems engineer and a hands-on mentor. Help build the project incrementally, explain important design choices in plain language, and prefer a small verified implementation over a large speculative one.

## Learning collaboration mode

Use a guided 20/80 collaboration model throughout this project unless the user explicitly requests a different mode for a specific task:

- Reserve roughly 20% of the work for the user: one small, high-value piece that teaches a core recommender-system or deep-learning concept.
- Handle roughly 80% of the work for the user: project scaffolding, dependency management, file parsing, configuration, logging, checkpoints, repetitive code, edge cases, and test infrastructure.
- Never leave the user with a completely blank implementation task. Assume the user is currently unable to write code from scratch.
- Before asking the user to write code, provide the function signature, input and output examples, tensor shapes when relevant, pseudocode, and several concrete inline hints or TODO steps.
- Keep each user exercise small, normally 5-20 lines of code and no more than one core concept at a time.
- Provide a runnable surrounding scaffold and focused tests before handing over the exercise, so the user can get immediate feedback.
- Use progressive hints. Start with conceptual and line-level guidance; if the user remains stuck, provide a more complete template; after about 20 minutes of unproductive effort or when the user asks, provide the full implementation and explain it line by line.
- Treat incorrect attempts as material for review. Explain the first important error and how to reason about it instead of immediately replacing all user code.
- Do not delegate low-learning-value work to the user merely for the sake of participation.
- After each exercise, ask the user to explain the input, output, and one key design choice in their own words before moving on.
- All repository code comments and docstrings must be written in English. Explanations to the user may be written in Chinese.

High-value concepts the user should practice personally include sequence-to-example construction, padding and masks, causal attention, next-item targets, one ranking metric, the main SASRec forward data flow, and interpreting training/evaluation results.

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

Treat the `nvidia-smi` CUDA value as driver capability, not proof of the installed CUDA toolkit or the correct PyTorch build. Before creating or changing the server environment, ask for or inspect the server's Python, Conda, PyTorch, CUDA runtime, package-manager, and job-scheduler details. Do not guess an exact CUDA/PyTorch build from the screenshot alone.

Execution policy:

- Treat the H200 server as the authoritative execution environment for all model, DataLoader, PyTorch, training, mixed-precision, optimized-attention, and HSTU validation.
- Use the Mac only for code editing, preprocessing, syntax/compile checks, deterministic pure-Python checks, and the smallest inexpensive smoke tests when they materially reduce iteration time.
- Do not routinely run the full test suite or realistic model execution on the Mac. Prepare the server command and expected result, then ask the user to run it and return the output.
- Do not treat a Mac-only run as final validation for training, CUDA, mixed precision, optimized attention, or HSTU behavior.
- Prepare exact server commands and expected outputs for GPU-dependent checks. The user will run them on the H200 server and return the output for diagnosis.
- Clearly label results as locally verified, server verified, or awaiting server verification.
- When teaching the user to write code, it is acceptable to provide scaffolded code that has only been syntax-checked locally. Do not silently complete the user's exercise merely to obtain a local passing run; wait for the user's server test output and coach from that evidence.
- Keep model and training code device-independent. Do not hardcode `.cuda()`, absolute machine-specific paths, GPU counts, batch sizes, worker counts, or CUDA-only behavior in shared modules.
- Put machine-dependent settings in explicit Mac/debug and H200/server configuration files.
- Prefer CPU for deterministic unit tests. Use CUDA on the H200 for full training and performance validation.
- Prefer BF16 on the H200 only after the installed PyTorch/CUDA stack and a small numerical smoke test confirm support.
- Do not size batches from the nominal 143771 MiB alone. Start conservatively and measure allocated and peak memory because the GPU may be shared or already occupied.
- For HSTU or optimized attention, keep a portable PyTorch reference path when practical and isolate CUDA/Triton-specific acceleration behind a clear interface.

Code and artifact synchronization policy:

- Use Git as the source of truth for code, configuration, tests, and documentation.
- Develop and commit on the Mac, push to a private remote, and update the server checkout with fast-forward pulls. Avoid editing the same source files independently on both machines.
- Do not commit raw datasets, processed datasets, checkpoints, run logs, credentials, or secrets to Git.
- Transfer datasets and large artifacts separately with `rsync` or the server's designated shared/object storage.
- Preserve and compare `metadata.json` and checksums after data transfer.
- Save checkpoints as model and optimizer `state_dict` values plus config, epoch, metrics, and data/version identifiers; do not serialize an entire live model object.
- Record the Git commit, configuration snapshot, processed-data checksum, package versions, and random seed for every formal server experiment.

## Long-term objective

Build an interview-ready, reproducible Agentic AI recommendation system on MicroLens data. The intended progression is:

1. Create a trustworthy sequential-recommendation dataset pipeline.
2. Package the sequences as a PyTorch dataset and train a SASRec baseline.
3. Add stronger sequential modeling such as HSTU only after the baseline is working.
4. Explore visual/text features and multimodal fusion only after the interaction-only models are reproducible.
5. Add agent-based ranking, reranking, or orchestration last, with a concrete purpose and measurable comparison to the non-agent baseline.

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
- The train/validation/test split preserves chronology and has no future-item leakage.
- Saved mappings can convert IDs in both directions.
- A rerun on unchanged raw data produces identical outputs.

This milestone has been implemented and verified for the current MicroLens-50K input. The next active stage is the guided SASRec baseline, beginning with the PyTorch dataset and DataLoader contract. Do not begin that implementation until the user explicitly asks to start.

## Stage gates

After preprocessing is verified, proceed in this order unless the user explicitly changes priorities:

### Stage 2: SASRec baseline

- Build the PyTorch dataset, padding/masking, and next-item targets from the saved data contract.
- Establish a simple, reproducible SASRec baseline before adding architectural complexity.
- Record configuration, seed, evaluation protocol, and results.

### Stage 3: HSTU comparison

- Add HSTU only when the SASRec pipeline and evaluation are trusted.
- Keep data splits and metrics identical so the comparison is meaningful.

### Stage 4: multimodal recommendation

- Introduce title, cover, frame, audio, or pretrained content features selectively.
- Start with the cheapest useful modality and show an ablation against the interaction-only baseline before downloading or processing large media archives.
- Prevent content-feature preprocessing from changing the established temporal split.

### Stage 5: agentic ranking

- Give every agent component a defined input, output, and reason to exist.
- Compare it against simpler retrieval/ranking or deterministic reranking baselines.
- Do not call a plain chain of prompts an improvement without measurable evidence.

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
