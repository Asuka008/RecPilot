# AgenticRec

A reproducible HSTU-based video recommendation project targeting MicroLens-50K.
The project starts with chronological interaction data, shared ranking metrics,
and simple baselines, then verifies HSTU against a pinned official implementation
before training it on MicroLens.

**Current milestone:** the official HSTU recipe completed 101 epochs on
MovieLens-1M, with final metrics close to the upstream reference. MicroLens
Popularity and ItemCF results are available. The MicroLens HSTU training and
evaluation integration passed 33 focused server tests and a three-epoch real-data
smoke run. The first full-data HSTU candidate completed training and frozen
test evaluation. A validation-only score correction selected alpha=0.5 and
improved validation NDCG@10 by 6.95%. Its fixed test evaluation passed with
NDCG@10 +9.62% over raw HSTU. A causal prefix-mean neural control is implemented
and passed server smoke and full 30-epoch training (best validation NDCG@10 about 0.029651). Low-frequency matching remains unresolved.

## Verified progress

Status as of **2026-10-22**. Completed server results are based on outputs
returned by the project owner. MicroLens integration tests and smoke training
passed, and the first full-data training and test evaluations returned PASS.

| Milestone | Status and evidence |
|---|---|
| MicroLens preprocessing | Deterministic sequences, reversible item mappings, and chronological leave-one-out splits verified locally; transferred file checksums matched |
| Dataset / DataLoader | Eight synthetic CPU tests passed on the server |
| Shared evaluation, Popularity, ItemCF | Eighteen combined Dataset/baseline tests passed; full MicroLens validation and test evaluation completed |
| Official HSTU environment | Pinned source and dependencies verified; FBGEMM operations and official block checks passed on CUDA |
| Portable dense HSTU block | Twelve server cases passed output and gradient comparisons against official blocks, covering position-only and time/position bias |
| Pure-ID HSTU training components | Four-user synthetic fit reduced fixed-task loss by 99.3%; checkpoint output, RNG, and next-step continuation checks passed |
| Official MovieLens-1M recipe | Batch 32/128 smoke checks and batch 128, 101-epoch training completed on the server |
| MicroLens HSTU baseline | Training and frozen test evaluation passed; test Recall@10=0.04882, NDCG@10=0.024729; validation-selected checkpoint from the first 30-epoch candidate |
| Prefix-mean neural control | Implemented with matched settings, shared scoring/objective and isolated source identity; 5 config tests passed locally; server smoke and full 30-epoch training PASS; best validation NDCG@10 about 0.029651; Mean beats raw HSTU on lower-frequency validation groups; paired identity audit and frozen test PASS (NDCG@10=0.017816); all 18 focused server tests confirmed OK by the user |
| Frozen title features | Alignment audit and 11 lightweight tests passed locally; pinned MiniLM cache pipeline implemented; 14 server tests, setup/preflight/audit and 64-title CUDA smoke and full cache build PASS; consumer load verification and fusion pending |
| Recommendation service | Planned; no measured results yet |

The portable implementation is in [src/models/hstu.py](src/models/hstu.py) and
[src/models/id_hstu.py](src/models/id_hstu.py). Block comparisons and synthetic
training checks validate specific implementation behavior; they do not establish
MicroLens recommendation quality or whole-model equivalence to the official recipe.

## Results

### MicroLens-50K baselines

Training uses training histories only. Evaluation ranks the full mapped catalog,
excludes padding and each user's visible history, and breaks score ties by
ascending mapped item ID. Each split contains 50,000 user targets.

| Split | Model | Recall@10 | NDCG@10 | Recall@20 | NDCG@20 |
|---|---|---:|---:|---:|---:|
| Validation | Popularity | 0.00424 | 0.002036 | 0.00718 | 0.002790 |
| Validation | ItemCF | 0.04804 | 0.024495 | 0.07234 | 0.030645 |
| Test | Popularity | 0.00244 | 0.001254 | 0.00378 | 0.001594 |
| Test | ItemCF | 0.03534 | 0.017473 | 0.05760 | 0.023088 |
| Test | PrefixMean | 0.03634 | 0.017816 | 0.05982 | 0.023729 |
| Test | Pure-ID HSTU | 0.04882 | 0.024729 | 0.07238 | 0.030686 |
| Test | HSTU + fixed frequency correction | 0.05174 | 0.027107 | 0.07572 | 0.033161 |

ItemCF uses sparse cosine similarity and 100 neighbors per item. Targets whose
items have zero training-history occurrences remain in the evaluation and are
reported as a separate frequency group. Settings are in
[configs/baselines-server.json](configs/baselines-server.json).

### First full-data HSTU candidate on MicroLens

The server reported `MicroLens HSTU train: PASS` after 30 epochs, seed 42,
using `configs/hstu-microlens-train-server.json`. The highest printed validation
NDCG@10 is **0.037724 at epoch 28**, compared with ItemCF **0.024495**
(approximately +54.0% relative, using rounded log values). The final epoch is
0.037618; evaluation must use validation-selected `best.pt`.

This is one candidate and one seed. The returned artifact summary confirms
best epoch 28, validation NDCG@10=0.03772422803068947 and Recall@20=0.10176.
Validation Recall@20 is 0 for frequency 1-5 and 0.01010315 for frequency 6-20,
versus ItemCF 0.04790749 and 0.06035440. The 21+ group reaches 0.21303345,
versus ItemCF 0.09257406. Both reports' data hashes and group counts match.
Full training took 1196.934s on A100 MIG 2g.20gb with peak PyTorch allocated/
reserved memory 0.16010/0.22656 GiB. The audit also reproduced original validation
Recall@10=0.07112 and NDCG@20=0.04544371132907886.
The frozen test evaluation passed for all 50,000 users: Recall@10=0.04882,
NDCG@10=0.02472889109808947, Recall@20=0.07238,
NDCG@20=0.030686079403136588. Relative to fixed ItemCF, these increase by
38.14%, 41.53%, 25.66%, and 32.91%, respectively, for this seed and protocol.
This is an offline result; no online gain or multi-seed robustness is established.

Test targets with training-history frequency 0 or 1-5 have zero Top-20 hits
(2,226 and 11,557 users). Of 3,619 Top-20 hits, 3,466 belong to frequency 21+.
The same low/mid-frequency regression is now confirmed on **validation** against ItemCF.
A local data-only audit also distinguishes history occurrences from positive
shifted labels: 482 history-observed items have no retained positive target at
L=32. This alone cannot explain all weak tail results. See
[results and diagnosis note 13](docs/learning-13-microlens-baseline-results.md).

Training run identity:
`runs/hstu-microlens/20261022T023420952080Z/results.json`.
Test run: `runs/hstu-microlens/20261022T030714112850Z/results.json`.

### Frozen score correction (validation and test passed)

The predeclared alpha=[0,0.25,0.5,1] audit selected **alpha=0.5** on validation
NDCG@10: **0.040344999048154036**, versus raw HSTU **0.03772422803068947**
(+6.95%). Validation Recall@20 is 0.10616 versus 0.10176. The model weights are
unchanged; the correction subtracts a fixed training-positive-frequency offset.
The 1-5 target-frequency group still has zero hits, so this does not solve tail
matching. The fixed test evaluation passed for 50,000 users:
Recall@10=0.05174, NDCG@10=0.027107023028108444, Recall@20=0.07572,
NDCG@20=0.03316138641732385. Relative to raw HSTU, gains are 5.98%, 9.62%,
4.61%, and 8.07%, respectively. This changes scoring only, not model weights.
Zero- and 1-5-frequency targets still have zero Top-20 hits. The 167 extra
Top-20 hits comprise 97 in frequency 6-20 and 70 in 21+. Test artifact:
`runs/hstu-score-selected-test/20261022T151722543561Z/results.json`.
See [corrected results note 16](docs/learning-16-corrected-test-results.md).

### Official HSTU on MovieLens-1M

Upstream: Meta's [generative-recommenders](https://github.com/meta-recsys/generative-recommenders/tree/10c4950343b98bacec9ffd5b8cd879f811fe2326),
pinned at `10c4950343b98bacec9ffd5b8cd879f811fe2326`.
The experiment uses `configs/ml-1m/hstu-sampled-softmax-n128-final.gin`:
2 blocks, 1 head, embedding/QK/V dimensions 50, maximum history 200,
128 sampled negatives, and batch 128 for 101 epochs.

The observed column reports the preselected final epoch, index 100.
The reference column is the **HSTU** row in the
[pinned upstream README](https://github.com/meta-recsys/generative-recommenders/blob/10c4950343b98bacec9ffd5b8cd879f811fe2326/README.md),
whose table is marked verified as of 2024-04-15.

| Metric | Observed | Upstream reference | Difference |
|---|---:|---:|---:|
| HR@10 | 0.305960 | 0.3097 | -0.003740 |
| NDCG@10 | 0.169352 | 0.1720 | -0.002648 |
| HR@50 | 0.579139 | 0.5754 | +0.003739 |
| NDCG@50 | 0.229818 | 0.2307 | -0.000882 |
| HR@200 | 0.771854 | 0.7716 | +0.000254 |
| NDCG@200 | 0.259080 | 0.2606 | -0.001520 |

All six absolute differences are below 0.00374. They fall within the predeclared
0.02 absolute engineering comparison band. This is one seeded run, not a
statistical equivalence claim. The upstream table does not specify its checkpoint
selection rule here; this project uses the final epoch without selecting the
highest held-out score.

The wrapper preserves official model, loss, split, and evaluation code. Adaptations
include the pinned current dependency stack, one CUDA rank, two DataLoader workers
with prefetch factor two, explicit Python/NumPy/PyTorch seeds, and isolated output
directories. These differences should accompany any reproduction claim.

The ML-1M and MicroLens results use different datasets and protocols and should
not be compared as competing model scores. In particular, the official recipe
uses its temporal bias and next-event timestamps; the MicroLens integration
uses position-only bias and full-catalog cross entropy, rather than the official
sampled-softmax objective. Its smoke, full-data training and frozen test evaluations passed.

## Data contract

The current MicroLens snapshot contains:

- **50,000 users, 19,220 mapped items, 359,708 interactions**.
- Full sequence length: mean 7.19, median 6, P99 24, maximum 218.
- 730 mapped items never occur in training histories; they account for 844
  validation targets and 2,226 test targets (4.452% of test users).

Required input: `data/raw/MicroLens-50k_pairs.csv`, with columns `user`, `item`,
and `timestamp`. Raw data is immutable and supplied separately from Git.
Titles are optional metadata; no text features are used in the current experiments.

```bash
python -m src.preprocess --min-seq-len 5
```

The pipeline validates interactions, sorts each user's history by time, remaps
item IDs to contiguous positive integers, and reserves ID 0 for padding. For each
retained user, the last item is test, the penultimate item is validation, and the
preceding items are training history. Generated files:

| File under `data/processed/` | Contents |
|---|---|
| `sequences.jsonl` | One record per user: `user_id`, `train`, `val`, `test` |
| `item_mapping.json` | Forward and reverse item-ID mappings |
| `metadata.json` | Preprocessing configuration, statistics, and source checksum |

The Dataset creates left-padded shifted next-item training pairs. Validation sees
training history; test sees training history plus the validation item. Padding and
held-out targets are masked appropriately. This preserves chronology within each
user; it does not establish a global-time cutoff across users.

## Execution environments

**Model, DataLoader, PyTorch, CUDA, and training verification run on the server.**
The Mac is used for editing, preprocessing, syntax checks, and lightweight
pure-Python checks. Do not download public datasets or launch model experiments
on the Mac as part of the server workflow.

The latest verified allocation was **NVIDIA A100-SXM4-80GB MIG 2g.20gb**, exposing
19.50 GiB, rather than the originally intended H200. Inspect `nvidia-smi` before
running; environment names do not identify the assigned hardware.

The HSTU reference environment uses Python 3.11.16, PyTorch 2.10.0+cu128,
FBGEMM-GPU 1.5.0, and FP32 with TF32 for the official recipe. Dependency specs are
[environments/hstu-reference.yml](environments/hstu-reference.yml) and
[environments/hstu-reference-requirements.txt](environments/hstu-reference-requirements.txt).

The formal ML-1M run reported peak PyTorch **allocated/reserved memory of
2.686/18.770 GiB**. Allocated measures live tensor storage; reserved includes
allocator cache. These are separate peaks and exclude some external CUDA/NCCL
allocations, so the run should not be described as using only 2.7 GiB of GPU memory.

For a new server environment, use
[scripts/prepare_hstu_reference.py](scripts/prepare_hstu_reference.py) from a
Python 3.11 baseline environment. Its `inspect`, `plan`, `install`, and `source`
actions separate hardware inspection, dependency planning, installation, and
pinned source retrieval. Conda prefixes and the upstream checkout stay outside
the repository. Review the generated dependency plan before installation.

## Run on the configured server

Run commands from the repository root. The examples assume the existing prefix
environments and transferred MicroLens files are available.

### MicroLens baselines

```bash
conda activate "$HOME/conda-envs/agenticrec-h200"
python -m unittest tests.test_dataset tests.test_baselines -v
python -m src.run_baselines --config configs/baselines-server.json --split val
```

Inspect validation results and freeze settings before evaluating test:

```bash
python -m src.run_baselines --config configs/baselines-server.json --split test
```

### HSTU implementation checks

```bash
conda activate "$HOME/conda-envs/agenticrec-hstu-reference"
python -m src.check_hstu_reference \
  --config configs/hstu-dense-comparison-server.json --compare-dense
python -m src.run_hstu_toy --config configs/hstu-toy-fit-server.json
```

The source checkout defaults to
`~/upstream/generative-recommenders-reference`; retrieve and verify it with
`python scripts/prepare_hstu_reference.py source` before these checks.

### MicroLens HSTU integration gate (passed)

Synchronize the code through Git first, then use the existing reference environment:

```bash
conda activate "$HOME/conda-envs/agenticrec-hstu-reference"
nvidia-smi
python -m unittest tests.test_dataset tests.test_baselines tests.test_hstu_training tests.test_microlens_config tests.test_microlens_training -v
python -m src.run_hstu_microlens train --config configs/hstu-microlens-smoke-server.json
```

Reported server result: **33 tests passed in 11.080s** and
`MicroLens HSTU train: PASS`. Training loss across the three epochs was
9.946178, 8.179191, 7.135582; fixed-batch probe loss was 8.586332, 7.200485,
6.391037; validation NDCG@10 was 0.001397, 0.004264, 0.005421.
Result: `runs/hstu-microlens-smoke/20261022T022602941154Z/results.json`.
Returned report: PyTorch 2.10.0+cu128, CUDA runtime 12.8, A100-SXM4-80GB
MIG 2g.20gb with 20,937,965,568 visible bytes. Peak PyTorch allocated/reserved
memory was **0.154/0.193 GiB**; total elapsed time was **17.091s**. Parameter
updates, exact checkpoint outputs, reduced probe loss and unchanged data checksums
were confirmed. These are smoke-run measurements, not full-data or serving results.

The smoke run uses 1,024 seeded training users, 1,024 seeded validation users,
the full item catalog, and three FP32 epochs. Expect `MicroLens HSTU train: PASS`,
finite training/validation values, a reduced fixed training-only probe loss,
updated ID/HSTU weights, exact restored checkpoint outputs, and memory/provenance
reports. Smoke metrics are not full-data quality results.

The entry point supports epoch-boundary continuation with `--resume PATH/last.pt`;
keep its original sibling `best.pt`. Every continuation writes a new run directory.
The first full-data candidate retains train/eval batch 32/64, runs 30 epochs
from fresh initialization with seed 42, and uses all 50,000 users per split.
Training selects checkpoints by validation NDCG@10 only. Explicit checkpoint
evaluation uses `evaluate --config CONFIG --checkpoint PATH/best.pt --split val`
(or `test` only after settings are frozen, using a full-data configuration).
For a fresh reproduction, start it on the server after pulling this version and checking `nvidia-smi` (the first run is already complete):

```bash
python -m src.run_hstu_microlens train --config configs/hstu-microlens-train-server.json
```

See [full-training note 12](docs/learning-12-microlens-full-training.md) for
fixed settings, completion checks and continuation. See
[teaching note 11](docs/learning-11-microlens-hstu.md) for shapes, protocol,
continuation checks and exact server commands.

### Official MovieLens-1M experiment

The preparation action downloads **only ML-1M on the Linux server**, calls the
official ML-1M preprocessor, and records data checksums. Existing prepared data
is verified and reused. No MicroLens data is changed.

```bash
conda activate "$HOME/conda-envs/agenticrec-hstu-reference"
python -m src.run_hstu_public prepare --config configs/hstu-ml1m-smoke-server.json
python -m src.run_hstu_public smoke --config configs/hstu-ml1m-b128-smoke-server.json
```

After the smoke check passes, start a fresh 101-epoch experiment:

```bash
python -m unittest tests.test_hstu_public_config tests.test_hstu_public_training -v
nvidia-smi
python -m src.run_hstu_public train --config configs/hstu-ml1m-train-server.json
```

`train: PASS` means the training and artifact checks succeeded. The separate
`WITHIN_TOLERANCE` / `REVIEW_REQUIRED` result describes the final metric comparison.
Keep the foreground session alive until completion; this official trainer wrapper
currently has no interrupted-run resume entry point.

## Experiment artifacts and reproducibility

Runs use separate UTC timestamp directories under `runs/`, excluded from Git.
A formal ML-1M run records:

- Git commit and dirty state, configuration and effective Gin bindings, random
  seed, package versions, upstream source identity, and data checksums.
- Actual GPU, precision, peak allocated/reserved memory, elapsed time, console
  logs, and TensorBoard scalars.
- All 101 epochs' six main ranking metrics, final metrics, and reference comparison.
- A final checkpoint with model/optimizer `state_dict` and epoch, plus a metadata
  sidecar linking its checksum, configuration, metrics, and experiment identity.

The reported server run identifiers are:

| Experiment | Directory under `runs/` |
|---|---|
| MicroLens validation baselines | `baselines/20261018T201816099815Z/` |
| MicroLens test baselines | `baselines/20261018T202845819876Z/` |
| Official ML-1M, 101 epochs | `hstu-ml1m-reference/20261020T033447917013Z/` |

These artifacts live on the server and are not included in the Git repository.
Raw/processed datasets, physical environments, logs, checkpoints, and credentials
must be transferred or managed separately.

## Repository layout

```text
configs/          Experiment settings and pinned reference manifests
src/data/         Sequential Dataset and DataLoader
src/models/       Portable dense HSTU block and pure-ID stack
controls/         Isolated prefix-mean neural control and matched experiment runner
title_features/   Frozen title alignment, encoding and verified cache loading
src/preprocess.py MicroLens preprocessing
src/evaluation.py Shared full-catalog ranking metrics
src/baselines.py  Popularity and sparse cosine ItemCF
src/run_*.py      Baseline, synthetic HSTU, and official ML-1M entry points
scripts/          Reference environment and source preparation
tests/            Focused configuration, data, evaluation, and model checks
environments/     Environment specifications
docs/             Teaching notes and workflow documentation
data/             Raw and processed data, supplied separately
runs/             Generated experiment artifacts, ignored by Git
```

## Next milestone

The 33 focused CPU tests (including synthetic epoch-boundary continuation) and
real-data smoke run passed on the server. The resource report has been reviewed.
The first full-data 30-epoch candidate and frozen test evaluation also passed.
The validation-selected alpha=0.5 recipe also passed its fixed test evaluation.
Keep both raw and corrected results frozen. The causal prefix-mean control is now implemented with matched configuration,
initial ID lookup, candidate scoring, objective and training budget. Each training
position uses only its visible prefix; no score correction is applied. It lives
in controls/, leaving the historical shared src identity unchanged.

HSTU wins in all four validation history-length groups, with larger relative gains
for longer histories. Mean still wins in the two observed lower-frequency target groups.
The frozen Mean test wrapper verifies paired artifacts before calling the existing
evaluator; see [note 18](docs/learning-18-frozen-mean-test.md). Six admission tests
passed locally. The server preflight and frozen Mean test now both returned PASS;
The user confirmed all 18 focused server tests OK. No repeated training or test
is needed. Raw HSTU test NDCG@10 is 38.80% higher than Mean, while Mean still
wins on the two lower observed target-frequency groups; see note 18 for exact results.

Three server smoke epochs returned PASS (validation NDCG@10 0.000836 to
0.006182). Mean/reference parameter counts were 1,230,144 / 1,273,406.
All 18 focused server tests were confirmed OK by the user; Mean resources and slices were returned.
full-data training has since returned PASS, with best validation NDCG@10 about
0.029651 at epoch 19. Mean is weaker overall but better on frequency 1–5 and 6–20
validation targets. Its run took 762.42 seconds on A100 MIG, versus 1196.93 seconds
for HSTU; these are whole-run timings, not serving benchmarks. Inspect paired reports using
[prefix-mean note 17](docs/learning-17-prefix-mean-control.md).
Real-data interrupted training continuation and multi-seed robustness remain
unreported. Corrected-test resource fields have not yet been returned.

Text/content features must earn their cost in controlled experiments. A full
SASRec comparison is optional; no superiority over SASRec is claimed. Agents are
outside the current scope. A recommendation service and serving performance
measurements remain future work; offline ranking results do not establish online
CTR, watch-time lift, or production deployment.


### Current experiment: title-aware HSTU controls

Title preparation and full consumer loading are server verified: 19,220 items,
FP32 `[19221,384]` cache, 10 truncated titles, 37.33 seconds for full encoding.
The frozen cache is `runs/title-features-minilm/20261023T013804609858Z`; no setup
or encoding repetition is needed. See [note 19](docs/learning-19-frozen-title-features.md).

Two complete isolated controls now use that cache: title initialization and
fixed additive ID/title fusion. Both share the original HSTU backbone, input/output
lookup, training budget and raw scoring protocol. A frozen training-item PCA
maps 384 to 64 dimensions without adding trainable parameters. Historical model
sources and configurations are unchanged. Stable gains remain unverified.

Nine lightweight checks passed locally; all 17 focused server tests and both
CUDA smokes subsequently passed. Both full 30-epoch title runs also returned PASS.
Full report pairing subsequently passed. Initialization validation NDCG@10 is
0.0379313235 (+0.549% vs raw ID); fusion is 0.0372523072 (-1.251%). Initialization
adds 32 Top-20 hits, 30 in frequency 6–20 and two in 21+; frequency 0 and 1–5
remain zero for all three models. A stable content gain or tail solution is not
established. Recorded run times are 1196.93/1688.93/1253.46 seconds for ID/init/
fusion on A100 MIG; these are not isolated model or serving benchmarks.

The validation-selected initialization epoch30 subsequently passed frozen
checkpoint admission and test evaluation. Test NDCG@10=0.025048004444575108
(+1.29% vs raw ID), Recall@20=.0728 (+21 hits), while Recall@10=.04878 is slightly
lower. Frequency 0 and 1–5 still have no hits (13,783 test targets combined).
The title comparison is complete for this seed; a stable gain or sparse-item
solution is not established. Do not repeat training/test or tune from these test
results. Test resource fields and multi-seed robustness remain unreported.

See [note 20](docs/learning-20-title-hstu-controls.md) for full evidence and
reproduction commands. Following the user's research-first request, automatic
progression to HSTU/ItemCF fusion is paused. The
[industrial cold-start review](docs/learning-21-industrial-cold-start-and-tail-review.md)
compares primary sources with the actual implementation: fixed additive titles
do not provide a learned shared content mapper or missing-ID training, and all
catalog items already receive scores. The next planned artifact is a validation-only
audit of title signal, target ranks and collaborative-hit complementarity, before
choosing another training change. This audit is not implemented or server verified.
Historical checkpoints and completed test recipes remain frozen; MiniOneRec stays deferred.
