# Experiment configurations

Keep portable model and experiment settings in this directory. Machine-local
paths and credentials must stay in ignored `*.local.yaml` files or environment
variables.

`baselines-server.json` runs CPU Popularity and ItemCF on the full processed
catalog. It deliberately has no sampled candidate or user limit. Execute it on
the server, not as a Mac smoke test. Paths resolve from the repository root;
`--data-dir` and `--output-dir` can override them without editing shared config.
Use `python -m unittest tests.test_baselines -v` for tiny synthetic local checks.

MicroLens HSTU configurations share the same Dataset, portable IDHSTU and exact
ranking protocol. Catalog size comes from `item_mapping.json`, not model config.

| Configuration | Purpose |
|---|---|
| `hstu-microlens-smoke-server.json` | Next server gate: FP32, batch 32, 3 epochs, 1,024 seeded training and validation users each |
| `hstu-microlens-train-server.json` | First full-data candidate after passed smoke/resource review: 30 epochs, fresh seed 42, validation NDCG@10 selection |
| `hstu-microlens-debug.json` | Optional tiny CPU debugging on 16 real-data users; not a benchmark |

The initial training objective is full-catalog cross entropy at observed shifted
positions, with tied normalized candidate embeddings and temperature 0.2. This
differs from the official ML-1M sampled-softmax recipe. Evaluation still excludes
padding and full visible histories and uses ascending ID ties. Smoke targets and
full-catalog classes remain present even if their items have no positive training
occurrences. A smoke run's frequency groups use its selected training users only.

Run `python -m src.run_hstu_microlens train --config CONFIG` from the repository.
Continue at completed epoch boundaries with the same config and
`--resume RUN/last.pt`; keep the sibling `best.pt`. `--stop-after-epoch 1` creates
an intentional partial run without changing the configured training horizon.
Evaluate explicitly with `evaluate --config CONFIG --checkpoint RUN/best.pt
--split val`. Full configs require all users, and smoke configs reject test
evaluation. See `docs/learning-11-microlens-hstu.md` for the server workflow.


`hstu-score-audit-server.json` references the completed full-data run and fixes
validation-only alpha=[0,0.25,0.5,1], smoothing=1. It applies frequency correction
using retained positive training labels, keeps all evaluation groups unchanged,
and requires the zero-correction anchor to reproduce the frozen validation run.
Run `python scripts/diagnose_hstu_scores.py --config configs/hstu-score-audit-server.json`.
The validation grid selected alpha=0.5, and its frozen test subsequently passed with NDCG@10 +9.62% over raw HSTU. Do not use test data to select alpha.


`hstu-score-selected-test-server.json` names the completed validation report and
an output directory only. Its evaluator verifies the saved validation winner and
runs that one correction recipe on test; it rejects alpha overrides, failed
validation status, or changed scoring/model/checkpoint/data identities.
See `docs/learning-15-selected-score-test.md` for the server commands.


Mean-control configs (`mean-pooling-debug.json`, `mean-pooling-smoke-server.json`,
`mean-pooling-train-server.json`) name the matching HSTU reference config. Every
original experiment field except output_dir must match; unknown fields or alpha
corrections are rejected. The model fields specify the frozen reference
initializer, not attention layers in the mean control. The only trainable tensor
is the shared ID table. Start with the three-epoch server smoke, then review the
full-data 30-epoch candidate. Both use raw cosine scoring and validation NDCG@10
selection; see teaching note 17. Run with `python -m controls.run_mean_pooling`.


`title-features-minilm-server.json` is a feature-preparation configuration, not
an HSTU training recipe. It pins the encoder repository/revision, transformers
version, 384-dimensional FP32 output, token truncation, batch size and device.
Use `python scripts/prepare_title_features.py preflight` before environment
changes. Then audit/smoke/build produce independent reports under runs/; the
64-item smoke artifact is not a valid full cache. See teaching note 19.


Title HSTU configs (`title-init-{debug,smoke-server,train-server}.json` and
`title-fusion-{debug,smoke-server,train-server}.json`) pin the verified full
feature cache and its SHA256. All original experiment fields except output_dir
must match their named HSTU reference. `train-item-pca-v1` is fitted on distinct
items in selected training histories; initialization uses fusion_weight=0 and
fixed fusion uses 0.5. Both use the same representation for history and candidates.
Run `python -m content_hstu.run train --config CONFIG`. Start with the 17 tests
and both server smokes in note 20; full-data configs are provided for after that
gate. No extra dependencies, title re-encoding or test-based tuning is required.


`hstu-title-selected-test-server.json` freezes the three completed validation
report paths and the selected title-init epoch30 candidate. It is an evaluation
admission recipe, not a training config. Run
`python scripts/evaluate_selected_title.py` (or `--check-only` for report/file
checks without model execution). No scoring/epoch override is exposed; see note20.
