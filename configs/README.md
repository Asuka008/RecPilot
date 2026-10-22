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
