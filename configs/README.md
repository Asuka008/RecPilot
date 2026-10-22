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
| `hstu-microlens-train-server.json` | Draft full-data run: 30 epochs, validation NDCG@10 selection; review after smoke results |
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
