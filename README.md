# AgenticRec

Generative recommendation system.

## Directory structure

```
AgenticRec/
├── configs/          # Portable experiment configurations.
├── data/
│   ├── raw/          # Original, immutable data dumps. Never edit in place.
│   └── processed/    # Cleaned / tokenized datasets ready for training.
├── docs/             # Project and server workflow documentation.
├── environments/     # Reproducible Mac and H200 environment specifications.
├── src/
│   ├── data/          # PyTorch Dataset and DataLoader code.
│   └── preprocess.py # Raw -> processed pipeline.
├── tests/            # Focused automated tests.
└── README.md
```

Datasets, physical Conda environments, checkpoints, and run logs are not stored
in Git.

## Mac setup

```bash
conda env create -f environments/mac.yml
conda activate agenticrec
```

If the `agenticrec` environment already exists, synchronize it with:

```bash
conda env update -n agenticrec -f environments/mac.yml --prune
```

The Mac is for editing, preprocessing, syntax checks, and the smallest local
smoke tests. The H200 server is authoritative for PyTorch model and training
validation. See `docs/server-workflow.md` for the persistent server layout.

## Preprocessing the data

```bash
python -m src.preprocess
```

The script validates the raw interactions, sorts each user's history by time,
maps item IDs to continuous indices, and creates a leave-one-out split. It
writes the following files to `data/processed/`:

- `sequences.jsonl`: one chronological train/validation/test record per user
- `item_mapping.json`: reversible original-item-ID and model-index mappings
- `metadata.json`: preprocessing configuration, counts, and source checksum

## Why preprocessing is needed

- Data checks catch missing, duplicated, or malformed interactions before training.
- Chronological user sequences let the model learn to predict the next item from past behavior.
- Continuous item IDs can be used directly by the model's embedding table.
- A time-preserving train/validation/test split prevents future behavior from leaking into training.
- Saved mappings and deterministic outputs make experiments reproducible and comparable.

In short: raw interaction rows are turned into trustworthy model inputs and
fair evaluation targets.

## Guided SASRec data-loading exercise

The PyTorch Dataset and DataLoader scaffold lives in `src/data/dataset.py`.
Complete only `build_training_example()` by following its English hints, then
run:

```bash
python -m unittest tests.test_dataset -v
```

The exercise is complete when all six dataset tests pass. Before the function
is implemented, two training-example tests are expected to fail with
`NotImplementedError`; the evaluation and DataLoader tests should already pass.

## Conventions

- `data/raw/` is read-only by convention. All derived artifacts go to `data/processed/`.
- Preprocessing must be deterministic: same raw input -> identical processed output.
- Intermediate artifacts are keyed by dataset name + config hash so multiple
  experiments can coexist.
