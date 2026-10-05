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

## SASRec data loading

The complete PyTorch Dataset and DataLoader implementation lives in
`src/data/dataset.py`. It builds shifted next-item training pairs and
chronology-safe validation and test examples. Verify it on the server with:

```bash
python -m unittest tests.test_dataset -v
```

The data-loading milestone is complete when all dataset tests pass. See
`docs/learning-01-sasrec-data-loader.md` for the first teaching note, including
the sequence transformation, tensor contract, masking, and leakage prevention.

## Conventions

- `data/raw/` is read-only by convention. All derived artifacts go to `data/processed/`.
- Preprocessing must be deterministic: same raw input -> identical processed output.
- Intermediate artifacts are keyed by dataset name + config hash so multiple
  experiments can coexist.
