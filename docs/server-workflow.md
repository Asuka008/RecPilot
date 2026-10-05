# H200 server workflow

## Persistent server layout

```text
/home/zichenl8/
├── projects/AgenticRec/
├── datasets/AgenticRec/
├── conda-envs/agenticrec-h200/
└── runs/AgenticRec/
```

Only `projects/AgenticRec/` is synchronized through Git. Datasets and training
artifacts are transferred separately. The physical Conda environment is never
committed or copied between macOS and Linux.

## Create the initial persistent environment

```bash
mkdir -p "$HOME/conda-envs"
conda create \
  --prefix "$HOME/conda-envs/agenticrec-h200" \
  python=3.11 \
  pip \
  -y
```

Activate it with:

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate "$HOME/conda-envs/agenticrec-h200"
```

Register it as a Jupyter kernel only after `ipykernel` is installed:

```bash
python -m ipykernel install \
  --user \
  --name agenticrec-h200 \
  --display-name "AgenticRec (H200)"
```

### Verified bootstrap state

The prefix was successfully created on 2026-10-04 with only the initial
runtime packages:

```text
Prefix:  /home/zichenl8/conda-envs/agenticrec-h200
Python:  3.11.16
pip:     26.2.1
Size:    approximately 232 MiB
```

PyTorch, CUDA runtime packages, pandas, and project libraries have not yet
been installed in this server prefix.

## Code synchronization

Develop on the Mac, commit and push to a private Git remote, then update the
server checkout with:

```bash
cd "$HOME/projects/AgenticRec"
git pull --ff-only
```

Do not edit the same source files independently on both machines.

## Data synchronization

Transfer data separately with `rsync` and compare the SHA-256 values recorded
in `metadata.json`. Never use Git for raw data, processed data, checkpoints, or
run logs.
