# Experiment configurations

Keep portable model and experiment settings in this directory. Machine-local
paths and credentials must stay in ignored `*.local.yaml` files or environment
variables.

`baselines-server.json` runs CPU Popularity and ItemCF on the full processed
catalog. It deliberately has no sampled candidate or user limit. Execute it on
the server, not as a Mac smoke test. Paths resolve from the repository root;
`--data-dir` and `--output-dir` can override them without editing shared config.
Use `python -m unittest tests.test_baselines -v` for tiny synthetic local checks.
