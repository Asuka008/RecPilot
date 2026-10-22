"""Pure-Python validation of controls matched to an explicit HSTU configuration."""

from pathlib import Path

from src.microlens_config import validate_config


ROOT = Path(__file__).resolve().parents[1]
CONTROL = "causal-prefix-mean-v1"


def resolved(path):
    """Resolve portable experiment paths relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def validate_control_config(config, reference):
    """Require identical experiment settings except the output path and encoder."""
    if config.get("control") != CONTROL or not isinstance(config.get("reference_config"), str) or not config["reference_config"].strip():
        raise ValueError("Expected causal-prefix-mean-v1 and an explicit reference_config")
    experiment = {key: value for key, value in config.items() if key not in {"control", "reference_config"}}
    validate_config(experiment)
    validate_config(reference)
    differences = [key for key in reference if key != "output_dir" and experiment[key] != reference[key]]
    if differences:
        raise ValueError(f"Control settings differ from HSTU reference: {differences}")
    if resolved(config["output_dir"]).resolve() == resolved(reference["output_dir"]).resolve():
        raise ValueError("Control and HSTU outputs must have separate directories")
