"""Configuration loading.

All hyperparameters live in YAML files under ``configs/``. A run's config is
built by deep-merging, in order:

1. ``configs/base.yaml``          (shared defaults from the architecture README)
2. ``configs/<dataset>.yaml``     (dataset-specific paths and settings)
3. an optional ``overrides`` dict (e.g. one cell of the experiment grid)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

#: Repository root (two levels above this file: src/utils/config.py).
REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
#: Only CICIoT2023 is used (N-BaIoT was dropped by project decision).
#: Add a ``configs/<name>.yaml`` and a name here to support another dataset.
SUPPORTED_DATASETS = ("ciciot2023",)


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Return a new dict where ``override`` is merged recursively into ``base``.

    Nested dicts are merged key by key; any other value in ``override``
    (including lists) replaces the value in ``base``. Inputs are not modified.
    """
    result = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_yaml(path: Path | str) -> dict:
    """Read a YAML file and return its content as a dict (empty file -> {})."""
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_config(
    dataset: str = "ciciot2023",
    overrides: Optional[Mapping[str, Any]] = None,
    config_dir: Path | str = CONFIG_DIR,
) -> dict:
    """Build the full config for ``dataset``: base + dataset yaml + overrides.

    Args:
        dataset: dataset name, currently only ``"ciciot2023"``.
        overrides: optional nested dict applied last (highest priority).
        config_dir: folder holding ``base.yaml`` and the dataset yaml files.

    Raises:
        ValueError: if the dataset name is not supported.
    """
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose one of {SUPPORTED_DATASETS}.")
    config_dir = Path(config_dir)
    cfg = load_yaml(config_dir / "base.yaml")
    cfg = deep_merge(cfg, load_yaml(config_dir / f"{dataset}.yaml"))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    cfg["dataset"] = dataset
    return cfg


def save_config(cfg: Mapping[str, Any], path: Path | str) -> None:
    """Write a config dict to YAML (used to store the exact config of each run)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(cfg), fh, sort_keys=False)
