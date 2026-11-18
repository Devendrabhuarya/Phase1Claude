"""Reproducibility helpers: one call fixes every random number generator."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Fix the seeds of Python, NumPy and (if installed) PyTorch.

    Also asks PyTorch for deterministic algorithms on CPU so that two runs
    with the same seed give the same model.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # torch is optional for data-only stages/tests
        return
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_rng(seed: int, *offsets: int) -> np.random.Generator:
    """Return an independent NumPy Generator for one component.

    Using separate generators (e.g. ``make_rng(seed, client_id)``) keeps
    partitioning, attacks and sampling reproducible even if the order in
    which components draw random numbers changes.
    """
    return np.random.default_rng([seed, *offsets])
