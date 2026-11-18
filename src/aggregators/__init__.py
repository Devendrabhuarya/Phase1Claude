"""Baseline robust aggregators and the proposed cohort-aware method (Stages 4, 7-8).

Use :func:`get_aggregator` to build one by its config name.
"""

from src.aggregators.base import Aggregator, FedAvg

_REGISTRY = {FedAvg.name: FedAvg}


def register(cls):
    """Class decorator adding an aggregator to the registry by its ``name``."""
    _REGISTRY[cls.name] = cls
    return cls


def get_aggregator(name: str, cfg: dict | None = None) -> Aggregator:
    """Instantiate the aggregator called ``name`` (e.g. ``"fedavg"``)."""
    if name not in _REGISTRY:
        raise ValueError(f"Unknown aggregator '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](cfg)
