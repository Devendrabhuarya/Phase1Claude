"""Glue: config -> data bundle -> partition -> FLData (used by scripts and tests)."""

from __future__ import annotations

from src.data.partition import partition_from_config
from src.data.preprocess import DatasetBundle, build_bundle, prepare_dataset
from src.data.synthetic import SYNTHETIC_FL_COUNTS, make_synthetic_sample
from src.fl.server import FLData, make_fl_data


def synthetic_bundle(cfg: dict, counts=None) -> DatasetBundle:
    """Bundle from in-memory synthetic data (root set shrunk to fit the small data)."""
    cfg["data"]["root_size"] = min(cfg["data"]["root_size"], 120)
    return build_bundle(make_synthetic_sample(counts or SYNTHETIC_FL_COUNTS, cfg["seed"]), cfg)


def build_fl_data(cfg: dict, synthetic: bool = False, alpha=None, verbose: bool = True):
    """Return ``(fl_data, partition, bundle)`` for the real (cached) or synthetic dataset."""
    bundle = synthetic_bundle(cfg) if synthetic else prepare_dataset(cfg, verbose=verbose)
    part = partition_from_config(bundle.y_pool, bundle.class_names, cfg, alpha=alpha, verbose=verbose)
    return make_fl_data(bundle, part), part, bundle
