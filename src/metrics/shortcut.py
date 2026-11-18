"""Shortcut-feature check (Architecture README 5.1).

``Time_To_Live`` may encode the attacker host's OS and ``IAT`` the capture
session, so a model could "cheat" with them. We train the same MLP
**centrally** (no FL) once with all 39 features and once without these two
(37 features) and compare test metrics.

Decision rule from the README: if performance drops sharply without them
(macro-F1 drop >= ``shortcut_check.sharp_drop``), the model relies on the
shortcuts, and the reduced 37-feature set is used for all experiments.
"""

from __future__ import annotations

import copy

import pandas as pd
import torch

from src.data.ciciot2023 import SampledData
from src.data.preprocess import build_bundle
from src.fl.client import train_local
from src.metrics.detection import detection_metrics, predict
from src.models.mlp import build_model


def train_centralized(bundle, cfg: dict, epochs: int) -> dict:
    """Train one MLP on the whole client pool and return its test metrics."""
    torch.manual_seed(cfg["seed"])
    model = build_model(bundle.n_features, bundle.n_classes, cfg["model"])
    train_local(model, bundle.X_pool, bundle.y_pool, cfg["training"], bundle.n_classes,
                seed=cfg["seed"], epochs=epochs)
    y_pred = predict(model, bundle.X_test, cfg["fl"]["eval_batch_size"])
    return detection_metrics(bundle.y_test, y_pred, bundle.class_names)


def shortcut_check(sample: SampledData, cfg: dict, verbose: bool = True):
    """Compare all features vs. without shortcut features.

    Returns:
        (table, use_reduced): DataFrame with one column per setting plus the
        difference, and True if the reduced feature set should be used.
    """
    sc = cfg["shortcut_check"]
    results, n_feats = {}, {}
    for drop in (False, True):
        c = copy.deepcopy(cfg)
        c["data"]["drop_shortcut_features"] = drop
        bundle = build_bundle(sample, c)
        label = f"{bundle.n_features} features" + (" (no " + ", ".join(c["data"]["shortcut_features"]) + ")" if drop else "")
        if verbose:
            print(f"Training centralized model: {label} ...")
        results[label] = train_centralized(bundle, c, sc["epochs"])
        n_feats[drop] = label
    table = pd.DataFrame(results)
    table["difference"] = table[n_feats[True]] - table[n_feats[False]]
    f1_drop = -table.loc["macro_f1", "difference"]
    use_reduced = bool(f1_drop >= sc["sharp_drop"])
    return table, use_reduced
