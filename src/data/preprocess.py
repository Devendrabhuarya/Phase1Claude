"""Preprocessing, splitting and the final dataset bundle (Architecture README 5.1).

Pipeline (``prepare_dataset``):

1. class-capped, de-duplicated sample (``ciciot2023.load_or_build_sample``)
2. optional removal of shortcut features (Time_To_Live, IAT)
3. stratified split: 20 % global test set -> ~500 root flows from the rest
   -> remaining rows form the client pool (partitioned in Stage 3)
4. ``Preprocessor`` fitted on the client pool only (median imputation of
   NaN/inf + standard scaling), then applied to pool, test and root.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.data.ciciot2023 import SampledData, load_or_build_sample


class Preprocessor:
    """Median imputation of NaN/inf followed by standard scaling.

    ``fit`` must be called on training data only, so no statistic of the test
    or root set leaks into the model.
    """

    def __init__(self, nan_fill: str = "train_median"):
        if nan_fill != "train_median":
            raise ValueError(f"Unsupported nan_fill '{nan_fill}'")
        self.nan_fill = nan_fill
        self.medians_: Optional[np.ndarray] = None
        self.scaler_ = StandardScaler()

    @staticmethod
    def _clean(X: np.ndarray) -> np.ndarray:
        X = np.array(X, dtype=np.float64, copy=True)
        X[~np.isfinite(X)] = np.nan
        return X

    def fit(self, X: np.ndarray) -> "Preprocessor":
        X = self._clean(X)
        med = np.nanmedian(X, axis=0) if len(X) else np.zeros(X.shape[1])
        self.medians_ = np.where(np.isnan(med), 0.0, med)  # all-NaN column -> 0
        self.scaler_.fit(self._fill(X))
        return self

    def _fill(self, X: np.ndarray) -> np.ndarray:
        rows, cols = np.where(np.isnan(X))
        X[rows, cols] = self.medians_[cols]
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.medians_ is None:
            raise RuntimeError("Preprocessor.fit must be called first")
        return self.scaler_.transform(self._fill(self._clean(X))).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


def select_features(feature_names: Sequence[str], drop: Sequence[str]) -> list:
    """Indices of features to keep after removing ``drop`` (each must exist)."""
    missing = [f for f in drop if f not in feature_names]
    if missing:
        raise ValueError(f"Features to drop not found: {missing}")
    return [i for i, f in enumerate(feature_names) if f not in set(drop)]


def make_splits(y: np.ndarray, test_fraction: float, root_size: int, seed: int):
    """Stratified index split into (pool_idx, test_idx, root_idx).

    The test set is ``test_fraction`` of all rows; the root set is
    ``root_size`` rows drawn (stratified) from the remaining held-out pool;
    everything else is the client pool. The three index sets are disjoint.
    """
    idx = np.arange(len(y))
    rest_idx, test_idx = train_test_split(idx, test_size=test_fraction, stratify=y,
                                          random_state=seed)
    root_idx, pool_idx = train_test_split(rest_idx, train_size=root_size, stratify=y[rest_idx],
                                          random_state=seed)
    return np.sort(pool_idx), np.sort(test_idx), np.sort(root_idx)


@dataclass
class DatasetBundle:
    """Everything later stages need. All X arrays are scaled float32."""

    X_pool: np.ndarray
    y_pool: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_root: np.ndarray
    y_root: np.ndarray
    feature_names: list
    class_names: list
    preprocessor: Preprocessor = field(repr=False)

    @property
    def n_features(self) -> int:
        return self.X_pool.shape[1]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def class_table(self) -> pd.DataFrame:
        """Rows per class in pool / test / root."""
        k = self.n_classes
        return pd.DataFrame({
            "pool": np.bincount(self.y_pool, minlength=k),
            "test": np.bincount(self.y_test, minlength=k),
            "root": np.bincount(self.y_root, minlength=k),
        }, index=self.class_names)


def build_bundle(sample: SampledData, cfg: dict) -> DatasetBundle:
    """Feature selection, stratified splits and preprocessing for a sample."""
    d = cfg["data"]
    drop = d["shortcut_features"] if d["drop_shortcut_features"] else []
    keep = select_features(sample.feature_names, drop)
    X = sample.X[:, keep]
    names = [sample.feature_names[i] for i in keep]

    pool_idx, test_idx, root_idx = make_splits(sample.y, d["test_fraction"], d["root_size"],
                                               cfg["seed"])
    prep = Preprocessor(d["nan_fill"]).fit(X[pool_idx])
    return DatasetBundle(
        X_pool=prep.transform(X[pool_idx]), y_pool=sample.y[pool_idx],
        X_test=prep.transform(X[test_idx]), y_test=sample.y[test_idx],
        X_root=prep.transform(X[root_idx]), y_root=sample.y[root_idx],
        feature_names=names, class_names=list(sample.class_names), preprocessor=prep,
    )


def prepare_dataset(cfg: dict, files=None, use_cache: bool = True,
                    verbose: bool = True) -> DatasetBundle:
    """Full Stage-1 pipeline: sample -> splits -> preprocessing."""
    sample = load_or_build_sample(cfg, files=files, use_cache=use_cache, verbose=verbose)
    bundle = build_bundle(sample, cfg)
    if verbose:
        print(f"Features: {bundle.n_features}  Classes: {bundle.n_classes}")
        print(bundle.class_table().to_string())
    return bundle
