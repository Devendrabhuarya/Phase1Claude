"""Memory-safe, class-capped loading of the 63 CICIoT2023 ``Merged*.csv`` files.

Why "bottom-k" sampling?
    We want up to ``class_cap`` rows per class drawn uniformly at random from
    *all* files, without ever holding all files in memory. Every row gets a
    random key in [0, 1); for each class we keep only the ``class_cap`` rows
    with the smallest keys seen so far. After the last chunk this is exactly
    a uniform random sample of that class (rare classes simply keep all
    their rows). Memory stays at roughly ``n_classes * class_cap`` rows plus
    one chunk.

Duplicates and inf values are handled while sampling so the kept sample is
already de-duplicated. The sample can be cached as ``.npz`` on Drive so the
slow full scan (~13 GB of CSV) is done only once.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import pandas as pd

from src.data.labels import UnmappedLabelError, map_labels

KEY_COL = "__sample_key__"
CLASS_COL = "__class__"


@dataclass
class SampledData:
    """Class-capped raw sample (not yet split or scaled)."""

    X: np.ndarray                 # (n_rows, n_features) float32, may contain NaN
    y: np.ndarray                 # (n_rows,) int64 class indices into class_names
    feature_names: list
    class_names: list
    raw_label_counts: pd.Series   # every raw label and its count over ALL scanned rows
    class_rows_seen: pd.Series    # rows per class over all files (before cap/dedup)

    def class_counts(self) -> pd.Series:
        """Rows kept per class, in class order."""
        counts = np.bincount(self.y, minlength=len(self.class_names))
        return pd.Series(counts, index=self.class_names, name="kept")


def list_csv_files(data_dir: Path | str, pattern: str) -> list:
    """Return the sorted list of CSV files matching ``pattern`` in ``data_dir``."""
    files = sorted(Path(data_dir).glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' in {data_dir}")
    return files


def _read_columns(path: Path) -> list:
    """Column names of a CSV (whitespace stripped), reading only the header."""
    return [c.strip() for c in pd.read_csv(path, nrows=0).columns]


def load_class_capped(
    files: Sequence[Path | str],
    label_column: str,
    label_keywords: Sequence[Sequence[str]],
    class_names: Sequence[str],
    class_cap: int,
    chunk_size: int,
    seed: int,
    drop_duplicates: bool = True,
    float_dtype: str = "float32",
    n_expected_features: Optional[int] = None,
    verbose: bool = True,
    class_cap_overrides: Optional[Mapping[str, int]] = None,
) -> SampledData:
    """Scan ``files`` chunk by chunk and keep a uniform sample of <= class_cap rows per class.

    ``class_cap_overrides`` gives a different cap for some classes (e.g. Benign).

    Raises:
        ValueError: if files have different columns, the feature count is not
            ``n_expected_features``, or a label maps to a class not in ``class_names``.
        UnmappedLabelError: if any raw label matches no keyword (file name included).
    """
    files = [Path(f) for f in files]
    columns = _read_columns(files[0])
    if label_column not in columns:
        raise ValueError(f"Label column '{label_column}' not found in {files[0].name}: {columns}")
    feature_names = [c for c in columns if c != label_column]
    if n_expected_features is not None and len(feature_names) != n_expected_features:
        raise ValueError(f"Expected {n_expected_features} features, found {len(feature_names)}: "
                         f"{feature_names}")

    class_index = {c: i for i, c in enumerate(class_names)}
    caps = {c: int((class_cap_overrides or {}).get(c, class_cap)) for c in class_names}
    rng = np.random.default_rng(seed)
    kept = {c: None for c in class_names}          # class -> DataFrame (features + key)
    raw_counts: dict = {}
    seen = dict.fromkeys(class_names, 0)
    missing_label_rows: dict = {}                  # file name -> rows dropped (empty label)

    for i, path in enumerate(files):
        cols = _read_columns(path)
        if cols != columns:
            raise ValueError(f"{path.name} has different columns than {files[0].name}")
        # Read with original (unstripped) header names, then strip them.
        reader = pd.read_csv(path, chunksize=chunk_size)
        for chunk in reader:
            chunk.columns = [c.strip() for c in chunk.columns]
            # Rows with an empty label (e.g. a truncated last line) cannot be
            # classified; drop them explicitly and report how many.
            no_label = chunk[label_column].isna()
            if no_label.any():
                n_bad = int(no_label.sum())
                missing_label_rows[path.name] = missing_label_rows.get(path.name, 0) + n_bad
                chunk = chunk[~no_label]
                if chunk.empty:
                    continue
            labels = chunk[label_column]
            for lab, n in labels.value_counts(dropna=False).items():
                raw_counts[lab] = raw_counts.get(lab, 0) + int(n)
            try:
                classes = map_labels(labels, label_keywords)
            except UnmappedLabelError as err:
                raise UnmappedLabelError(f"{path.name}: {err}") from None
            unknown = set(classes.unique()) - set(class_names)
            if unknown:
                raise ValueError(f"Keyword map produced classes not in data.classes: {unknown}")

            feats = chunk[feature_names].apply(pd.to_numeric, errors="coerce").astype(float_dtype)
            feats = feats.replace([np.inf, -np.inf], np.nan)
            feats[KEY_COL] = rng.random(len(feats))
            for cls, part in feats.groupby(classes.to_numpy(), sort=False):
                seen[cls] += len(part)
                merged = part if kept[cls] is None else pd.concat([kept[cls], part], ignore_index=True)
                if drop_duplicates:
                    merged = merged.drop_duplicates(subset=feature_names)
                if len(merged) > caps[cls]:
                    merged = merged.nsmallest(caps[cls], KEY_COL)
                kept[cls] = merged
        if verbose:
            sizes = {c: (0 if kept[c] is None else len(kept[c])) for c in class_names}
            print(f"[{i + 1}/{len(files)}] {path.name}: kept per class {sizes}")
            if path.name in missing_label_rows:
                print(f"    WARNING: dropped {missing_label_rows[path.name]} row(s) with an "
                      f"empty '{label_column}' in {path.name} (truncated/corrupt line?)")

    if missing_label_rows and verbose:
        print(f"Rows dropped for empty label, total {sum(missing_label_rows.values())}: "
              f"{missing_label_rows}")

    X_parts, y_parts = [], []
    for cls in class_names:
        df = kept[cls]
        if df is None or len(df) == 0:
            continue
        df = df.sort_values(KEY_COL)  # random order within class, reproducible
        X_parts.append(df[feature_names].to_numpy(dtype=float_dtype))
        y_parts.append(np.full(len(df), class_index[cls], dtype=np.int64))
    if not X_parts:
        raise ValueError("No rows were loaded.")

    return SampledData(
        X=np.concatenate(X_parts),
        y=np.concatenate(y_parts),
        feature_names=feature_names,
        class_names=list(class_names),
        raw_label_counts=pd.Series(raw_counts, name="count").sort_values(ascending=False),
        class_rows_seen=pd.Series(seen, name="seen"),
    )


# ------------------------------------------------------------------ cache


def cache_path(cfg: dict, n_files: int) -> Path:
    """Cache file name encodes everything that changes the sample."""
    d = cfg["data"]
    over = "".join(f"_{k}{v}" for k, v in sorted((d.get("class_cap_overrides") or {}).items()))
    name = (f"{cfg['dataset']}_files{n_files}_cap{d['class_cap']}{over}_seed{cfg['seed']}"
            f"_dedup{int(d['drop_duplicates'])}.npz")
    return Path(cfg["paths"]["cache_dir"]) / name


def save_sample(sample: SampledData, path: Path | str) -> None:
    """Store a SampledData as compressed ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, X=sample.X, y=sample.y,
        feature_names=np.array(sample.feature_names, dtype=object),
        class_names=np.array(sample.class_names, dtype=object),
        raw_labels=np.array([str(k) for k in sample.raw_label_counts.index], dtype=object),
        raw_counts=sample.raw_label_counts.to_numpy(),
        seen=sample.class_rows_seen.to_numpy(),
    )


def load_sample(path: Path | str) -> SampledData:
    """Inverse of :func:`save_sample`."""
    z = np.load(path, allow_pickle=True)
    class_names = list(z["class_names"])
    return SampledData(
        X=z["X"], y=z["y"], feature_names=list(z["feature_names"]), class_names=class_names,
        raw_label_counts=pd.Series(z["raw_counts"], index=list(z["raw_labels"]), name="count"),
        class_rows_seen=pd.Series(z["seen"], index=class_names, name="seen"),
    )


def load_or_build_sample(cfg: dict, files: Optional[Sequence[Path]] = None,
                         use_cache: bool = True, verbose: bool = True) -> SampledData:
    """Return the class-capped sample, reading it from the Drive cache if present.

    Args:
        cfg: full config (``load_config("ciciot2023")``).
        files: CSV files to read; default = all files matching ``paths.file_pattern``.
        use_cache: read/write ``paths.cache_dir``.
    """
    if files is None:
        files = list_csv_files(cfg["paths"]["data_dir"], cfg["paths"]["file_pattern"])
    path = cache_path(cfg, len(files))
    if use_cache and path.is_file():
        if verbose:
            print(f"Loading cached sample: {path}")
        return load_sample(path)
    d = cfg["data"]
    sample = load_class_capped(
        files, label_column=d["label_column"], label_keywords=d["label_keywords"],
        class_names=d["classes"], class_cap=d["class_cap"], chunk_size=d["chunk_size"],
        seed=cfg["seed"], drop_duplicates=d["drop_duplicates"], float_dtype=d["float_dtype"],
        n_expected_features=d["n_expected_features"], verbose=verbose,
        class_cap_overrides=d.get("class_cap_overrides"),
    )
    if use_cache:
        save_sample(sample, path)
        if verbose:
            print(f"Saved sample cache: {path}")
    return sample
