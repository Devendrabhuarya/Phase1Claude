"""Stage 1 tests on synthetic CICIoT2023-like files: label mapping, chunked
class-capped sampling, cleaning, splits, preprocessing and caching."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.ciciot2023 import (list_csv_files, load_class_capped, load_or_build_sample,
                                 load_sample, save_sample)
from src.data.labels import (UnmappedLabelError, build_label_map, inspect_labels, map_label,
                             normalize_label)
from src.data.preprocess import Preprocessor, build_bundle, make_splits, select_features
from src.data.synthetic import CICIOT_FEATURES, SYNTHETIC_LABELS, write_synthetic_csvs
from src.utils.config import load_config


@pytest.fixture
def cfg(tmp_path):
    c = load_config("ciciot2023")
    c["paths"]["data_dir"] = str(tmp_path / "raw")
    c["paths"]["cache_dir"] = str(tmp_path / "cache")
    c["data"]["chunk_size"] = 300        # force several chunks per file
    c["data"]["class_cap"] = 250
    c["data"]["root_size"] = 80
    return c


@pytest.fixture
def files(cfg):
    return write_synthetic_csvs(cfg["paths"]["data_dir"], n_files=3, rows_per_label=200,
                                rare_rows=20, seed=0)


def _load(cfg, files, **kw):
    d = cfg["data"]
    args = dict(label_column=d["label_column"], label_keywords=d["label_keywords"],
                class_names=d["classes"], class_cap=d["class_cap"], chunk_size=d["chunk_size"],
                seed=cfg["seed"], n_expected_features=39, verbose=False)
    args.update(kw)
    return load_class_capped(files, **args)


# ------------------------------------------------------------ label mapping

def test_normalize_label():
    assert normalize_label("DDoS-ICMP_Flood") == "DDOSICMPFLOOD"
    assert normalize_label(" dos tcp flood ") == "DOSTCPFLOOD"


def test_all_synthetic_labels_map_correctly(cfg):
    kws = cfg["data"]["label_keywords"]
    for raw, expected in SYNTHETIC_LABELS.items():
        assert map_label(raw, kws) == expected, raw


def test_ddos_checked_before_dos(cfg):
    kws = cfg["data"]["label_keywords"]
    assert map_label("DDoS-UDP_Flood", kws) == "DDoS"
    assert map_label("DoS-UDP_Flood", kws) == "DoS"


def test_more_real_label_variants(cfg):
    kws = cfg["data"]["label_keywords"]
    cases = {"Recon-HostDiscovery": "Recon", "Recon-OSScan": "Recon", "Recon-PingSweep": "Recon",
             "DDoS-ACK_Fragmentation": "DDoS", "Mirai-udpplain": "Mirai",
             "BENIGN": "Benign", "Backdoor_Malware": "Web", "Brute Force": "BruteForce"}
    for raw, expected in cases.items():
        assert map_label(raw, kws) == expected, raw


def test_unmapped_labels_raise_and_are_listed(cfg):
    with pytest.raises(UnmappedLabelError) as e:
        build_label_map(["BenignTraffic", "WeirdAttack", "Other_Thing"], cfg["data"]["label_keywords"])
    assert "WeirdAttack" in str(e.value) and "Other_Thing" in str(e.value)


def test_unmapped_label_in_file_raises(cfg):
    paths = write_synthetic_csvs(cfg["paths"]["data_dir"], n_files=1, extra_labels={"ZeroDay": 5})
    with pytest.raises(UnmappedLabelError, match="ZeroDay"):
        _load(cfg, paths)


def test_inspect_labels_counts(files, capsys):
    counts = inspect_labels(files, chunk_size=300)
    assert counts["BenignTraffic"] >= 3 * 200          # + any injected duplicate rows
    assert set(counts.index) == set(SYNTHETIC_LABELS)
    assert counts.sum() == sum(len(pd.read_csv(f)) for f in files)
    assert "unique raw labels" in capsys.readouterr().out


# ------------------------------------------------------------ chunked sampling

def test_class_cap_and_rare_classes_kept(cfg, files):
    s = _load(cfg, files)
    counts = s.class_counts()
    assert (counts <= cfg["data"]["class_cap"]).all()
    # Web: 6 raw labels x 20 rows x 3 files = 360 rows -> capped at 250
    assert counts["Web"] == 250
    # BruteForce: 1 label x 20 rows x 3 files = 60 rows -> all kept (rare class)
    assert counts["BruteForce"] == 60
    assert counts["Benign"] == 250
    assert s.X.shape == (len(s.y), 39) and s.X.dtype == np.float32
    assert s.feature_names == CICIOT_FEATURES


def test_every_class_present(cfg, files):
    s = _load(cfg, files)
    assert (s.class_counts() > 0).all()


def test_rows_seen_matches_files(cfg, files):
    s = _load(cfg, files)
    total_rows = sum(len(pd.read_csv(f)) for f in files)
    assert s.class_rows_seen.sum() == total_rows == s.raw_label_counts.sum()


def test_duplicates_removed_and_inf_replaced(cfg, files):
    cfg["data"]["class_cap"] = 10**6
    s = _load(cfg, files, class_cap=10**6)
    full = pd.concat([pd.read_csv(f) for f in files])
    assert len(s.y) == len(full.drop_duplicates())
    assert not np.isinf(s.X).any()
    assert np.isnan(s.X).any()          # NaN kept for train-median imputation later


def test_sampling_is_deterministic_and_chunk_independent(cfg, files):
    a = _load(cfg, files)
    b = _load(cfg, files)
    assert np.array_equal(a.y, b.y) and np.array_equal(a.X, b.X, equal_nan=True)
    c = _load(cfg, files, seed=cfg["seed"] + 1)
    assert not np.array_equal(a.X, c.X, equal_nan=True)


def test_wrong_feature_count_raises(cfg, files):
    with pytest.raises(ValueError, match="Expected 37 features"):
        _load(cfg, files, n_expected_features=37)


def test_list_files_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_csv_files(tmp_path, "Merged*.csv")


def test_cache_roundtrip_and_reuse(cfg, files):
    s1 = load_or_build_sample(cfg, verbose=False)
    cached = list(Path(cfg["paths"]["cache_dir"]).glob("*.npz"))
    assert len(cached) == 1
    s2 = load_or_build_sample(cfg, verbose=False)      # served from cache
    assert np.array_equal(s1.X, s2.X, equal_nan=True) and s1.feature_names == s2.feature_names
    assert s1.raw_label_counts.sort_index().equals(s2.raw_label_counts.sort_index())


# ------------------------------------------------------------ splits + preprocessing

def test_splits_disjoint_stratified(cfg, files):
    s = _load(cfg, files)
    pool, test, root = make_splits(s.y, 0.2, 80, seed=1)
    assert len(set(pool) & set(test)) == len(set(pool) & set(root)) == len(set(test) & set(root)) == 0
    assert len(pool) + len(test) + len(root) == len(s.y)
    assert len(root) == 80
    assert abs(len(test) - 0.2 * len(s.y)) <= 1
    # every class in test and pool
    assert set(np.unique(s.y[test])) == set(np.unique(s.y)) == set(np.unique(s.y[pool]))


def test_preprocessor_fits_on_train_only():
    rng = np.random.default_rng(0)
    train = rng.normal(5, 2, size=(500, 3))
    train[0, 0] = np.inf
    train[1, 1] = np.nan
    other = rng.normal(100, 2, size=(50, 3))
    p = Preprocessor().fit(train)
    t = p.transform(train)
    assert np.isfinite(t).all() and t.dtype == np.float32
    assert np.allclose(t.mean(axis=0), 0, atol=1e-5)
    assert (p.transform(other).mean(axis=0) > 10).all()   # test stats NOT used


def test_select_features_shortcut():
    keep = select_features(CICIOT_FEATURES, ["Time_To_Live", "IAT"])
    assert len(keep) == 37
    with pytest.raises(ValueError):
        select_features(CICIOT_FEATURES, ["NotAColumn"])


@pytest.mark.parametrize("drop", [False, True])
def test_build_bundle(cfg, files, drop, capsys):
    cfg["data"]["drop_shortcut_features"] = drop
    s = _load(cfg, files)
    b = build_bundle(s, cfg)
    assert b.n_features == (37 if drop else 39)
    assert b.n_classes == 8
    assert len(b.y_root) == 80
    for X in (b.X_pool, b.X_test, b.X_root):
        assert np.isfinite(X).all() and X.dtype == np.float32
    table = b.class_table()
    print(table)                                   # class counts printed (Stage 1 "done when")
    assert table.sum().sum() == len(s.y)
    assert (table["test"] > 0).all()
    assert "Benign" in capsys.readouterr().out


def test_class_cap_override(cfg, files):
    s = _load(cfg, files, class_cap_overrides={"Benign": 400})
    counts = s.class_counts()
    assert counts["Benign"] > 250 and counts["DDoS"] == 250
