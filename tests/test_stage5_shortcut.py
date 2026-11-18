"""Stage 5 tests: shortcut-feature check outputs a comparison table and detects a real shortcut."""

import numpy as np
import pytest

from src.data.synthetic import CICIOT_FEATURES, CLASS_ORDER, make_synthetic_sample
from src.metrics.shortcut import shortcut_check
from src.utils.config import load_config

COUNTS = {c: 600 for c in CLASS_ORDER}


@pytest.fixture
def cfg():
    c = load_config("ciciot2023")
    c["data"]["root_size"] = 80
    c["shortcut_check"]["epochs"] = 5
    return c


def test_table_format_no_shortcut(cfg):
    sample = make_synthetic_sample(COUNTS, seed=0, separation=0.6)
    table, use_reduced = shortcut_check(sample, cfg, verbose=False)
    assert list(table.columns) == ["39 features", "37 features (no Time_To_Live, IAT)", "difference"]
    assert {"accuracy", "macro_f1", "mcc"} <= set(table.index)
    assert not use_reduced          # signal is spread over all features


def test_detects_shortcut(cfg):
    """If only TTL and IAT carry the class signal, removing them must cause a sharp drop."""
    cfg["shortcut_check"]["epochs"] = 20     # small data: needs more SGD steps
    rng = np.random.default_rng(0)
    sample = make_synthetic_sample(COUNTS, seed=0, separation=0.0, noise=1.0)
    sample.X[:] = rng.normal(size=sample.X.shape)            # pure noise ...
    for name in ("Time_To_Live", "IAT"):                       # ... except the shortcuts
        sample.X[:, CICIOT_FEATURES.index(name)] = sample.y * 3.0 + rng.normal(size=len(sample.y))
    table, use_reduced = shortcut_check(sample, cfg, verbose=False)
    assert use_reduced
    assert table.loc["macro_f1", "difference"] < -0.3
