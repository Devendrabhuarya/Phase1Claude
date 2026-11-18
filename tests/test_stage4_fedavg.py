"""Stage 4 tests: MLP, local training, FedAvg, metrics, end-to-end FL run."""

import numpy as np
import pytest
import torch

from src.aggregators import get_aggregator
from src.aggregators.base import FedAvg
from src.fl.client import class_weights, client_update
from src.fl.pipeline import build_fl_data
from src.fl.server import run_federated, screening_rates
from src.metrics.detection import attack_success_rate, detection_metrics, majority_baseline
from src.models.mlp import build_model, final_layer_slice, get_weights, set_weights
from src.utils.config import load_config
from src.utils.seed import set_seed


@pytest.fixture
def cfg():
    c = load_config("ciciot2023")
    set_seed(c["seed"])
    return c


def test_mlp_shapes_from_data(cfg):
    for n_feat in (39, 37):
        m = build_model(n_feat, 8, cfg["model"])
        assert m(torch.zeros(5, n_feat)).shape == (5, 8)
        n_params = len(get_weights(m))
        assert n_params == n_feat * 128 + 128 + 128 * 64 + 64 + 64 * 8 + 8


def test_weights_roundtrip_and_final_layer(cfg):
    m = build_model(39, 8, cfg["model"])
    w = get_weights(m)
    set_weights(m, torch.zeros_like(w))
    assert get_weights(m).abs().sum() == 0
    sl = final_layer_slice(m)
    assert sl.stop == len(w) and sl.stop - sl.start == 64 * 8 + 8


def test_class_weights_balanced_and_capped():
    y = np.array([0] * 90 + [1] * 10)
    w = class_weights(y, 3, max_weight=10.0)
    assert w[0] < 1 < w[1] and w[2] == 1.0
    w = class_weights(np.array([0] * 999 + [1]), 2, max_weight=10.0)
    assert w[1] == 10.0


def test_client_update_deterministic_and_nonzero(cfg):
    rng = np.random.default_rng(0)
    X, y = rng.normal(size=(300, 39)).astype(np.float32), rng.integers(0, 8, 300)
    m = build_model(39, 8, cfg["model"])
    g = get_weights(m)
    d1 = client_update(m, g, X, y, cfg["training"], 8, seed=1)
    d2 = client_update(m, g, X, y, cfg["training"], 8, seed=1)
    assert torch.equal(d1, d2) and d1.abs().sum() > 0


def test_fedavg_weighted_mean():
    u = torch.tensor([[1.0, 0.0], [3.0, 2.0]])
    out, info = FedAvg().aggregate(u, torch.zeros(2), {"n_samples": torch.tensor([1.0, 3.0]),
                                                       "client_ids": [0, 1]})
    assert torch.allclose(out, torch.tensor([2.5, 1.5]))
    assert info["selected"] == [0, 1]
    assert isinstance(get_aggregator("fedavg"), FedAvg)
    with pytest.raises(ValueError):
        get_aggregator("nope")


def test_metrics():
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([0, 0, 1, 0, 2, 0])
    m = detection_metrics(y, p, ["Benign", "A", "B"])
    assert m["accuracy"] == pytest.approx(4 / 6)
    assert m["recall_A"] == 0.5 and m["recall_Benign"] == 1.0
    assert attack_success_rate(y, p, benign_id=0) == 0.5
    base = majority_baseline(np.array([0, 0, 1]), y, ["Benign", "A", "B"])
    assert base["accuracy"] == pytest.approx(2 / 6) and base["mcc"] == 0.0


def test_screening_rates():
    r = screening_rates(selected=[0, 1, 3], client_ids=[0, 1, 2, 3], malicious={2, 3})
    assert r["screen_tpr"] == 0.5 and r["screen_fpr"] == 0.0


def test_fedavg_beats_majority_on_synthetic(cfg):
    """Stage 4 'done when': FedAvg clearly beats a majority-class baseline."""
    data, part, bundle = build_fl_data(cfg, synthetic=True, verbose=False)
    res = run_federated(data, cfg, get_aggregator("fedavg", cfg), rounds=10, verbose=False)
    base = majority_baseline(bundle.y_pool, bundle.y_test, bundle.class_names)
    print(f"FedAvg acc={res['final']['accuracy']:.3f} F1={res['final']['macro_f1']:.3f} | "
          f"majority acc={base['accuracy']:.3f} F1={base['macro_f1']:.3f}")
    assert res["final"]["accuracy"] > base["accuracy"] + 0.2
    assert res["final"]["macro_f1"] > base["macro_f1"] + 0.3
    assert len(res["history"]) == 10 and "time_server_s" in res["history"][0]


def test_fl_run_reproducible(cfg):
    data, _, _ = build_fl_data(cfg, synthetic=True, verbose=False)
    r1 = run_federated(data, cfg, get_aggregator("fedavg"), rounds=2, verbose=False)
    r2 = run_federated(data, cfg, get_aggregator("fedavg"), rounds=2, verbose=False)
    assert torch.equal(r1["weights"], r2["weights"])
