"""Stage 7 tests: each baseline on hand-made updates with a known outlier,
plus an end-to-end check that robust rules survive an attack that breaks FedAvg."""

import pytest
import torch

from src.aggregators import available, get_aggregator
from src.aggregators.robust import (default_f, fltrust_scores, krum_scores, multikrum_select,
                                    trimmed_mean)
from src.attacks import attack_from_config
from src.fl.pipeline import build_fl_data
from src.fl.server import run_federated
from src.utils.config import load_config
from src.utils.seed import set_seed

D = 20
OUTLIER = 6


@pytest.fixture
def cfg():
    c = load_config("ciciot2023")
    set_seed(c["seed"])
    return c


def honest_plus_outlier(outlier_value=-100.0, n_honest=6):
    """6 honest updates ≈ 1 (small noise) + 1 outlier at index 6."""
    g = torch.Generator().manual_seed(0)
    honest = torch.ones(n_honest, D) + 0.05 * torch.randn(n_honest, D, generator=g)
    return torch.cat([honest, torch.full((1, D), outlier_value)])


def ctx(n, g0=None):
    c = {"client_ids": list(range(n)), "n_samples": torch.ones(n), "round": 1}
    if g0 is not None:
        c["root_update"] = lambda: g0
    return c


def test_registry_has_all_baselines():
    assert {"fedavg", "median", "trimmed_mean", "krum", "multikrum", "fltrust"} <= set(available())


def test_fedavg_is_dragged_by_outlier():
    u = honest_plus_outlier()
    out, _ = get_aggregator("fedavg").aggregate(u, torch.zeros(D), ctx(7))
    assert out.mean() < -10                           # baseline is broken by one outlier


def test_median_ignores_outlier():
    u = honest_plus_outlier()
    out, info = get_aggregator("median").aggregate(u, torch.zeros(D), ctx(7))
    assert torch.allclose(out, torch.ones(D), atol=0.15) and info["selected"] is None


def test_median_even_n_averages_middle():
    u = torch.tensor([[1.0], [2.0], [3.0], [100.0]])
    out, _ = get_aggregator("median").aggregate(u, torch.zeros(1), ctx(4))
    assert out.item() == pytest.approx(2.5)


def test_trimmed_mean_removes_outlier(cfg):
    u = honest_plus_outlier()
    cfg["aggregator"]["trimmed_mean_beta"] = 0.15     # k = floor(0.15·7) = 1 per side
    out, info = get_aggregator("trimmed_mean", cfg).aggregate(u, torch.zeros(D), ctx(7))
    assert info["trimmed_per_side"] == 1
    assert torch.allclose(out, torch.ones(D), atol=0.15)
    with pytest.raises(ValueError):
        trimmed_mean(u, 0.6)                           # k = 4 per side > n / 2


def test_krum_scores_and_selection():
    u = honest_plus_outlier()
    f = default_f(7)
    assert f == 2
    scores = krum_scores(u, f)
    assert int(torch.argmax(scores)) == OUTLIER       # outlier has the worst score
    out, info = get_aggregator("krum").aggregate(u, torch.zeros(D), ctx(7))
    assert info["selected"][0] != OUTLIER and len(info["selected"]) == 1
    assert torch.allclose(out, torch.ones(D), atol=0.3)


def test_multikrum_drops_outlier():
    u = honest_plus_outlier()
    out, info = get_aggregator("multikrum").aggregate(u, torch.zeros(D), ctx(7))
    assert OUTLIER not in info["selected"] and len(info["selected"]) == 7 - 2
    assert torch.allclose(out, torch.ones(D), atol=0.15)


def test_multikrum_catches_colluding_pair():
    u = torch.cat([honest_plus_outlier(n_honest=7)[:7], torch.full((2, D), 8.0)])   # 2 identical attackers
    keep = multikrum_select(u, f=default_f(9), m=9 - default_f(9))
    assert 7 not in keep and 8 not in keep


def test_fltrust_rejects_opposite_and_normalises_scaled():
    g0 = torch.ones(D)
    u = honest_plus_outlier(outlier_value=-1.0)       # outlier points against g0
    out, info = get_aggregator("fltrust").aggregate(u, torch.zeros(D), ctx(7, g0))
    assert info["trust_scores"][OUTLIER] == 0.0 and OUTLIER not in info["selected"]
    assert out.norm() == pytest.approx(g0.norm().item(), rel=0.01)

    boosted = honest_plus_outlier(outlier_value=1000.0)   # same direction, huge magnitude
    ts, normed = fltrust_scores(boosted, g0)
    assert normed[OUTLIER].norm() == pytest.approx(g0.norm().item(), rel=1e-4)
    out, _ = get_aggregator("fltrust").aggregate(boosted, torch.zeros(D), ctx(7, g0))
    assert out.norm() <= g0.norm() + 1e-4             # magnitude bounded by ||g0||


def test_fltrust_all_untrusted_gives_zero_update():
    g0 = torch.ones(D)
    u = -torch.ones(3, D)
    out, info = get_aggregator("fltrust").aggregate(u, torch.zeros(D), ctx(3, g0))
    assert out.abs().sum() == 0 and info["selected"] == []


@pytest.mark.parametrize("name", ["median", "trimmed_mean", "multikrum", "fltrust"])
def test_robust_rules_beat_fedavg_under_scaling(cfg, name):
    """Scaling attack (20 % spread) ruins FedAvg; each robust rule stays far better."""
    data, _, _ = build_fl_data(cfg, synthetic=True, verbose=False)
    cfg["attacks"].update(type="scaling", malicious_fraction=0.2, placement="spread")
    attack, _ = attack_from_config(cfg, data.client_groups, data.benign_id)
    fedavg = run_federated(data, cfg, get_aggregator("fedavg", cfg), attack=attack,
                           rounds=6, verbose=False)["final"]
    robust = run_federated(data, cfg, get_aggregator(name, cfg), attack=attack,
                           rounds=6, verbose=False)["final"]
    print(name, round(robust["accuracy"], 3), "vs FedAvg", round(fedavg["accuracy"], 3))
    assert robust["accuracy"] > fedavg["accuracy"] + 0.2


def test_multikrum_screening_rates_under_scaling(cfg):
    data, _, _ = build_fl_data(cfg, synthetic=True, verbose=False)
    cfg["attacks"].update(type="scaling", malicious_fraction=0.2, placement="spread")
    attack, _ = attack_from_config(cfg, data.client_groups, data.benign_id)
    res = run_federated(data, cfg, get_aggregator("multikrum", cfg), attack=attack,
                        rounds=3, verbose=False)
    assert all(r["screen_tpr"] == 1.0 for r in res["history"])   # all 4 attackers rejected


def test_train_local_exact_steps(cfg):
    """Root update g0 runs a fixed number of SGD steps (matching clients), not epochs."""
    import numpy as np
    from src.fl.client import steps_per_round, train_local
    from src.models.mlp import build_model

    assert steps_per_round(1000, cfg["training"]) == 2 * 4        # 2 epochs × ceil(1000/256)
    calls = {"n": 0}
    m = build_model(39, 8, cfg["model"])
    m.register_forward_hook(lambda *a: calls.__setitem__("n", calls["n"] + 1))
    X = np.random.default_rng(0).normal(size=(100, 39)).astype(np.float32)
    train_local(m, X, np.arange(100) % 8, cfg["training"], 8, seed=0, steps=7)
    assert calls["n"] == 7                                        # 7 steps over a 1-batch dataset
