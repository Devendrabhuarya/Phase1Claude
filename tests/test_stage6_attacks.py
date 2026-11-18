"""Stage 6 tests: every attack and the malicious-client placement."""

from collections import Counter

import numpy as np
import pytest
import torch

from src.aggregators import get_aggregator
from src.attacks import attack_from_config
from src.attacks.attacks import (ALIE, GaussianNoise, LabelFlip, MinMax, Scaling, SignFlip,
                                 alie_z, get_attack)
from src.attacks.placement import select_malicious
from src.fl.pipeline import build_fl_data
from src.fl.server import run_federated
from src.utils.config import load_config
from src.utils.seed import set_seed

GROUPS = ["A"] * 7 + ["B"] * 7 + ["C"] * 6


@pytest.fixture
def cfg():
    c = load_config("ciciot2023")
    set_seed(c["seed"])
    return c


@pytest.fixture
def updates():
    g = torch.Generator().manual_seed(0)
    u = torch.randn(10, 50, generator=g) * 0.1 + 0.05
    mask = torch.zeros(10, dtype=torch.bool)
    mask[[2, 7]] = True
    return u, mask


RNG = np.random.default_rng(0)

# ------------------------------------------------------------------ placement


@pytest.mark.parametrize("fraction,n", [(0.0, 0), (0.1, 2), (0.2, 4), (0.3, 6), (0.4, 8)])
def test_placement_counts(fraction, n):
    for placement in ("spread", "concentrated"):
        mal = select_malicious(GROUPS, fraction, placement, seed=1)
        assert len(mal) == n and len(set(mal)) == n


def test_spread_is_balanced():
    mal = select_malicious(GROUPS, 0.3, "spread", seed=1)
    assert Counter(GROUPS[i] for i in mal) == {"A": 2, "B": 2, "C": 2}
    mal = select_malicious(GROUPS, 0.4, "spread", seed=1)
    counts = Counter(GROUPS[i] for i in mal)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_concentrated_fills_one_zone():
    mal = select_malicious(GROUPS, 0.3, "concentrated", seed=1, concentrated_zone="A")
    assert {GROUPS[i] for i in mal} == {"A"}
    mal = select_malicious(GROUPS, 0.4, "concentrated", seed=1, concentrated_zone="A")
    counts = Counter(GROUPS[i] for i in mal)
    assert counts["A"] == 7 and sum(counts.values()) == 8          # zone full + 1 overflow
    mal = select_malicious(GROUPS, 0.2, "concentrated", seed=1, concentrated_zone="C")
    assert {GROUPS[i] for i in mal} == {"C"}


def test_placement_reproducible_and_seed_dependent():
    a = select_malicious(GROUPS, 0.2, "spread", seed=1)
    assert a == select_malicious(GROUPS, 0.2, "spread", seed=1)
    assert any(select_malicious(GROUPS, 0.2, "spread", seed=s) != a for s in range(2, 10))


def test_bad_placement_raises():
    with pytest.raises(ValueError):
        select_malicious(GROUPS, 0.2, "everywhere", seed=1)

# ------------------------------------------------------------------ attacks


def test_label_flip():
    X, y = np.ones((6, 3)), np.array([0, 1, 2, 3, 0, 5])
    X2, y2 = LabelFlip([0], benign_id=0).poison_data(0, X, y)
    assert (y2 == 0).all() and X2 is X
    assert list(y) == [0, 1, 2, 3, 0, 5]                     # input untouched


def test_sign_flip(updates):
    u, m = updates
    out = SignFlip([2, 7], scale=2.0).poison_updates(u, m, RNG)
    assert torch.allclose(out[m], -2.0 * u[m]) and torch.equal(out[~m], u[~m])


def test_gaussian(updates):
    u, m = updates
    out = GaussianNoise([2, 7], std=1.0).poison_updates(u, m, np.random.default_rng(0))
    assert torch.equal(out[~m], u[~m])
    assert 0.7 < float(out[m].std()) < 1.3 and abs(float(out[m].mean())) < 0.3


def test_scaling(updates):
    u, m = updates
    out = Scaling([2, 7], factor=10).poison_updates(u, m, RNG)
    assert torch.allclose(out[m], 10 * u[m]) and torch.equal(out[~m], u[~m])


def test_alie(updates):
    u, m = updates
    att = ALIE([2, 7])
    out = att.poison_updates(u, m, RNG)
    benign = u[~m]
    mu, sd = benign.mean(0), benign.std(0, unbiased=False)
    assert torch.equal(out[2], out[7])                       # colluding: identical
    assert torch.allclose(out[2], mu - att.last_z * sd)
    assert torch.equal(out[~m], u[~m])
    assert alie_z(20, 4) == pytest.approx(0.3853, abs=1e-3)  # s=7, Φ⁻¹(0.65)


def test_minmax(updates):
    u, m = updates
    att = MinMax([2, 7], gamma_init=10.0, tol=1e-5)
    out = att.poison_updates(u, m, RNG)
    benign = u[~m]
    max_pair = torch.cdist(benign, benign).max()
    assert torch.equal(out[2], out[7])
    assert (benign - out[2]).norm(dim=1).max() <= max_pair + 1e-5   # stays within benign spread
    assert att.last_gamma > 0
    # a slightly larger gamma would break the constraint (gamma is maximal)
    mu, sd = benign.mean(0), benign.std(0, unbiased=False)
    bigger = mu - (att.last_gamma * 1.01 + 1e-3) * sd
    assert (benign - bigger).norm(dim=1).max() > max_pair


def test_get_attack_factory(cfg):
    for kind, cls in [("label_flip", LabelFlip), ("sign_flip", SignFlip), ("gaussian", GaussianNoise),
                      ("scaling", Scaling), ("alie", ALIE), ("minmax", MinMax)]:
        cfg["attacks"]["type"] = kind
        assert isinstance(get_attack(cfg, [1], benign_id=0), cls)
    cfg["attacks"]["type"] = "none"
    assert get_attack(cfg, [1], 0) is None
    cfg["attacks"]["type"] = "sign_flip"
    assert get_attack(cfg, [], 0) is None
    cfg["attacks"]["type"] = "bogus"
    with pytest.raises(ValueError):
        get_attack(cfg, [1], 0)

# ------------------------------------------------------------------ integration


def test_attacks_hurt_fedavg(cfg):
    """Sign flipping at 40 % should clearly damage undefended FedAvg; label flip raises ASR."""
    data, _, _ = build_fl_data(cfg, synthetic=True, verbose=False)
    clean = run_federated(data, cfg, get_aggregator("fedavg"), rounds=6, verbose=False)["final"]

    cfg["attacks"].update(type="sign_flip", malicious_fraction=0.4, placement="spread")
    attack, mal = attack_from_config(cfg, data.client_groups, data.benign_id)
    assert len(mal) == 8
    res = run_federated(data, cfg, get_aggregator("fedavg"), attack=attack, rounds=6, verbose=False)
    assert res["final"]["accuracy"] < clean["accuracy"] - 0.1
    assert "screen_tpr" in res["history"][0]                  # FedAvg flags nobody: TPR 0
    assert res["history"][0]["screen_tpr"] == 0.0

    cfg["attacks"].update(type="label_flip")
    attack, _ = attack_from_config(cfg, data.client_groups, data.benign_id)
    lf = run_federated(data, cfg, get_aggregator("fedavg"), attack=attack, rounds=6, verbose=False)["final"]
    assert lf["attack_success_rate"] > clean["attack_success_rate"]
