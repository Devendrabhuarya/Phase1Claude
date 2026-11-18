"""Stage 3 tests: zone-based Dirichlet partitioning."""

import numpy as np
import pytest

from src.data.partition import (_dominance, client_class_table, partition_from_config,
                                solve_home_share, zone_class_table)
from src.data.synthetic import CLASS_ORDER, SYNTHETIC_FL_COUNTS
from src.utils.config import load_config


def _labels(counts, seed=0):
    y = np.concatenate([np.full(n, CLASS_ORDER.index(c)) for c, n in counts.items()])
    return np.random.default_rng(seed).permutation(y)


@pytest.fixture
def cfg():
    return load_config("ciciot2023")


@pytest.mark.parametrize("alpha", [0.5, 0.1])
def test_partition_basic_properties(cfg, alpha, capsys):
    y = _labels(SYNTHETIC_FL_COUNTS)
    part = partition_from_config(y, CLASS_ORDER, cfg, alpha=alpha)
    table = client_class_table(part, y, CLASS_ORDER)
    print(table.to_string())                                     # per-client table printed
    assert part.n_clients == 20
    assert part.client_groups.count("A") == 7 and part.client_groups.count("B") == 7
    assert part.client_groups.count("C") == 6
    assert (table["total"] > 0).all()                            # no client empty
    assert (table["Benign"] > 0).all()                           # every client has Benign
    assert table["benign_frac"].between(0.28, 0.52).all()        # 30-50 % Benign (rounding)
    all_idx = np.concatenate(part.client_indices)
    assert len(all_idx) == len(np.unique(all_idx))               # no row shared
    for idx, tr, va in zip(part.client_indices, part.train_indices, part.val_indices):
        assert set(tr) | set(va) == set(idx) and not set(tr) & set(va)
    assert "client00" in capsys.readouterr().out


def test_every_class_in_federation_and_zones_dominant(cfg):
    y = _labels(SYNTHETIC_FL_COUNTS)
    part = partition_from_config(y, CLASS_ORDER, cfg)
    zt = zone_class_table(part, y, CLASS_ORDER)
    assert (zt[CLASS_ORDER].sum(axis=0) > 0).all()
    attack = [c for c in CLASS_ORDER if c != "Benign"]
    for zone, dom in [("A", ["DDoS", "DoS"]), ("B", ["Recon", "Spoofing", "BruteForce"]),
                      ("C", ["Mirai", "Web"])]:
        frac = zt.loc[zone, dom].sum() / zt.loc[zone, attack].sum()
        assert frac > 0.55, (zone, frac)
    assert abs(np.mean(list(part.info["zone_dominance"].values())) - 0.75) < 0.01


def test_home_share_bisection():
    zones = ["A", "B"]
    w = np.array([0.5, 0.5])
    counts = {"x": 100, "y": 100}
    homes = {"x": "A", "y": "B"}
    s = solve_home_share(counts, homes, zones, w, 0.75)
    assert abs(s - 0.75) < 1e-6
    assert np.allclose(_dominance(s, counts, homes, zones, w), 0.75)


def test_partition_deterministic(cfg):
    y = _labels(SYNTHETIC_FL_COUNTS)
    p1 = partition_from_config(y, CLASS_ORDER, cfg, verbose=False)
    p2 = partition_from_config(y, CLASS_ORDER, cfg, verbose=False)
    assert all(np.array_equal(a, b) for a, b in zip(p1.client_indices, p2.client_indices))
    cfg["seed"] += 1
    p3 = partition_from_config(y, CLASS_ORDER, cfg, verbose=False)
    assert not all(np.array_equal(a, b) for a, b in zip(p1.client_indices, p3.client_indices))


def test_low_alpha_more_skewed(cfg):
    """Clients under alpha=0.1 have more uneven class mixes than under alpha=0.5."""
    y = _labels(SYNTHETIC_FL_COUNTS)

    def mean_entropy(alpha):
        t = client_class_table(partition_from_config(y, CLASS_ORDER, cfg, alpha=alpha, verbose=False),
                               y, CLASS_ORDER)[CLASS_ORDER[1:]].to_numpy(float)
        p = t / t.sum(axis=1, keepdims=True)
        return float(np.mean(-(p * np.log(p + 1e-12)).sum(axis=1)))

    assert mean_entropy(0.1) < mean_entropy(0.5)


def test_insufficient_benign_subsamples_attack(cfg, capsys):
    counts = dict(SYNTHETIC_FL_COUNTS, Benign=500)
    y = _labels(counts)
    part = partition_from_config(y, CLASS_ORDER, cfg)
    table = client_class_table(part, y, CLASS_ORDER)
    assert part.info["attack_rows_kept"] < 1.0
    assert table["benign_frac"].between(0.28, 0.52).all()
    assert "WARNING" in capsys.readouterr().out


def test_bad_zone_config_raises(cfg):
    cfg["partition"]["zones"]["C"]["dominant"] = ["Mirai", "DDoS"]   # DDoS already in A
    with pytest.raises(ValueError, match="two zones"):
        partition_from_config(_labels(SYNTHETIC_FL_COUNTS), CLASS_ORDER, cfg)
