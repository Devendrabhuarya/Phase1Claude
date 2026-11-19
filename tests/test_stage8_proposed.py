"""Stage 8 tests: ABAC (PyCasbin), reputation, clipping, cohorts, intra-/inter-cohort
screening, ablation variants and an end-to-end run under sign flipping."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sklearn.metrics import adjusted_rand_score

from src.abac.admission import AdmissionController, client_name
from src.abac.policy import AbacPolicy
from src.abac.registry import AttributeRegistry, ClientAttributes, firmware_to_int
from src.aggregators import get_aggregator
from src.aggregators.cohorts import cosine_distance_matrix, form_cohorts, median_norm_clip
from src.aggregators.proposed import VARIANTS, capped_weights, reputation_scores
from src.attacks import attack_from_config
from src.fl.methods import build_method
from src.fl.pipeline import build_fl_data
from src.fl.server import run_federated
from src.reputation.reputation import ReputationManager
from src.utils.config import load_config
from src.utils.seed import set_seed


@pytest.fixture
def cfg():
    c = load_config("ciciot2023")
    set_seed(c["seed"])
    c["aggregator"]["name"] = "proposed"
    return c


# ------------------------------------------------------------------ ABAC


def test_firmware_to_int():
    assert firmware_to_int("2.1.3") == 20103
    assert firmware_to_int("1.4") < firmware_to_int("2.0.0") < firmware_to_int("2.0.1")


def test_registry_unknown_identity_is_unregistered():
    reg = AttributeRegistry()
    reg.register("client00")
    assert reg.lookup("client00").registered
    assert not reg.lookup("ghost").registered and "ghost" not in reg


def test_policy_each_condition():
    pol = AbacPolicy("2.0.0", freshness_window=3, tau=0.3)
    good = ClientAttributes("c", registered=True, firmware_version="2.1.0", last_seen=10,
                            reputation=0.5, quarantined_until=0)
    assert pol.permit(good, 10) and pol.deny_reasons(good, 10) == []
    bad = {
        "registered": False, "firmware_version": "1.9.9", "last_seen": 6,
        "reputation": 0.29, "quarantined_until": 10,
    }
    for field, value in bad.items():
        a = copy.copy(good)
        setattr(a, field, value)
        assert not pol.permit(a, 10), field
        assert len(pol.deny_reasons(a, 10)) == 1, field
    edge = copy.copy(good)
    edge.reputation, edge.last_seen = 0.3, 7                  # reputation == tau, age == window
    assert pol.permit(edge, 10)


def test_casbin_agrees_with_explanations():
    pol = AbacPolicy("2.0.0", 3, 0.3)
    rng = np.random.default_rng(0)
    for _ in range(300):
        a = ClientAttributes("c", registered=bool(rng.integers(2)),
                             firmware_version=f"{rng.integers(1, 3)}.{rng.integers(0, 3)}.0",
                             last_seen=int(rng.integers(0, 11)), reputation=float(rng.random()),
                             quarantined_until=int(rng.integers(0, 15)))
        assert pol.permit(a, 10) == (pol.deny_reasons(a, 10) == [])


def test_admission_blocks_intruders_admits_clients(cfg):
    adm = AdmissionController(cfg, n_clients=20)
    res = adm.admit(1, list(range(20)))
    assert res.admitted == list(range(20))
    assert res.n_intruders == 4 and res.intruders_blocked == 4
    reasons = {k: v["reason"] for k, v in res.decisions.items() if k.startswith("intruder")}
    assert any("unregistered" in r for r in reasons.values())
    assert any("firmware" in r for r in reasons.values())
    assert any("stale" in r for r in reasons.values())
    assert adm.block_rate() == 1.0


def test_quarantine_closed_loop(cfg):
    """Low scores -> reputation < tau -> quarantined 5 rounds -> back on probation (r = tau)."""
    adm = AdmissionController(cfg, n_clients=3)
    newly, rnd = [], 0
    while not newly and rnd < 20:
        rnd += 1
        adm.admit(rnd, [0, 1, 2])
        newly = adm.after_round(rnd, {"round_scores": {0: 0.9, 1: 0.9, 2: 0.0}})
    assert newly == [client_name(2)]
    q_round = rnd                                             # 0.5 -> 0.4 -> 0.32 -> 0.256
    assert q_round == 3
    for r in range(q_round + 1, q_round + 6):                # denied for 5 rounds
        res = adm.admit(r, [0, 1, 2])
        assert 2 not in res.admitted and "quarantined" in res.decisions[client_name(2)]["reason"]
    res = adm.admit(q_round + 6, [0, 1, 2])                   # released on probation
    assert 2 in res.admitted and res.released == [client_name(2)]
    assert adm.registry.get(client_name(2)).reputation == pytest.approx(0.3)


def test_reputation_ema():
    reg = AttributeRegistry()
    reg.register("a", reputation=0.5)
    rm = ReputationManager(reg, beta=0.8, threshold=0.3, quarantine_rounds=5, probation_value=0.3)
    rm.update(1, {"a": 1.0})
    assert reg.get("a").reputation == pytest.approx(0.6)
    rm.update(2, {"a": 0.0})
    assert reg.get("a").reputation == pytest.approx(0.48)


def test_quarantine_cap_holds_extra_clients_at_tau():
    """With max 2 quarantined, 4 clients falling below tau: the 2 lowest are quarantined,
    the other 2 are held at tau (still admissible) instead of emptying the round."""
    reg = AttributeRegistry()
    for c in "abcde":
        reg.register(c, reputation=0.5)
    rm = ReputationManager(reg, beta=0.8, threshold=0.3, quarantine_rounds=5,
                           probation_value=0.3, max_quarantined=2)
    newly = []
    for rnd in range(1, 5):
        newly += rm.update(rnd, {"a": 0.0, "b": 0.05, "c": 0.1, "d": 0.12, "e": 1.0})
    assert sorted(newly) == ["a", "b"] and rm.quarantined == ["a", "b"]
    assert reg.get("c").reputation >= 0.3 and reg.get("d").reputation >= 0.3
    assert reg.get("e").reputation > 0.5


def test_quarantine_cap_from_config(cfg):
    assert AdmissionController(cfg, n_clients=20).reputation.max_quarantined == 8
    cfg["reputation"]["max_quarantine_fraction"] = None
    assert AdmissionController(cfg, n_clients=20).reputation.max_quarantined is None


def test_no_feedback_without_reputation(cfg):
    adm = AdmissionController(cfg, n_clients=2, use_reputation=False)
    for r in range(1, 10):
        adm.admit(r, [0, 1])
        assert adm.after_round(r, {"round_scores": {0: 0.0, 1: 0.0}}) == []
    assert adm.registry.get(client_name(0)).reputation == 0.5

# ------------------------------------------------------------------ clipping + cohorts


def test_median_norm_clip():
    u = torch.stack([torch.ones(4) * s for s in (1.0, 2.0, 3.0, 100.0, 2.5)])
    clipped, med, idx = median_norm_clip(u)
    norms = clipped.norm(dim=1)
    assert med == pytest.approx(5.0)                          # norm of 2.5·1 in 4-D
    assert (norms <= med + 1e-5).all() and set(idx) == {2, 3}
    assert torch.allclose(clipped[0], u[0])                   # small update untouched
    assert torch.allclose(clipped[3] / clipped[3].norm(), u[3] / u[3].norm())   # direction kept


def test_cosine_distance_matrix():
    d = cosine_distance_matrix(torch.tensor([[1.0, 0], [2.0, 0], [0, 1.0], [-1.0, 0]]))
    assert d[0, 1] == pytest.approx(0) and d[0, 2] == pytest.approx(1) and d[0, 3] == pytest.approx(2)
    assert np.allclose(d, d.T) and np.all(np.diag(d) == 0)


def _three_groups(sizes=(7, 7, 6), dim=30, noise=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    centers = torch.eye(dim)[:3] * 5
    feats, truth = [], []
    for k, n in enumerate(sizes):
        feats.append(centers[k] + noise * torch.randn(n, dim, generator=g))
        truth += [k] * n
    return torch.cat(feats), np.array(truth)


def test_form_cohorts_recovers_groups():
    feats, truth = _three_groups()
    labels, info = form_cohorts(feats, [2, 3], min_size=6)
    assert info["k_chosen"] == 3 and adjusted_rand_score(truth, labels) == 1.0


def test_small_cluster_is_merged():
    feats, truth = _three_groups(sizes=(8, 9, 3))
    labels, info = form_cohorts(feats, [2, 3], min_size=6)
    assert info["n_cohorts"] == 2
    assert np.bincount(labels).min() >= 6


def test_too_few_clients_single_cohort():
    feats, _ = _three_groups(sizes=(4, 4, 3))
    labels, info = form_cohorts(feats, [2, 3], min_size=6)
    assert info["n_cohorts"] == 1 and (labels == 0).all()

# ------------------------------------------------------------------ inter-cohort weights


def test_capped_weights():
    assert np.allclose(capped_weights([0.8, 0.2], 0.5), [0.5, 0.5])
    assert np.allclose(capped_weights([6, 3, 1], 0.5), [0.5, 0.375, 0.125])
    assert np.allclose(capped_weights([1, 1, 1], 0.5), [1 / 3] * 3)
    assert np.allclose(capped_weights([5.0], 0.5), [1.0])                # infeasible -> equal
    w = capped_weights([10, 1, 1, 1], 0.5)
    assert w.max() <= 0.5 + 1e-9 and w.sum() == pytest.approx(1.0)

# ------------------------------------------------------------------ proposed aggregator


def _ctx(n, g0, final_layer=slice(None)):
    return {"client_ids": list(range(n)), "n_samples": torch.ones(n), "round": 1,
            "root_update": lambda: g0, "final_layer": final_layer}


def test_proposed_rejects_sign_flippers(cfg):
    """3 honest groups (7/7/6) near g0, two of zone-A clients sign-flipped."""
    dim = 30
    feats, truth = _three_groups(dim=dim, noise=0.3)
    g0 = torch.eye(dim)[:3].sum(0)                              # every honest group is aligned with g0
    updates = feats.clone()
    updates[[0, 1]] = -updates[[0, 1]]                          # attackers
    agg = get_aggregator("proposed", cfg)
    new, info = agg.aggregate(updates, torch.zeros(dim), _ctx(20, g0))
    assert 0 not in info["selected"] and 1 not in info["selected"]
    assert info["round_scores"][0] == 0.0 and info["round_scores"][1] == 0.0
    assert torch.nn.functional.cosine_similarity(new, g0, dim=0) > 0.8
    assert len(info["cohort_weights"]) >= 2
    assert max(info["cohort_weights"].values()) <= 0.5 + 1e-9
    assert sum(info["cohort_weights"].values()) == pytest.approx(1.0)


def test_relative_scores_keep_low_cosine_honest_clients():
    """Honest non-IID clients with cosine ~0.15-0.25 to g0 must not be quarantined
    (they were under raw scores); a TS=0 client and dropped/unselected ones score 0."""
    # 0-2, 4 selected; 3 anti-aligned (TS 0); 5 honest but dropped by Multi-Krum;
    # 6 in a cohort that was dropped as root-misaligned.
    ts = {0: 0.15, 1: 0.2, 2: 0.25, 3: 0.0, 4: 0.3, 5: 0.2, 6: 0.25}
    selected, kept = [0, 1, 2, 4], {0, 1, 2, 3, 4, 6}
    ok = {c: c != 6 for c in ts}
    raw = {c: (ts[c] if c in kept else 0.0) for c in ts}
    rel = reputation_scores(raw, ts, selected, kept, ok, mode="relative")
    assert rel[1] == pytest.approx(0.2 / 0.225) and rel[4] == 1.0
    assert rel[5] == pytest.approx(0.5 * 0.2 / 0.225)   # Multi-Krum drop -> half score
    assert rel[3] == rel[6] == 0.0
    assert reputation_scores(raw, ts, selected, kept, ok, mode="raw") is raw

    def ema(s, rounds=10, r=0.5):
        for _ in range(rounds):
            r = 0.8 * r + 0.2 * s
        return r
    assert ema(raw[0]) < 0.3 and ema(raw[5]) < 0.3  # raw: honest clients fall below tau
    assert ema(rel[0]) > 0.3 and ema(rel[5]) > 0.3  # relative: they stay admitted
    assert ema(rel[3]) < 0.3                        # attacker still pushed out


def test_captured_cohort_is_dropped(cfg):
    """A cohort made only of attackers (all anti-aligned with g0) gets zero trust and is dropped."""
    dim = 30
    feats, truth = _three_groups(dim=dim, noise=0.1)
    g0 = torch.eye(dim)[:3].sum(0)
    updates = feats.clone()
    captured = np.flatnonzero(truth == 2)                       # the 6-client group
    updates[captured] = -updates[captured]
    _, info = get_aggregator("proposed", cfg).aggregate(updates, torch.zeros(dim), _ctx(20, g0))
    assert len(info["dropped_cohorts"]) >= 1
    assert not set(captured.tolist()) & set(info["selected"])


@pytest.mark.parametrize("variant", ["V1", "V2"])
def test_v1_v2_are_fedavg_like(cfg, variant):
    cfg["ablation"]["variant"] = variant
    u = torch.tensor([[1.0, 1.0], [1.0, 1.0], [100.0, 100.0]])
    out, info = get_aggregator("proposed", cfg).aggregate(u, torch.zeros(2), _ctx(3, torch.ones(2)))
    if variant == "V1":
        assert torch.allclose(out, u.mean(0))
    else:                                                       # clipped to median norm first
        assert torch.allclose(out, torch.ones(2))
    assert info["selected"] == [0, 1, 2]


def test_variant_flags_and_build_method(cfg):
    assert VARIANTS["V5"]["reputation"] and not VARIANTS["V4"]["reputation"]
    agg, adm = build_method(cfg, 20)
    assert agg.name == "proposed" and adm is not None and adm.use_reputation
    cfg["ablation"]["variant"] = "V4"
    assert not build_method(cfg, 20)[1].use_reputation
    cfg["aggregator"]["name"] = "fedavg"
    assert build_method(cfg, 20)[1] is None

# ------------------------------------------------------------------ server integration


def test_intruders_that_pass_abac_are_aggregated(cfg):
    """If the policy were misconfigured, admitted intruders' updates do reach aggregation."""
    cfg["abac"]["stale_firmware_version"] = "9.9.9"            # 'stale' intruders now pass
    cfg["ablation"]["variant"] = "V1"
    data, _, _ = build_fl_data(cfg, synthetic=True, verbose=False)
    agg, adm = build_method(cfg, data.n_clients)
    res = run_federated(data, cfg, agg, admission=adm, rounds=1, verbose=False)
    row = res["history"][0]
    assert row["intruders"] == 4 and row["intruders_blocked"] == 3
    assert any(isinstance(c, str) for c in row["info"]["selected"])


def test_end_to_end_beats_fedavg_under_sign_flip(cfg):
    """Stage 8 'done when': full method (V5) beats FedAvg under sign flipping (synthetic)."""
    data, _, _ = build_fl_data(cfg, synthetic=True, verbose=False)
    cfg["attacks"].update(type="sign_flip", malicious_fraction=0.2, placement="spread")
    attack, mal = attack_from_config(cfg, data.client_groups, data.benign_id)

    fed = copy.deepcopy(cfg)
    fed["aggregator"]["name"] = "fedavg"
    base = run_federated(data, fed, get_aggregator("fedavg", fed), attack=attack, rounds=10,
                         verbose=False)["final"]
    agg, adm = build_method(cfg, data.n_clients)
    res = run_federated(data, cfg, agg, attack=attack, admission=adm, rounds=10, verbose=False)
    final = res["final"]
    print(f"V5 acc={final['accuracy']:.3f} F1={final['macro_f1']:.3f} | "
          f"FedAvg acc={base['accuracy']:.3f} F1={base['macro_f1']:.3f}")
    assert final["macro_f1"] > base["macro_f1"] + 0.05
    assert final["accuracy"] >= base["accuracy"] - 0.01
    quarantined = {c for r in res["history"] for c in r["newly_quarantined"]}
    assert set(mal) <= quarantined                              # every attacker quarantined
    assert all(r["intruders_blocked"] == r["intruders"] for r in res["history"])
    assert "cohort_ari" in res["history"][0]
