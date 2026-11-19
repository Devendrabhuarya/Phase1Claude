"""Stage 9 tests: Merkle tree, hash-chained ledger, verify(), tamper test, server wiring."""

import copy
import hashlib
import json

import pytest
import torch

from src.attacks import attack_from_config
from src.fl.methods import build_method
from src.fl.pipeline import build_fl_data
from src.fl.server import run_federated
from src.ledger import HashChainLedger, build_ledger, hash_tensor, merkle_proof, merkle_root, \
    verify_merkle_proof
from src.ledger.ledger import BLOCK_FIELDS, GENESIS_PREV_HASH, compute_block_hash
from src.utils.config import load_config
from src.utils.seed import set_seed

H = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


@pytest.fixture
def cfg():
    c = load_config("ciciot2023")
    set_seed(c["seed"])
    return c


def _record(rnd, n=3, model_value=0.0):
    return {
        "round": rnd,
        "abac_decisions": {f"client{i:02d}": {"permit": True, "reason": "ok"} for i in range(n)},
        "update_hashes": {f"client{i:02d}": hash_tensor(torch.full((4,), float(i + rnd)))
                          for i in range(n)},
        "cohorts": {f"client{i:02d}": i % 2 for i in range(n)},
        "screening": {"selected": [f"client{i:02d}" for i in range(n)],
                      "trust_scores": {f"client{i:02d}": 0.5 for i in range(n)}},
        "reputation": {"scores": {f"client{i:02d}": 0.5 for i in range(n)}, "quarantined": []},
        "global_model_hash": hash_tensor(torch.full((4,), model_value + rnd)),
    }


@pytest.fixture
def chain():
    led = HashChainLedger()
    for r in range(1, 6):
        led.append(_record(r))
    return led


# ------------------------------------------------------------------ Merkle


def test_merkle_root_small_cases():
    a, b, c = H(b"a"), H(b"b"), H(b"c")
    pair = lambda x, y: H(bytes.fromhex(x) + bytes.fromhex(y))  # noqa: E731
    assert merkle_root([]) == H(b"")
    assert merkle_root([a]) == a
    assert merkle_root([a, b]) == pair(a, b)
    assert merkle_root([a, b, c]) == pair(pair(a, b), pair(c, c))     # odd: last is duplicated


def test_merkle_root_changes_with_any_leaf_or_order():
    leaves = [H(bytes([i])) for i in range(7)]
    root = merkle_root(leaves)
    assert merkle_root(leaves[::-1]) != root
    assert merkle_root(leaves[:-1]) != root
    changed = list(leaves)
    changed[3] = H(b"x")
    assert merkle_root(changed) != root


@pytest.mark.parametrize("n", [1, 2, 5, 8, 21])
def test_merkle_proofs(n):
    leaves = [H(bytes([i])) for i in range(n)]
    root = merkle_root(leaves)
    for i in range(n):
        proof = merkle_proof(leaves, i)
        assert verify_merkle_proof(leaves[i], proof, root)
        assert not verify_merkle_proof(H(b"fake"), proof, root)


def test_hash_tensor_dtype_independent():
    t = torch.tensor([0.1, -2.0, 3.5])
    assert hash_tensor(t) == hash_tensor(t.double()) == hash_tensor(t.numpy())
    assert hash_tensor(t) != hash_tensor(t + 1e-3)


# ------------------------------------------------------------------ chain


def test_blocks_are_linked(chain):
    assert len(chain) == 5
    assert chain.blocks[0]["prev_hash"] == GENESIS_PREV_HASH
    for prev, b in zip(chain.blocks, chain.blocks[1:]):
        assert b["prev_hash"] == prev["block_hash"]
    for i, b in enumerate(chain.blocks):
        assert b["index"] == i and set(BLOCK_FIELDS) <= set(b)
        assert b["block_hash"] == compute_block_hash(b)
        assert b["update_merkle_root"] == merkle_root(list(b["update_hashes"].values()))
    assert chain.head() == chain.blocks[-1]["block_hash"]


def test_verify_valid_chain(chain):
    v = chain.verify()
    assert v["valid"] and v["errors"] == [] and v["first_invalid"] is None and v["n_blocks"] == 5


@pytest.mark.parametrize("edit", [
    lambda b: b["screening"]["trust_scores"].__setitem__("client01", 1.0),
    lambda b: b["abac_decisions"].__setitem__("client02", {"permit": False, "reason": "x"}),
    lambda b: b["cohorts"].__setitem__("client00", 7),
    lambda b: b["reputation"]["quarantined"].append("client01"),
    lambda b: b.__setitem__("global_model_hash", H(b"other model")),
    lambda b: b.__setitem__("round", 99),
])
def test_tamper_any_field_fails_verification(chain, edit):
    """Stage 9 'done when': modifying one past record makes verify() fail at that block."""
    edit(chain.blocks[2])
    v = chain.verify()
    assert not v["valid"] and v["first_invalid"] == 2


def test_tamper_with_rehash_breaks_next_link(chain):
    """Recomputing the edited block's own hash is caught by the next block's prev_hash."""
    b = chain.blocks[2]
    b["screening"]["trust_scores"]["client01"] = 1.0
    b["block_hash"] = compute_block_hash(b)
    v = chain.verify()
    assert not v["valid"] and v["first_invalid"] == 3
    assert any("prev_hash" in msg for _, msg in v["errors"])


def test_tamper_update_hash_breaks_merkle_root(chain):
    chain.blocks[1]["update_hashes"]["client00"] = H(b"swapped update")
    errors = chain.verify()["errors"]
    assert any(i == 1 and "merkle" in msg.lower() for i, msg in errors)


def test_deleting_or_reordering_blocks_detected(chain):
    deleted = copy.deepcopy(chain)
    del deleted.blocks[2]
    assert not deleted.verify()["valid"]
    swapped = copy.deepcopy(chain)
    swapped.blocks[1], swapped.blocks[2] = swapped.blocks[2], swapped.blocks[1]
    assert not swapped.verify()["valid"]


def test_append_returns_copy(chain):
    blk = chain.append(_record(6))
    blk["round"] = -1
    assert chain.verify()["valid"]


def test_query(chain):
    assert [b["round"] for b in chain.query(round=3)] == [3]
    assert len(chain.query()) == 5
    trail = chain.query(client="client01")
    assert [e["round"] for e in trail] == [1, 2, 3, 4, 5]
    assert trail[0]["selected"] and trail[0]["abac"]["permit"] and trail[0]["cohort"] == 1
    assert trail[0]["trust_score"] == 0.5 and not trail[0]["quarantined"]


def test_save_load_roundtrip(chain, tmp_path):
    p = tmp_path / "ledger.jsonl"
    chain.save(p)
    loaded = HashChainLedger.load(p)
    assert loaded.verify()["valid"] and loaded.head() == chain.head()
    lines = p.read_text(encoding="utf-8").splitlines()
    tampered = json.loads(lines[1])
    tampered["abac_decisions"]["client00"]["permit"] = False
    lines[1] = json.dumps(tampered)
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    assert HashChainLedger.load(p).verify()["first_invalid"] == 1


def test_streaming_file_matches_memory(tmp_path):
    p = tmp_path / "led.jsonl"
    led = HashChainLedger(path=p)
    for r in range(1, 4):
        led.append(_record(r))
    assert HashChainLedger.load(p).head() == led.head()


def test_numpy_and_int_keys_are_normalised():
    import numpy as np
    led = HashChainLedger()
    rec = _record(1)
    rec["cohorts"] = {0: np.int64(1)}
    rec["screening"] = {"x": np.float32(0.25), "nan": float("nan")}
    blk = led.append(rec)
    assert blk["cohorts"] == {"0": 1} and blk["screening"]["nan"] is None
    assert led.verify()["valid"]


def test_build_ledger_respects_config(cfg):
    assert isinstance(build_ledger(cfg), HashChainLedger)
    cfg["ledger"]["enabled"] = False
    assert build_ledger(cfg) is None


# ------------------------------------------------------------------ server wiring


def test_server_writes_one_block_per_round(cfg):
    cfg["aggregator"]["name"] = "proposed"
    data, _, _ = build_fl_data(cfg, synthetic=True, verbose=False)
    cfg["attacks"].update(type="sign_flip", malicious_fraction=0.2, placement="spread")
    attack, mal = attack_from_config(cfg, data.client_groups, data.benign_id)
    agg, adm = build_method(cfg, data.n_clients)
    led = build_ledger(cfg)
    res = run_federated(data, cfg, agg, attack=attack, admission=adm, ledger=led, rounds=3,
                        verbose=False)

    assert len(led) == 3 and led.verify()["valid"] and res["ledger"] is led
    for row, blk in zip(res["history"], led.blocks):
        assert row["block_hash"] == blk["block_hash"] and row["ledger_append_s"] >= 0
        assert any(k.startswith("intruder-") and not v["permit"]
                   for k, v in blk["abac_decisions"].items())          # blocked intruders logged
        assert len(blk["update_hashes"]) == row["n_admitted"]          # admitted updates only
        assert blk["cohorts"] and "trust_scores" in blk["screening"]
        assert set(blk["reputation"]["scores"]) == {f"client{i:02d}" for i in range(data.n_clients)}
    assert led.blocks[-1]["global_model_hash"] == hash_tensor(res["weights"])
    # off-chain: no full vectors on the ledger
    assert max(len(json.dumps(b)) for b in led.blocks) < 20_000


def test_ledger_does_not_change_training(cfg):
    """The ledger only records: results are identical with and without it."""
    cfg["aggregator"]["name"] = "fedavg"
    data, _, _ = build_fl_data(cfg, synthetic=True, verbose=False)
    from src.aggregators import get_aggregator
    a = run_federated(data, cfg, get_aggregator("fedavg", cfg), rounds=2, verbose=False)
    led = build_ledger(cfg)
    b = run_federated(data, cfg, get_aggregator("fedavg", cfg), rounds=2, verbose=False, ledger=led)
    assert torch.equal(a["weights"], b["weights"])
    assert led.blocks[0]["abac_decisions"] == {} and led.blocks[0]["reputation"] == {}
    assert led.verify()["valid"]
