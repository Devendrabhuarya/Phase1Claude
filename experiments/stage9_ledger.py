"""Stage 9: audit ledger demo — run FL with the ledger, verify it, then tamper with it.

    python experiments/stage9_ledger.py --synthetic --rounds 5
    python experiments/stage9_ledger.py --rounds 10 --attack sign_flip --fraction 0.2

Steps:
  1. Run the proposed method (V5) with a hash-chained ledger (one block per round).
  2. verify() the chain -> valid.
  3. Show one client's audit trail (query) and a Merkle proof for its update.
  4. Tamper test: change one past record (a trust score / an ABAC decision) and
     verify() again -> invalid, naming the first bad block. Re-hashing the
     tampered block is also caught, because the next block's prev_hash no
     longer matches.
The ledger is saved as JSON lines to <results_dir>/ledger_<method>.jsonl.
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from src.attacks import attack_from_config  # noqa: E402
from src.fl.methods import build_method  # noqa: E402
from src.fl.pipeline import build_fl_data  # noqa: E402
from src.fl.server import run_federated  # noqa: E402
from src.ledger import HashChainLedger, build_ledger, merkle_proof, verify_merkle_proof  # noqa: E402
from src.ledger.ledger import compute_block_hash  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def tamper_test(ledger: HashChainLedger, block_index: int) -> dict:
    """Run the two tamper scenarios on copies of ``ledger``; return their verify() results."""
    out = {}
    # (a) edit a past record only
    a = copy.deepcopy(ledger)
    b = a.blocks[block_index]
    scores = b["screening"].get("trust_scores") or {}
    if scores:
        victim = sorted(scores)[0]
        scores[victim] = 1.0
        out["edited"] = f"block {block_index}: trust_scores[{victim}] -> 1.0"
    else:
        victim = sorted(b["abac_decisions"])[0]
        b["abac_decisions"][victim] = {"permit": True, "reason": "ok"}
        out["edited"] = f"block {block_index}: abac_decisions[{victim}] -> permit"
    out["edit_only"] = a.verify()
    # (b) edit and recompute that block's own hash (an attacker covering tracks)
    b["block_hash"] = compute_block_hash(b, a.algo)
    out["edit_and_rehash"] = a.verify()
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--attack", default="sign_flip")
    ap.add_argument("--fraction", type=float, default=0.2)
    ap.add_argument("--method", default="proposed:V5")
    ap.add_argument("--client", type=int, default=0, help="client whose audit trail is printed")
    ap.add_argument("--out", help="ledger .jsonl path (default: <results_dir>/ledger_<method>.jsonl)")
    args = ap.parse_args(argv)

    cfg = load_config("ciciot2023")
    set_seed(cfg["seed"])
    cfg["attacks"].update(type=args.attack, malicious_fraction=args.fraction)
    name, _, variant = args.method.partition(":")
    cfg["aggregator"]["name"] = name
    if variant:
        cfg["ablation"]["variant"] = variant
    cfg["ledger"]["enabled"] = True

    data, _, _ = build_fl_data(cfg, synthetic=args.synthetic, verbose=False)
    attack, mal = attack_from_config(cfg, data.client_groups, data.benign_id)
    out = Path(args.out) if args.out else (
        Path(cfg["paths"]["results_dir"]) / f"ledger_{args.method.replace(':', '_')}.jsonl")
    ledger = build_ledger(cfg, path=out)
    agg, admission = build_method(cfg, data.n_clients)
    res = run_federated(data, cfg, agg, attack=attack, rounds=args.rounds, admission=admission,
                        ledger=ledger, verbose=True)

    print(f"\nmalicious clients: {mal}")
    print(f"ledger: {len(ledger)} blocks -> {out}")
    print(f"head hash: {ledger.head()}")
    v = ledger.verify()
    print(f"verify(): valid={v['valid']} errors={v['errors']}")
    print(f"append time: mean {np.mean(ledger.timings['append_s']) * 1e3:.2f} ms/block; "
          f"verify time: {ledger.timings['verify_s'][-1] * 1e3:.2f} ms for {len(ledger)} blocks")

    # one client's audit trail + Merkle inclusion proof
    cname = f"client{args.client:02d}"
    print(f"\naudit trail of {cname}:")
    for e in ledger.query(client=cname):
        print("  ", e)
    last = ledger.blocks[-1]
    leaves = list(last["update_hashes"].values())
    if cname in last["update_hashes"]:
        i = list(last["update_hashes"]).index(cname)
        proof = merkle_proof(leaves, i, ledger.algo)
        ok = verify_merkle_proof(leaves[i], proof, last["update_merkle_root"], ledger.algo)
        print(f"Merkle proof for {cname} in round {last['round']}: {len(proof)} hashes, valid={ok}")

    # tamper test on a middle block
    idx = len(ledger) // 2
    t = tamper_test(ledger, idx)
    print(f"\nTAMPER TEST ({t['edited']})")
    print(f"  edit only      : valid={t['edit_only']['valid']} first_invalid={t['edit_only']['first_invalid']}"
          f" errors={t['edit_only']['errors']}")
    print(f"  edit + rehash  : valid={t['edit_and_rehash']['valid']} "
          f"first_invalid={t['edit_and_rehash']['first_invalid']} errors={t['edit_and_rehash']['errors']}")
    reloaded = HashChainLedger.load(out, ledger.algo).verify()
    print(f"\nreloaded from disk: valid={reloaded['valid']}")
    print("final:", {k: round(res['final'][k], 4) for k in ('accuracy', 'macro_f1', 'mcc')})


if __name__ == "__main__":
    main()
