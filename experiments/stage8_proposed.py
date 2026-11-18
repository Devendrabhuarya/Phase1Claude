"""Stage 8: proposed method (V1–V5) vs FedAvg (and optionally baselines) under one attack.

    python experiments/stage8_proposed.py --synthetic --rounds 10 --attack sign_flip --fraction 0.2
    python experiments/stage8_proposed.py --rounds 10 --attack sign_flip --placement concentrated
    python experiments/stage8_proposed.py --synthetic --methods fedavg proposed:V5 --verbose
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.attacks import attack_from_config  # noqa: E402
from src.fl.methods import build_method  # noqa: E402
from src.fl.pipeline import build_fl_data  # noqa: E402
from src.fl.server import run_federated  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def summarize(res: dict, malicious: set) -> dict:
    """Final metrics + averages of the screening / ABAC / cohort statistics."""
    f, h = res["final"], res["history"]
    row = {k: f[k] for k in ("accuracy", "macro_f1", "mcc", "attack_success_rate")}

    def avg(key):
        vals = [r[key] for r in h if key in r and r[key] == r[key]]
        return float(np.mean(vals)) if vals else float("nan")

    row.update(screen_tpr=avg("screen_tpr"), screen_fpr=avg("screen_fpr"), cohort_ari=avg("cohort_ari"),
               server_s=avg("time_server_s"))
    if "intruders" in h[0]:
        row["intruder_block_rate"] = sum(r["intruders_blocked"] for r in h) / max(1, sum(r["intruders"] for r in h))
        q = {}
        for r in h:
            for c in r["newly_quarantined"]:
                q.setdefault(c, r["round"])
        mal_q = [q[c] for c in malicious if c in q]
        row["attackers_quarantined"] = f"{len(mal_q)}/{len(malicious)}"
        row["first_quarantine_round"] = min(mal_q) if mal_q else float("nan")
        row["honest_quarantined"] = sum(1 for c in q if c not in malicious)
    return row


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--attack", default="sign_flip")
    ap.add_argument("--fraction", type=float, default=0.2)
    ap.add_argument("--placement", default="spread", choices=["spread", "concentrated"])
    ap.add_argument("--methods", nargs="*",
                    default=["fedavg", "proposed:V1", "proposed:V2", "proposed:V3", "proposed:V4", "proposed:V5"],
                    help="aggregator names; proposed:Vk selects an ablation variant")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config("ciciot2023")
    set_seed(cfg["seed"])
    cfg["attacks"].update(type=args.attack, malicious_fraction=args.fraction, placement=args.placement)
    data, _, _ = build_fl_data(cfg, synthetic=args.synthetic, verbose=False)
    attack, mal = attack_from_config(cfg, data.client_groups, data.benign_id)
    print(f"attack={args.attack} fraction={args.fraction} placement={args.placement} "
          f"malicious={mal} zones={[data.client_groups[i] for i in mal]}")

    rows = {}
    for m in args.methods:
        c = copy.deepcopy(cfg)
        name, _, variant = m.partition(":")
        c["aggregator"]["name"] = name
        if variant:
            c["ablation"]["variant"] = variant
        agg, admission = build_method(c, data.n_clients)
        res = run_federated(data, c, agg, attack=attack, rounds=args.rounds,
                            admission=admission, verbose=args.verbose)
        rows[m] = summarize(res, set(mal))
        print(m, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in rows[m].items()})
    pd.set_option("display.width", 250)
    print("\n" + pd.DataFrame(rows).T.to_string(float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
