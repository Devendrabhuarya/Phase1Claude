"""Stage 7: compare FedAvg with the robust baselines under one attack.

    python experiments/stage7_baselines.py --synthetic --rounds 10 --attack scaling
    python experiments/stage7_baselines.py --rounds 10 --attack sign_flip --fraction 0.2
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.aggregators import get_aggregator  # noqa: E402
from src.attacks import attack_from_config  # noqa: E402
from src.fl.pipeline import build_fl_data  # noqa: E402
from src.fl.server import run_federated  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--attack", default="sign_flip")
    ap.add_argument("--fraction", type=float, default=0.2)
    ap.add_argument("--placement", default="spread", choices=["spread", "concentrated"])
    ap.add_argument("--aggregators", nargs="*", help="default: experiment.baselines from config")
    args = ap.parse_args(argv)

    cfg = load_config("ciciot2023")
    set_seed(cfg["seed"])
    cfg["attacks"].update(type=args.attack, malicious_fraction=args.fraction, placement=args.placement)
    data, _, _ = build_fl_data(cfg, synthetic=args.synthetic, verbose=False)
    attack, mal = attack_from_config(cfg, data.client_groups, data.benign_id)
    print(f"attack={args.attack} fraction={args.fraction} placement={args.placement} malicious={mal}")

    rows = {}
    for name in args.aggregators or cfg["experiment"]["baselines"]:
        res = run_federated(data, copy.deepcopy(cfg), get_aggregator(name, cfg), attack=attack,
                            rounds=args.rounds, verbose=False)
        f = res["final"]
        row = {k: f[k] for k in ("accuracy", "macro_f1", "mcc", "attack_success_rate")}
        tprs = [r["screen_tpr"] for r in res["history"] if "screen_tpr" in r]
        fprs = [r["screen_fpr"] for r in res["history"] if "screen_fpr" in r]
        row["screen_tpr"] = float(np.mean(tprs)) if tprs else float("nan")
        row["screen_fpr"] = float(np.mean(fprs)) if fprs else float("nan")
        row["server_s_per_round"] = float(np.mean([r["time_server_s"] for r in res["history"]]))
        rows[name] = row
        print(name, {k: round(v, 4) for k, v in row.items()})
    print("\n" + pd.DataFrame(rows).T.round(4).to_string())


if __name__ == "__main__":
    main()
