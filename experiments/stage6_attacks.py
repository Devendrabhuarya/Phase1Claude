"""Stage 6: FedAvg under every attack (shows the attacks hurt an undefended model).

    python experiments/stage6_attacks.py --synthetic --rounds 10
    python experiments/stage6_attacks.py --rounds 10 --fraction 0.2 --placement concentrated
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
    ap.add_argument("--fraction", type=float, default=0.2)
    ap.add_argument("--placement", default="spread", choices=["spread", "concentrated"])
    ap.add_argument("--attacks", nargs="*", help="subset of attacks (default: all from config)")
    args = ap.parse_args(argv)

    cfg = load_config("ciciot2023")
    set_seed(cfg["seed"])
    data, _, _ = build_fl_data(cfg, synthetic=args.synthetic, verbose=False)
    rows = {}
    for kind in ["none"] + (args.attacks or cfg["experiment"]["attacks"]):
        c = copy.deepcopy(cfg)
        c["attacks"].update(type=kind, malicious_fraction=0.0 if kind == "none" else args.fraction,
                            placement=args.placement)
        attack, mal = attack_from_config(c, data.client_groups, data.benign_id)
        print(f"\n=== {kind}  malicious={mal} ({[data.client_groups[i] for i in mal]})")
        res = run_federated(data, c, get_aggregator("fedavg", c), attack=attack,
                            rounds=args.rounds, verbose=False)
        f = res["final"]
        rows[kind] = {k: f[k] for k in ("accuracy", "macro_f1", "mcc", "attack_success_rate")}
        print({k: round(v, 4) for k, v in rows[kind].items()})
    print(f"\nFedAvg, fraction={args.fraction}, placement={args.placement}")
    print(pd.DataFrame(rows).T.round(4).to_string())


if __name__ == "__main__":
    main()
