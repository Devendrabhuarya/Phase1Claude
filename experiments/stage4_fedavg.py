"""Stage 4: clean FedAvg run vs. the majority-class baseline.

    python experiments/stage4_fedavg.py --synthetic
    python experiments/stage4_fedavg.py --rounds 5      # real data (Stage-1 cache)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.aggregators import get_aggregator  # noqa: E402
from src.fl.pipeline import build_fl_data  # noqa: E402
from src.fl.server import run_federated  # noqa: E402
from src.metrics.detection import majority_baseline  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--rounds", type=int, help="override fl.rounds")
    args = ap.parse_args(argv)

    cfg = load_config("ciciot2023")
    set_seed(cfg["seed"])
    data, _, bundle = build_fl_data(cfg, synthetic=args.synthetic)
    result = run_federated(data, cfg, get_aggregator("fedavg", cfg), rounds=args.rounds)

    base = majority_baseline(bundle.y_pool, bundle.y_test, bundle.class_names)
    keys = ["accuracy", "macro_f1", "mcc"] + [f"recall_{c}" for c in bundle.class_names]
    table = pd.DataFrame({"FedAvg": [result["final"][k] for k in keys],
                          "Majority class": [base[k] for k in keys]}, index=keys)
    print("\n" + table.round(4).to_string())


if __name__ == "__main__":
    main()
