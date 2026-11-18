"""Stage 3: zone-based Dirichlet partitioning, printed per client and per zone.

    python experiments/stage3_partition.py              # real data (needs Stage-1 cache)
    python experiments/stage3_partition.py --alpha 0.1  # hard non-IID case
    python experiments/stage3_partition.py --synthetic  # local check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from src.data.partition import client_class_table, partition_from_config, zone_class_table  # noqa: E402
from src.data.preprocess import build_bundle, prepare_dataset  # noqa: E402
from src.data.synthetic import SYNTHETIC_FL_COUNTS, make_synthetic_sample  # noqa: E402
from src.utils.config import load_config  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, help="Dirichlet alpha (default: config)")
    ap.add_argument("--synthetic", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config("ciciot2023")
    if args.synthetic:
        cfg["data"]["root_size"] = 80
        bundle = build_bundle(make_synthetic_sample(SYNTHETIC_FL_COUNTS, cfg["seed"]), cfg)
    else:
        bundle = prepare_dataset(cfg, verbose=False)

    part = partition_from_config(bundle.y_pool, bundle.class_names, cfg, alpha=args.alpha)
    pd.set_option("display.width", 200)
    print(f"alpha={part.info['alpha']}  home share={part.info['home_share']:.3f}  "
          f"zone dominance={ {z: round(v, 3) for z, v in part.info['zone_dominance'].items()} }")
    print(f"attack rows kept={part.info['attack_rows_kept']:.1%}  unused Benign={part.info['benign_unused']}")
    print("\nPer-client class counts:")
    print(client_class_table(part, bundle.y_pool, bundle.class_names).to_string())
    print("\nPer-zone class counts:")
    print(zone_class_table(part, bundle.y_pool, bundle.class_names).to_string())


if __name__ == "__main__":
    main()
