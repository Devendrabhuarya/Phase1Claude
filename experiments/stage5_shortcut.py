"""Stage 5: TTL/IAT shortcut-feature check (centralized MLP, 39 vs 37 features).

    python experiments/stage5_shortcut.py              # real data (Stage-1 cache)
    python experiments/stage5_shortcut.py --synthetic
The comparison table is printed and saved to <results_dir>/shortcut_check.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ciciot2023 import load_or_build_sample  # noqa: E402
from src.data.synthetic import SYNTHETIC_FL_COUNTS, make_synthetic_sample  # noqa: E402
from src.metrics.shortcut import shortcut_check  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--epochs", type=int, help="override shortcut_check.epochs")
    ap.add_argument("--out", help="CSV path (default: <results_dir>/shortcut_check.csv)")
    args = ap.parse_args(argv)

    cfg = load_config("ciciot2023")
    set_seed(cfg["seed"])
    if args.epochs:
        cfg["shortcut_check"]["epochs"] = args.epochs
    if args.synthetic:
        cfg["data"]["root_size"] = 120
        sample = make_synthetic_sample(SYNTHETIC_FL_COUNTS, cfg["seed"])
    else:
        sample = load_or_build_sample(cfg)

    table, use_reduced = shortcut_check(sample, cfg)
    print("\n" + table.round(4).to_string())
    drop = -table.loc["macro_f1", "difference"]
    print(f"\nMacro-F1 drop without shortcut features: {drop:.4f} "
          f"(sharp if >= {cfg['shortcut_check']['sharp_drop']})")
    if use_reduced:
        print("=> SHARP DROP: the model relies on the shortcut features. Use the 37-feature set:\n"
              "   set data.drop_shortcut_features: true in configs/base.yaml")
    else:
        print("=> No sharp drop: keep all 39 features (data.drop_shortcut_features: false).")

    out = Path(args.out) if args.out else Path(cfg["paths"]["results_dir"]) / "shortcut_check.csv"
    if not args.synthetic or args.out:
        out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(out)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
