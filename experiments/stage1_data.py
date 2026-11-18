"""Stage 1: inspect CICIoT2023 labels, build the class-capped sample and splits.

Examples (Colab):
    # only list raw labels and their counts (reads the Label column only)
    python experiments/stage1_data.py --inspect-only
    # build sample + splits for the first 3 files (quick check)
    python experiments/stage1_data.py --max-files 3 --no-cache
    # full run over all 63 files (cached to Drive afterwards)
    python experiments/stage1_data.py

Locally with synthetic data:
    python experiments/stage1_data.py --synthetic
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.ciciot2023 import list_csv_files, load_or_build_sample  # noqa: E402
from src.data.labels import inspect_labels, summarize_mapping  # noqa: E402
from src.data.preprocess import build_bundle  # noqa: E402
from src.data.synthetic import write_synthetic_csvs  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", help="override paths.data_dir")
    ap.add_argument("--max-files", type=int, help="use only the first N files")
    ap.add_argument("--inspect-only", action="store_true", help="only print raw label counts")
    ap.add_argument("--no-cache", action="store_true", help="do not read/write the Drive cache")
    ap.add_argument("--synthetic", action="store_true", help="run on generated synthetic files")
    args = ap.parse_args(argv)

    cfg = load_config("ciciot2023")
    set_seed(cfg["seed"])
    if args.synthetic:
        tmp = Path(tempfile.mkdtemp(prefix="ciciot_synth_"))
        write_synthetic_csvs(tmp)
        cfg["paths"]["data_dir"] = str(tmp)
        cfg["data"]["class_cap"] = 300
        cfg["data"]["root_size"] = 80
        args.no_cache = True
    elif args.data_dir:
        cfg["paths"]["data_dir"] = args.data_dir

    files = list_csv_files(cfg["paths"]["data_dir"], cfg["paths"]["file_pattern"])
    if args.max_files:
        files = files[: args.max_files]
    print(f"{len(files)} file(s) from {cfg['paths']['data_dir']}")

    if args.inspect_only:
        counts = inspect_labels(files, cfg["data"]["label_column"], cfg["data"]["chunk_size"])
        print("\nMapping to 8 classes:")
        print(summarize_mapping(counts.to_dict(), cfg["data"]["label_keywords"]).to_string())
        return

    sample = load_or_build_sample(cfg, files=files, use_cache=not args.no_cache)
    print("\nRaw label -> class mapping (all scanned rows):")
    print(summarize_mapping(sample.raw_label_counts.to_dict(), cfg["data"]["label_keywords"]).to_string())
    print("\nRows per class: seen in files vs kept after cap + dedup")
    table = sample.class_rows_seen.to_frame().join(sample.class_counts())
    print(table.to_string())
    print(f"Total kept: {len(sample.y)}")

    bundle = build_bundle(sample, cfg)
    print(f"\nFeatures used: {bundle.n_features}")
    print("Rows per class in client pool / global test / root:")
    print(bundle.class_table().to_string())


if __name__ == "__main__":
    main()
