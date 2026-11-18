# FL-IDS — ABAC-Secured, Byzantine-Resilient Federated IoT IDS

M.Tech Phase-1 implementation of the system described in
[`Architecture_README.md`](Architecture_README.md) (the design document and
single source of truth). Plain PyTorch sequential FL simulation, CPU only,
designed for Google Colab (~12 GB RAM).

## Repository layout

```
configs/      base.yaml (all hyperparameters) + ciciot2023.yaml
src/data      loaders, label mapping, partitioning
src/models    MLP
src/fl        client training, server loop
src/attacks   poisoning attacks, malicious-client placement
src/aggregators  baselines + proposed method
src/abac      PyCasbin model/policy + attribute registry
src/reputation   reputation EMA + quarantine
src/ledger    hash chain + Merkle tree
src/metrics   evaluation metrics
src/utils     config, seeding, logging, I/O
experiments/  runner + grid definitions
notebooks/colab_main.ipynb   Colab entry point
tests/        pytest unit tests (synthetic data)
```

## Configuration

Every hyperparameter is in `configs/`. A run's config = `base.yaml` ⟵
`<dataset>.yaml` ⟵ overrides (deep merge):

```python
from src.utils.config import load_config
cfg = load_config("ciciot2023", overrides={"attacks": {"type": "sign_flip", "malicious_fraction": 0.2}})
```

Values not fixed by the architecture README are marked `[NOT IN README]` in
`configs/base.yaml`.

**Dataset scope:** only CICIoT2023 is used. N-BaIoT (Architecture README 5.3)
was dropped by project decision on 2026-11-18; the code stays dataset-agnostic
(input/output sizes are read from the data).

## Data pipeline (Stage 1)

`src/data/` — `labels.py` (inspection + ordered keyword mapping to 8 classes,
error on unmapped labels), `ciciot2023.py` (chunked reading with per-class
"bottom-k" random sampling, so at most `class_cap` rows per class are kept
uniformly from all 63 files; de-duplication and inf→NaN while reading; `.npz`
cache on Drive), `preprocess.py` (shortcut-feature removal option, stratified
20 % test / 500-row root / client pool, train-median imputation + standard
scaling fitted on the client pool only).

```bash
python experiments/stage1_data.py --inspect-only     # raw labels + mapping
python experiments/stage1_data.py                    # sample, splits, class tables
python experiments/stage1_data.py --synthetic        # local check without data
```

## Running on Google Colab

1. Put the data on Drive: `MyDrive/CICIoT2023/Merged01.csv … Merged63.csv`
2. Open `notebooks/colab_main.ipynb` in Colab (File → Open notebook → GitHub,
   or upload it), CPU runtime.
3. Run the cells top to bottom: mount Drive → clone repo (asks for a
   read-only GitHub token) → install PyCasbin → run tests → load config.
4. Results are written to `MyDrive/fl-ids-results/`.

## Running locally

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows  (Linux/macOS: source .venv/bin/activate)
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m pytest
```

## Stage log

| Stage | Date | What was built | Tests | Known issues |
|---|---|---|---|---|
| 0 | 2026-11-18 | Repo skeleton (`src/*` packages, `experiments/`, `tests/`), `configs/base.yaml` + `ciciot2023.yaml` + `nbaiot.yaml` with README hyperparameters, `.gitignore`, pinned `requirements.txt`, utils (config deep-merge loader, seeding, logging/JSON/CSV I/O, resumable-run check), Colab notebook (mount Drive, clone, install, pytest, config check) | 23 passed (local, Python 3.11, torch 2.15.0 CPU) | Several values are not fixed by the design README and use marked defaults (attack strengths, firmware/freshness, unauthorized-client count, client val split, trimmed-mean β). |
| 1 | 2026-11-18 | CICIoT2023 pipeline: label inspection + ordered keyword mapping (raises on unmapped labels), chunked class-capped sampling (bottom-k, uniform over all files), dedup + inf handling, `.npz` Drive cache, stratified test/root/pool splits, train-only median imputation + standard scaling, shortcut-feature drop option, synthetic data generator, `experiments/stage1_data.py`, notebook cells | 42 passed (19 new); synthetic run prints class tables; ~6 s per 420k rows locally | Full 63-file scan estimated 20–30 min on Colab (once, then cached). Not yet run on real data. Duplicates are removed within the kept sample (a dropped row cannot duplicate a kept one). |
| 2 | — | **Skipped**: N-BaIoT dropped by project decision (`nbaiot.yaml` removed, `nbaiot` no longer a valid dataset). | — | Deviation from Architecture README 5.3 / 7.3. |
