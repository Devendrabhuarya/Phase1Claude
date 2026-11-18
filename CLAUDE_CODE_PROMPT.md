Read `Architecture_README.md` in this folder. It is the single source of
truth for this project. Implement the system it describes as a Python /
PyTorch codebase that runs on Google Colab, building it stage by stage and
pushing to GitHub after every stage.

## 1. Environment and constraints

- Target runtime: **Google Colab, CPU only, ~12 GB RAM**. No GPU assumed.
- FL framework: **plain PyTorch sequential simulation**. Do not use Flower,
  Ray or any distributed framework.
- Development machine: my laptop (Ryzen 5 5500U, 16 GB RAM, no GPU). You
  may not have the full datasets locally, so every stage must also run on
  **small synthetic data** for testing.
- If `data/sample/Merged01.csv` exists locally, also test against it.
- Dependencies: torch, numpy, pandas, scikit-learn, pycasbin, matplotlib,
  pyyaml, pytest. Pin versions in `requirements.txt`.

## 2. Repository structure

```
fl-ids/
├── README.md                 # how to run on Colab + stage log
├── Architecture_README.md    # design (do not change without asking me)
├── requirements.txt
├── .gitignore                # data/, results/, checkpoints/, *.csv, *.pt
├── configs/
│   ├── base.yaml             # all hyperparameters from the README
│   ├── ciciot2023.yaml
│   └── nbaiot.yaml
├── src/
│   ├── data/                 # loaders, label mapping, partitioning
│   ├── models/               # MLP
│   ├── fl/                   # client training, server loop
│   ├── attacks/
│   ├── aggregators/          # baselines + proposed
│   ├── abac/                 # PyCasbin model/policy + attribute registry
│   ├── reputation/
│   ├── ledger/               # hash chain + Merkle tree
│   ├── metrics/
│   └── utils/                # seeding, logging, I/O
├── experiments/              # runner + grid definitions
├── notebooks/
│   └── colab_main.ipynb      # mounts Drive, clones repo, runs experiments
└── tests/
```

## 3. Coding rules

- Every hyperparameter lives in the YAML configs, with the exact values from
  `Architecture_README.md`. No magic numbers in code.
- Dataset is a config option (`ciciot2023` | `nbaiot`). Model, FL loop,
  attacks, aggregators, ABAC, reputation and ledger must work unchanged for
  both (input/output size read from the data).
- All aggregators share one interface, for example
  `aggregate(updates, global_weights, context) -> (new_update, info)`, where
  `info` returns selections, scores and cohorts for logging and the ledger.
- Fixed seeds everywhere (Python, NumPy, PyTorch, partitioning).
- Memory-safe data loading: read CSVs in chunks; never load all 63
  CICIoT2023 files fully into memory.
- Results saved as CSV/JSON per run (config + per-round metrics), resumable:
  skip a run whose result file already exists.
- Clear docstrings; simple, readable code (this is a student M.Tech project
  and I must be able to explain it in my viva).

## 4. Data paths (Colab)

- CICIoT2023: `/content/drive/MyDrive/CICIoT2023/Merged01.csv … Merged63.csv`
  (39 features + `Label`).
- N-BaIoT: `/content/drive/MyDrive/N-BaIoT/` (layout unknown: write a
  loader that detects either per-device folders with `benign_traffic.csv`,
  `gafgyt_attacks/`, `mirai_attacks/`, or flat files like
  `1.benign.csv`, `1.gafgyt.combo.csv`, `1.mirai.ack.csv`).
- Results: `/content/drive/MyDrive/fl-ids-results/`.
- Paths are config values, never hard-coded in modules.

## 5. CICIoT2023 label mapping

Do not assume exact label spelling. Implement:

1. A data-inspection function that prints all unique labels and counts.
2. A keyword-based mapping to 8 classes on upper-cased labels with `-`,
   `_` and spaces removed, checked **in this order**:
   - `BENIGN` → Benign
   - `DDOS` → DDoS (must be checked before DoS)
   - `DOS` → DoS
   - `MIRAI` → Mirai
   - `RECON`, `VULNERABILITYSCAN` → Recon
   - `SPOOFING`, `MITM` → Spoofing
   - `BRUTEFORCE` → BruteForce
   - `SQLINJECTION`, `XSS`, `COMMANDINJECTION`, `BACKDOOR`, `UPLOADING`,
     `BROWSERHIJACKING` → Web
3. **Raise an error listing any unmapped label.** Never silently drop or
   mislabel data.

## 6. Stages

Work through these stages in order. For each stage: implement → write tests
→ run tests → update the stage log in `README.md` → commit → push → **stop
and report to me**. Do not start the next stage until I say "continue".

| Stage | Deliverable | Done when |
|---|---|---|
| 0 | Repo skeleton, configs, `.gitignore`, `requirements.txt`, Colab notebook that mounts Drive and clones the repo | `pytest` runs; notebook cells exist |
| 1 | CICIoT2023 loading (chunked, class-capped sampling), label inspection + 8-class mapping, preprocessing, global test set, root dataset | Unit tests pass on synthetic data; class counts printed |
| 2 | N-BaIoT loader (both layouts), 11-class labels, device IDs | Unit tests pass on synthetic files of both layouts |
| 3 | Partitioning: zone-based Dirichlet (CICIoT2023, zones A:7, B:7, C:6) and device-based split (N-BaIoT, 20 clients) | Per-client class tables printed; no client empty; every client has Benign |
| 4 | MLP + client training + FedAvg server loop + metrics (accuracy, macro-F1, MCC, per-class recall) | FedAvg clearly beats a majority-class baseline on synthetic data |
| 5 | TTL/IAT shortcut-feature check (centralized model, 39 vs 37 features) | Script outputs a comparison table |
| 6 | Attacks: label flip, sign flip, Gaussian, scaling, ALIE, Min-Max; malicious-client placement (spread / concentrated) | Each attack unit-tested; placement correct |
| 7 | Baselines: Median, Trimmed Mean, Krum, Multi-Krum, FLTrust | Each unit-tested on hand-made updates with a known outlier |
| 8 | Proposed method: ABAC (PyCasbin) + unauthorized-client injection, median-norm clipping, cohort formation, intra-cohort Multi-Krum + FLTrust, capped inter-cohort aggregation, reputation feedback + quarantine | Unit tests per component; end-to-end synthetic run beats FedAvg under sign flipping |
| 9 | Hash-chain ledger + Merkle root, `verify()`, tamper test, wired into the server loop | Tamper test fails verification as expected |
| 10 | Experiment runner (grid + priority order from README Section 7, ablation V1–V5), resumable runs, plots and result tables | A small smoke grid runs end-to-end and produces plots |

## 7. GitHub workflow

- Repository: **<YOUR_GITHUB_USERNAME>/fl-ids**, private. If it does not
  exist, create it with `gh repo create` (ask me first if `gh` is not
  authenticated).
- Work on `main`. After each stage, once tests pass:
  - `git add` only source, configs, tests, notebook and docs
  - commit message: `Stage N: <short description>`
  - `git tag stage-N`
  - `git push origin main --tags`
- **Never commit** datasets, CSV files, model checkpoints, result files,
  Drive paths with personal data, or any tokens/keys. Check `git status`
  before every commit.
- If a push fails, stop and tell me; do not force-push.
- In `README.md`, keep a **stage log**: stage, date, what was built, test
  results, known issues.

## 8. Reporting after each stage

After pushing, report:

1. What was implemented (files changed).
2. Test results.
3. Any deviation from `Architecture_README.md` and why.
4. Exact Colab cells I should run to try this stage on real data.
5. Questions for me, if any.

Then wait for me to say "continue".
