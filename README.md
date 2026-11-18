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

## Client partitioning (Stage 3)

`src/data/partition.py` — zone-based Dirichlet split of the client pool into
20 clients (zones A:7, B:7, C:6). One home-share `s` (found by bisection) sends
part of every attack class to its home zone so zones are ~75 % dominant on
average; the rest is spread over the other zones, so every class exists in the
federation. Inside a zone each class is split by Dirichlet(α). Each client gets
a random 30–50 % Benign share. `python experiments/stage3_partition.py [--alpha 0.1]`.

## FL core (Stage 4)

`src/models/mlp.py` (MLP, flat-weight helpers), `src/fl/client.py` (class-weighted
CE, SGD; returns Δ = w_local − w_global), `src/fl/server.py` (sequential rounds,
lazy root update g₀ for FLTrust, per-round test metrics, attack hooks),
`src/aggregators/` (common `aggregate(updates, global_weights, context) ->
(new_update, info)` interface, FedAvg), `src/metrics/detection.py`.
`python experiments/stage4_fedavg.py [--synthetic] [--rounds N]`.

## Shortcut-feature check (Stage 5)

`src/metrics/shortcut.py` trains the MLP centrally on the client pool with all 39
features and without `Time_To_Live` and `IAT` (37), and compares test metrics.
A macro-F1 drop ≥ `shortcut_check.sharp_drop` means the model relies on the
shortcuts → set `data.drop_shortcut_features: true`.
`python experiments/stage5_shortcut.py` (saves `shortcut_check.csv` to the results folder).

## Attacks (Stage 6)

`src/attacks/placement.py` picks `round(fraction·20)` malicious clients — *spread*
(round-robin over zones) or *concentrated* (zone A filled first, overflow spread).
`src/attacks/attacks.py`: label flip (attack labels → Benign), sign flip, Gaussian
noise, scaling, ALIE (μ − z·σ, z from Baruch et al.), Min-Max (largest γ keeping
the malicious update within the max benign pairwise distance). The server calls
`poison_data` once and `poison_updates` every round.
`python experiments/stage6_attacks.py [--synthetic] --rounds 10 --fraction 0.2 --placement spread`.

## Robust baselines (Stage 7)

`src/aggregators/robust.py`: coordinate-wise **Median**, **Trimmed Mean**
(β per side), **Krum** / **Multi-Krum** (score = sum of squared distances to the
n − f − 2 nearest neighbours, f = ⌊(n − 3)/2⌋, Multi-Krum averages the n − f best),
**FLTrust** (TSᵢ = ReLU(cos(Δᵢ, g₀)), Δᵢ scaled to ‖g₀‖, trust-weighted mean).
The helpers `krum_scores`, `multikrum_select`, `fltrust_scores` are reused by the
proposed method. `python experiments/stage7_baselines.py --attack sign_flip`.

## Proposed method (Stage 8)

* `src/abac/` — attribute registry, PyCasbin ABAC policy (permit iff registered ∧
  firmware ≥ min ∧ fresh ∧ reputation ≥ τ ∧ not quarantined) with readable deny
  reasons, and `AdmissionController` (runs before training, injects unauthorized
  joins: unregistered / stale firmware / expired).
* `src/reputation/` — EMA reputation, quarantine for 5 rounds, probation at τ.
* `src/aggregators/cohorts.py` — median-norm clipping; cohort formation
  (final-layer cosine distance, average-linkage clustering, K ∈ {2,3} by silhouette,
  min cohort size 6, re-clustered every round).
* `src/aggregators/proposed.py` — per-cohort Multi-Krum + FLTrust, root-alignment
  check, trust-proportional weights capped at 0.5 (water-filling), reputation scores;
  ablation variants V1–V5 (`ablation.variant`).
* `src/fl/methods.py` — `build_method(cfg, n_clients)` → (aggregator, admission).
* `python experiments/stage8_proposed.py --synthetic --rounds 10 --attack sign_flip`.

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
| 3 | 2026-11-18 | Zone-based Dirichlet partitioning (`src/data/partition.py`), per-client / per-zone class tables, `experiments/stage3_partition.py`; Benign cap raised to 200k (`class_cap_overrides`) | 51 passed (9 new); 20 clients, none empty, all have 30–50 % Benign, zone dominance ≈ 0.70–0.80 (mean 0.75) on synthetic | With 50k Benign the 30–50 % rule is impossible (≈290k attack rows), hence the Benign cap; if Benign is still short, attack rows are sub-sampled with a warning. Stage-1 cache must be rebuilt (new cache name). |
| 4 | 2026-11-18 | MLP (sizes from data), client training (class-weighted CE capped at 10, SGD), sequential FL server with lazy root update + attack hooks, aggregator interface + FedAvg, metrics (accuracy, macro-F1, MCC, per-class recall, attack success rate, screening TPR/FPR), `experiments/stage4_fedavg.py` | 60 passed (9 new). Synthetic, 10 rounds: FedAvg acc 0.744 / F1 0.475 / MCC 0.679 vs majority acc 0.408 / F1 0.073 / MCC 0 | Synthetic data enlarged (≈50k rows) because tiny clients get too few SGD steps at lr 0.01. Rare synthetic classes (Web, BruteForce) are still not learned in 10 rounds; real-data behaviour to be checked on Colab. |
| 5 | 2026-11-18 | Shortcut-feature check: centralized MLP 39 vs 37 features, comparison table (accuracy, macro-F1, MCC, per-class recall, difference), decision by macro-F1 drop, `experiments/stage5_shortcut.py` | 62 passed (2 new): no false alarm when signal is spread; a planted TTL/IAT-only signal gives F1 drop 0.65 → reduced set recommended | "Sharp drop" threshold (0.05 macro-F1) and 10 centralized epochs are my choices. Real-data result pending (Colab). |
| 6 | 2026-11-18 | Attacks (label flip, sign flip, Gaussian, scaling, ALIE, Min-Max) with data/update hooks in the server, malicious placement (spread / concentrated with overflow), `attack_from_config`, `experiments/stage6_attacks.py`, notebook cells for Stages 3–6 | 79 passed (17 new). Synthetic FedAvg, 20 % spread, 10 rounds — acc: clean 0.744, label flip 0.688 (ASR 0.07→0.20), sign flip 0.645, Gaussian 0.551, scaling 0.157, ALIE 0.734, Min-Max 0.726 | Attack strengths (sign scale 1, noise std 1, scaling ×10), Min-Max perturbation (std) and concentrated zone (A) are my choices. Adaptive attacks use this round's benign-update statistics (omniscient). |
| 7 | 2026-11-18 | *(branch `stage-7-baselines`)* Median, Trimmed Mean, Krum, Multi-Krum, FLTrust behind the common interface + registry; reusable `krum_scores` / `multikrum_select` / `fltrust_scores`; FLTrust root update now runs as many SGD steps as an average client (`fltrust.root_steps: match_clients`); `experiments/stage7_baselines.py` | 95 passed (16 new): each rule tested on hand-made updates with a known outlier; under scaling (20 %) all robust rules beat FedAvg by > 0.2 acc. Synthetic sign flip 20 %, 10 rounds — acc: FedAvg 0.645, Median 0.600, TrMean 0.650, Krum 0.524 (FPR 0.94), Multi-Krum 0.536 (TPR 0.75, FPR 0.31), FLTrust 0.604 (TPR 1.0, FPR 0) | Under non-IID data, global Krum/Multi-Krum reject many honest clients (high FPR) — the problem the cohort design targets. Root-step matching is a deviation (README only says "train on the root dataset"). |
| 8 | 2026-11-18 | *(branch `stage-8-proposed`)* ABAC (PyCasbin) + registry + unauthorized-join injection, reputation/quarantine loop, median-norm clipping, cohort formation, intra-cohort Multi-Krum + FLTrust, capped inter-cohort aggregation, ablation V1–V5, server admission hook (only admitted clients train), per-round ABAC/cohort-ARI logging, `experiments/stage8_proposed.py` | 116 passed (21 new). Synthetic sign flip 20 % spread, 10 rounds — FedAvg acc 0.645 / F1 0.324; V5 acc 0.647 / F1 0.474, all 4 attackers quarantined by round 3, 100 % intruders blocked. 40 %: FedAvg 0.540 / 0.208, V4 0.667 / 0.472, V5 0.547 / 0.305 | **False quarantines:** V5 quarantines 10–11 honest clients — Multi-Krum always drops f = ⌊(n−3)/2⌋ (8 of 20) updates and the README sets sᵢ = 0 for them, so honest non-IID clients dropped 2–3 times fall below τ. Cohort ARI ≈ 0 on synthetic data (zones not separable there). Needs a design decision (see report). |
