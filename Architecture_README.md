# System Architecture (Final — Phase 1)

## Project Title

**ABAC-Secured, Byzantine-Resilient Federated Edge Intrusion Detection
System with Cohort-Aware Screening and Reputation Feedback**

## 1. Objective

Reduce the impact of malicious clients on the global model in a
federated IoT Intrusion Detection System (IDS), while keeping clean
(attack-free) detection performance close to standard federated
averaging.

The system defends at three levels:

1.  **Before training** — ABAC admission control blocks unauthorized,
    stale or low-reputation clients.
2.  **During aggregation** — norm clipping, similarity-based cohorts,
    Multi-Krum and FLTrust-style scoring limit the influence of poisoned
    updates that come from admitted clients.
3.  **Across rounds** — a reputation score feeds screening results back
    into the ABAC policy, so repeatedly suspicious clients are
    quarantined.

A tamper-evident audit ledger records every decision.

## 2. Contributions

1.  A **closed-loop ABAC policy**: client reputation, computed from
    robust-aggregation outcomes, is an ABAC attribute that controls
    admission in later rounds.
2.  **Cohort-aware two-stage screening**: Multi-Krum (peer-distance
    check) followed by client-level FLTrust scoring (root-reference
    check), applied inside similarity-based cohorts.
3.  **Cohort-capture safeguards**: minimum cohort size, per-round
    re-clustering, cohort-level root-alignment check and capped
    inter-cohort weights, all evaluated under a concentrated-attacker
    scenario.
4.  A **lightweight tamper-evident audit ledger** (hash chain + Merkle
    root) that runs on commodity hardware and can later be replaced by
    Hyperledger Fabric.
5.  A systematic evaluation on CICIoT2023 (controlled non-IID) and
    N-BaIoT (natural device-level non-IID), with ablations of every
    component.

## 3. Threat Model

| Item | Assumption |
|---|---|
| Server | Honest; holds a small clean root dataset (≈500 labelled flows). |
| Unauthorized clients | Unregistered, stale-firmware or spoofed identities try to join. **Handled by ABAC.** |
| Compromised authorized clients | Pass ABAC but send poisoned updates. **Handled by screening + reputation.** |
| Malicious fraction | 0 %, 10 %, 20 %, 30 %, 40 % of the 20 clients. |
| Collusion | Malicious clients may coordinate (identical or similar updates). |
| Placement | **Spread** across zones, or **concentrated** in one zone (cohort-capture test). |
| Knowledge | Non-adaptive attacks know nothing about the defence; adaptive attacks (ALIE, Min-Max) know benign update statistics. |

### Attacks evaluated

| Type | Attack | Description |
|---|---|---|
| Data poisoning | Label flipping | Attack labels flipped to Benign. |
| Model poisoning | Sign flipping | Update multiplied by −1 (optionally scaled). |
| Model poisoning | Gaussian noise | Update replaced by random noise. |
| Model poisoning | Scaling | Update multiplied by a large factor. |
| Adaptive | ALIE | Small, coordinated shift within benign variance (evades distance-based rules). |
| Adaptive | Min-Max | Maximal deviation that stays within benign pairwise distances. |
| Access | Unauthorized / Sybil join | Fake or non-compliant identities attempt to participate. |

## 4. End-to-End Architecture Flow

``` text
          Registered IoT Clients (20)          Unauthorized / Sybil Clients
                     |                                     |
              Local Traffic Data                           |
                     |                                     |
              Local IDS Training                           |
                     |                                     |
               Client Updates  <---------------------------+
                     |
                     v
   +--------------------------------------------+
   |  ABAC Admission Control (PyCasbin)         |<------------------+
   |  registered, firmware, freshness,          |                   |
   |  reputation >= tau, not quarantined        |                   |
   +--------------------------------------------+                   |
                     |  admitted updates                            |
                     v                                              |
           Median-Norm Clipping                                     |
                     |                                              |
                     v                                              |
     Cohort Formation (Stage 1, every round)                        |
     cosine distance + agglomerative clustering                     |
     min cohort size, small clusters merged                         |
                     |                                              |
        +------------+------------+                                 |
        v            v            v                                 |
    Cohort 1     Cohort 2     Cohort K                              |
        |            |            |                                 |
   Multi-Krum   Multi-Krum   Multi-Krum      (Stage 2A: peers)      |
        |            |            |                                 |
   FLTrust      FLTrust      FLTrust         (Stage 2B: root ref)   |
   scoring      scoring      scoring                                |
        |            |            |                                 |
   Cohort       Cohort       Cohort                                 |
   update       update       update                                 |
        +------------+------------+                                 |
                     v                                              |
   Inter-Cohort Aggregation (Stage 3)                               |
   root-alignment check + capped cohort weights                     |
                     |                                              |
                     v                                              |
           Updated Global IDS Model ---------> Distribute (next round)
                     |                                              |
                     v                                              |
           Reputation Update  ---------------------------------------+
                     |
                     v
   Audit Ledger: hash-chained block per round
   (ABAC decisions, update hashes, cohorts, scores, model hash)
```

## 5. Data Design

### 5.1 Primary dataset — CICIoT2023

-   **Source files:** `Merged01.csv` … `Merged63.csv` (63 files).
-   **Labels:** the 34 original labels are mapped to **8 classes**:
    Benign, DDoS, DoS, Recon, Spoofing, Mirai, Web, BruteForce.
-   **Sampling (fits Colab RAM):** read files in chunks and keep a
    class-capped sample (e.g. up to ≈50,000 rows per class; all rows of
    rare classes Web and BruteForce). Expected total ≈300–400k rows.
-   **Features:** 39 numeric features (Header_Length, Protocol Type,
    Time_To_Live, Rate, 7 TCP flag numbers, 4 flag counts, 15 protocol
    indicators HTTP…LLC, Tot sum, Min, Max, AVG, Std, Tot size, IAT,
    Number, Variance) + `Label`.
-   **Shortcut-feature check:** `Time_To_Live` (may encode the attacker
    host's OS) and `IAT` (may encode the capture session) can act as
    shortcuts. A centralized model is trained with all 39 features and
    without these two; if performance drops sharply, the reduced
    37-feature set is used for all experiments.
-   **Preprocessing:** remove duplicates, replace inf/NaN, standard
    scaling fitted on training data only.
-   **Splits:**
    -   Global test set: 20 % (stratified), never seen by clients.
    -   Root dataset: ≈500 stratified flows from the held-out pool,
        server-only, used by FLTrust.
    -   Remaining data partitioned among clients (5.2).

### 5.2 Client partitioning — zone-based Dirichlet split

CICIoT2023 has no device-identity column, so heterogeneity is created
with three hidden **network zones** that emulate different IoT
environments:

| Zone | Clients | Dominant attacks (~75 % of zone's attack data) |
|---|---|---|
| A — Smart-home flooding | 7 | DDoS, DoS |
| B — Recon & spoofing | 7 | Recon, Spoofing, BruteForce |
| C — Botnet & web | 6 | Mirai, Web (+ some DDoS) |

Rules:

-   Every client receives Benign traffic (≈30–50 % of its data).
-   The remaining ≈25 % of each zone's attack data comes from the other
    classes, so every class exists in the federation.
-   Inside a zone, data is divided among clients with **Dirichlet
    α = 0.5** (main) and **α = 0.1** (hard case), giving different class
    mixes and data quantities.
-   Each client keeps a local train/validation split.

Zone sizes (7/7/6) match the Multi-Krum condition n > 2f + 2, so a
7-client cohort tolerates two attackers.

Zones are **ground truth for evaluation only**; the server never sees
them. The quality of cohort formation is measured by the Adjusted Rand
Index (ARI) between discovered cohorts and true zones.

### 5.3 Secondary dataset — N-BaIoT

-   9 real devices, 115 features, Benign + Mirai + Bashlite (Gafgyt).
-   **Labels:** 11 classes — Benign + 5 Gafgyt attacks (combo, junk,
    scan, tcp, udp) + 5 Mirai attacks (ack, scan, syn, udp, udpplain).
    Devices without Mirai data (Ennio doorbell, Samsung webcam) give
    natural label skew.
-   **Model:** same MLP, 115 → 128 → 64 → 11.
-   **Natural device split:** each device's data divided into ≈2
    clients (20 clients total: 2 per device, plus 1 extra client from
    each of the 2 largest devices), so clients from the same device
    form natural groups. ARI is measured against device identity.
-   Same global test set / root dataset procedure as CICIoT2023.
-   Used for one validation experiment that shows the method works on
    real device-level non-IID data. Not used for headline results
    (the task is close to saturated).

## 6. Architecture Components

### 6.1 Local IDS Model and Training

| Setting | Value |
|---|---|
| Model | MLP: 39 (or 37) → 128 → 64 → 8, ReLU, dropout 0.2 |
| Loss | Cross-entropy (class-weighted) |
| Optimizer | SGD, lr 0.01, momentum 0.9 |
| Local epochs / batch | 2 / 256 |
| Rounds | 40 |
| Participation | All admitted clients every round |

Each client sends its update Δᵢ = wᵢ − w_global, never raw traffic.

### 6.2 ABAC Admission Control (closed loop)

Implemented with **PyCasbin** (ABAC model) over a client attribute
registry.

| Attribute | Example |
|---|---|
| `registered` | true / false |
| `device_type` | camera, thermostat, gateway … |
| `firmware_version` | must be ≥ minimum version |
| `last_seen` | must be within freshness window |
| `reputation` | from 6.7, range [0, 1] |
| `quarantined_until` | round number |

**Policy:** permit if `registered ∧ firmware ≥ min ∧ fresh ∧
reputation ≥ τ (0.3) ∧ not quarantined`.

Unauthorized clients (unregistered, stale firmware, expired) are
injected in experiments to show ABAC blocks them before aggregation.
Every decision is written to the audit ledger.

### 6.3 Pre-screening — Median-Norm Clipping

Each admitted update is scaled down to the median update norm of that
round if it is larger. This neutralizes scaling/boosting attacks before
clustering and keeps distances comparable.

### 6.4 Stage 1 — Cohort Formation

-   Feature for similarity: the flattened update of the final layer
    (cheap and most class-informative).
-   Distance: cosine distance matrix between admitted clients.
-   Clustering: agglomerative clustering (average linkage); number of
    cohorts K ∈ {2, 3} chosen by silhouette score.
-   **Minimum cohort size = 6**; smaller clusters are merged into the
    nearest cohort.
-   **Re-clustered every round**, so attackers cannot hold a fixed
    cohort.

Similarity is not proof of honesty: cohorts only organize screening so
that clients are compared with peers that have similar data.

### 6.5 Stage 2 — Intra-Cohort Screening

**A. Multi-Krum (peer-distance check)**

-   For a cohort of size n_c, assume f_c = ⌊(n_c − 3) / 2⌋ attackers.
-   Score each update by the sum of squared distances to its
    n_c − f_c − 2 nearest neighbours.
-   Keep the m_c = n_c − f_c lowest-scoring updates.

**B. FLTrust-style client scoring (root-reference check)**

-   Server computes root update g₀ by training the current global model
    on the root dataset.
-   Trust score: TSᵢ = ReLU(cos(Δᵢ, g₀)).
-   Magnitude normalization: Δᵢ′ = Δᵢ · ‖g₀‖ / ‖Δᵢ‖.
-   Applied to Multi-Krum-selected clients (client level, not to a
    cohort aggregate).
-   Cohort update: Δ_c = Σ TSᵢ Δᵢ′ / Σ TSᵢ. If Σ TSᵢ = 0, the cohort is
    dropped for this round.

### 6.6 Stage 3 — Inter-Cohort Aggregation

1.  **Root-alignment check:** a cohort update with cos(Δ_c, g₀) ≤ 0 is
    discarded.
2.  **Cohort weight:** proportional to the cohort's total trust Σ TSᵢ.
3.  **Weight cap:** no cohort may exceed w_max = 0.5 of the total
    (weights are renormalized after capping), which limits the damage of
    a captured cohort.
4.  Global update: w_global ← w_global + Σ w_c Δ_c.

### 6.7 Reputation Update (feedback to ABAC)

-   Per-round score: sᵢ = TSᵢ if the client survived Multi-Krum,
    otherwise 0.
-   Reputation (exponential moving average): rᵢ ← β·rᵢ + (1 − β)·sᵢ,
    β = 0.8, initial rᵢ = 0.5.
-   If rᵢ < τ = 0.3, the client is **quarantined for 5 rounds**, then
    re-admitted on probation with rᵢ = τ.
-   False quarantines of honest clients are measured and reported.

### 6.8 Audit Ledger (blockchain replacement)

A Python **hash-chained ledger** emulates a permissioned blockchain
audit trail at near-zero cost.

One block per round:

| Field | Content |
|---|---|
| `index`, `round`, `timestamp` | Block metadata |
| `prev_hash` | SHA-256 of the previous block |
| `abac_decisions` | Client ID → permit/deny + reason |
| `update_merkle_root` | Merkle root of SHA-256 hashes of all admitted updates |
| `cohorts` | Client → cohort assignment |
| `screening` | Multi-Krum selections, trust scores, dropped cohorts |
| `reputation` | Updated reputation and quarantine list |
| `global_model_hash` | SHA-256 of new global weights |
| `block_hash` | SHA-256 of all fields above |

-   Raw traffic and full update vectors stay **off-chain**; only hashes
    and decisions are stored.
-   A **verification routine** recomputes the chain; a tamper test
    modifies one past record and shows verification fails.
-   The ledger sits behind a small interface (`append`, `verify`,
    `query`) so it can later be replaced by Hyperledger Fabric without
    changing the FL code.
-   The ledger records decisions; it does not itself decide whether an
    update is benign.

## 7. Evaluation Plan

### 7.1 Baselines

FedAvg, Coordinate-wise Median, Trimmed Mean, Krum, Multi-Krum (global),
FLTrust (global).

### 7.2 Ablation of the proposed method

| Variant | Components |
|---|---|
| V1 | ABAC only (+ FedAvg) |
| V2 | V1 + norm clipping |
| V3 | V2 + global Multi-Krum + FLTrust (no cohorts) |
| V4 | V3 + cohorts |
| V5 (full) | V4 + capped inter-cohort weights + reputation feedback |

### 7.3 Experiment grid

| Factor | Values |
|---|---|
| Dataset | CICIoT2023 (main), N-BaIoT (validation) |
| Attack | Label flip, sign flip, Gaussian, scaling, ALIE, Min-Max |
| Malicious fraction | 0, 0.1, 0.2, 0.3, 0.4 |
| Placement | Spread, concentrated |
| Non-IID | α = 0.5 (main), α = 0.1 (hard) |
| Seeds | 3 (mean ± 95 % CI) |

Priority order for Colab runtime: (1) main grid on CICIoT2023 with
α = 0.5 and spread placement; (2) concentrated placement; (3) ablation;
(4) α = 0.1; (5) N-BaIoT validation.

### 7.4 Metrics

| Category | Metric |
|---|---|
| Detection | Accuracy, macro-F1, MCC, per-class recall |
| Attack impact | Accuracy/F1 drop vs. attack-free; attack success rate (label flip) |
| Screening | Malicious-client detection TPR / FPR per round |
| Cohorts | ARI between cohorts and true zones |
| Reputation | Rounds until attacker quarantine; false-quarantine rate |
| ABAC | Unauthorized-join block rate |
| Cost | Server time per round; ledger append/verify time |
| Clean premium | Proposed vs. FedAvg at 0 % attackers |

## 8. Implementation Environment

| Item | Choice |
|---|---|
| Platform | Google Colab (CPU sufficient; GPU optional) |
| FL framework | Plain PyTorch sequential simulation (no Flower/Ray overhead) |
| ABAC | PyCasbin |
| Clustering / metrics | scikit-learn |
| Ledger | Python `hashlib` hash chain + Merkle tree |
| Storage | Google Drive (data, results, checkpoints saved after each run) |

Notebook structure: setup → data loading and 8-class mapping → zone
partitioning → model → attacks → aggregation rules → ABAC + reputation
→ ledger → FL loop → experiment runner → plots and tables.

## 9. Design Principles

-   **Data locality:** raw traffic never leaves the client.
-   **Defence in depth:** access control, update screening and
    reputation act at different points.
-   **Peer-fair screening:** clients are compared with similar peers,
    reducing false rejection of honest non-IID clients.
-   **Bounded influence:** no single client or cohort can dominate the
    global update.
-   **Auditability:** every decision is recorded in a tamper-evident
    ledger.
-   **Reproducibility:** fixed seeds, saved configs and results.

## 10. Component Status

| Component | Status |
|---|---|
| Federated IDS pipeline (PyTorch, Colab) | Phase 1 — to implement |
| Baseline aggregators | Phase 1 — to implement |
| ABAC admission control (PyCasbin, simulated registry) | Phase 1 — to implement |
| Norm clipping + cohort formation | Phase 1 — to implement |
| Multi-Krum + client-level FLTrust within cohorts | Phase 1 — to implement |
| Inter-cohort capped aggregation + safeguards | Phase 1 — to implement |
| Reputation feedback to ABAC | Phase 1 — to implement |
| Hash-chain audit ledger | Phase 1 — to implement |
| Hyperledger Fabric ledger | Future work (simulated ledger used for now) |
| Edge-IIoTset / TON_IoT evaluation | Phase 2 |

## 11. Summary

Clients are first admitted by an ABAC policy that includes a
reputation attribute. Admitted updates are clipped, grouped into
similarity-based cohorts each round, and screened inside each cohort by
Multi-Krum and client-level FLTrust scoring. Cohort updates pass a
root-alignment check and are combined with capped weights to update the
global IoT IDS model. Screening results update each client's reputation,
which feeds back into the ABAC policy. All decisions are recorded in a
tamper-evident hash-chained ledger. The design is evaluated on
CICIoT2023 with zone-based non-IID partitioning and on N-BaIoT with
natural device partitioning, against six baselines and five ablation
variants.
