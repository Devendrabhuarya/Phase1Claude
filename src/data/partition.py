"""Zone-based Dirichlet partitioning of the client pool (Architecture README 5.2).

Three hidden network zones emulate different IoT environments:

    Zone A (7 clients): DDoS, DoS                    dominant
    Zone B (7 clients): Recon, Spoofing, BruteForce  dominant
    Zone C (6 clients): Mirai, Web                   dominant

Procedure
---------
1. **Attack classes -> zones.** Each attack class has a *home* zone (where it
   is dominant). A share ``s`` of its rows goes to the home zone; the rest
   ``1 - s`` is spread over the other zones in proportion to their client
   counts. One ``s`` is used for all classes and is found by bisection so
   that, on average, ``zone_dominant_fraction`` (~75 %) of a zone's attack
   data comes from its dominant classes. Every class therefore exists in
   every zone, i.e. in the whole federation.
2. **Zone -> clients.** Inside a zone, the rows of each class are split
   among the zone's clients with proportions drawn from Dirichlet(alpha)
   (alpha = 0.5 main, 0.1 hard). This gives different class mixes and data
   quantities per client. A zone is re-drawn if a client would get fewer
   than ``min_samples`` attack rows.
3. **Benign.** Each client draws a target Benign fraction b ~ U(0.3, 0.5)
   and receives ``a * b / (1 - b)`` Benign rows for its ``a`` attack rows.
   If the pool does not hold enough Benign rows, attack rows are
   sub-sampled (equally for all clients) so the fractions still hold.
4. **Local split.** Each client's rows are shuffled and split into local
   train / validation sets.

Zones are ground truth for evaluation only (ARI); the server never uses them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.utils.seed import make_rng

BENIGN = "Benign"


@dataclass
class Partition:
    """Which pool rows belong to which client."""

    client_indices: list            # per client: indices into the pool (train + val)
    train_indices: list             # per client: local training indices into the pool
    val_indices: list               # per client: local validation indices into the pool
    client_groups: list             # per client: zone name (ground truth for ARI)
    info: dict = field(default_factory=dict)

    @property
    def n_clients(self) -> int:
        return len(self.client_indices)


# ------------------------------------------------------------------ helpers


def _split_by_shares(idx: np.ndarray, shares: np.ndarray) -> list:
    """Split ``idx`` into consecutive parts with sizes proportional to ``shares``."""
    shares = np.asarray(shares, dtype=float)
    cuts = np.round(np.cumsum(shares / shares.sum()) * len(idx)).astype(int)[:-1]
    return np.split(idx, cuts)


def _zone_shares(home: str | None, zones: Sequence[str], weights: np.ndarray, s: float) -> np.ndarray:
    """Share of one attack class going to each zone for home share ``s``."""
    if home is None:                       # class without a home zone: spread by weight
        return weights.copy()
    h = zones.index(home)
    out = np.zeros(len(zones))
    others = [i for i in range(len(zones)) if i != h]
    out[others] = (1.0 - s) * weights[others] / weights[others].sum()
    out[h] = s
    return out


def _dominance(s: float, counts: Mapping[str, int], homes: Mapping[str, str | None],
               zones: Sequence[str], weights: np.ndarray) -> np.ndarray:
    """Per-zone fraction of attack rows that come from the zone's dominant classes."""
    dom, tot = np.zeros(len(zones)), np.zeros(len(zones))
    for cls, n in counts.items():
        shares = _zone_shares(homes[cls], zones, weights, s) * n
        tot += shares
        if homes[cls] is not None:
            dom[zones.index(homes[cls])] += shares[zones.index(homes[cls])]
    return np.divide(dom, tot, out=np.zeros_like(dom), where=tot > 0)


def solve_home_share(counts, homes, zones, weights, target: float, iters: int = 60) -> float:
    """Bisection for the home share ``s`` whose mean zone dominance equals ``target``."""
    lo, hi = 0.0, 1.0
    if _dominance(hi, counts, homes, zones, weights).mean() <= target:
        return hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        if _dominance(mid, counts, homes, zones, weights).mean() < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _local_split(idx: np.ndarray, val_fraction: float, rng: np.random.Generator):
    idx = rng.permutation(idx)
    n_val = max(1, int(round(val_fraction * len(idx)))) if len(idx) > 1 else 0
    return np.sort(idx[n_val:]), np.sort(idx[:n_val])


# ------------------------------------------------------------------ main function


def zone_dirichlet_partition(
    y: np.ndarray,
    class_names: Sequence[str],
    zones_cfg: Mapping[str, Mapping],
    alpha: float,
    benign_fraction_range: tuple,
    dominant_fraction: float,
    min_samples: int,
    val_fraction: float,
    seed: int,
    max_tries: int = 200,
    verbose: bool = True,
) -> Partition:
    """Split pool labels ``y`` into zone-based non-IID clients (see module docstring).

    Raises:
        ValueError: on an invalid zone config, if a zone cannot be drawn with
            ``min_samples`` attack rows per client, or if any client ends up
            empty or without Benign data.
    """
    rng = make_rng(seed, 3)
    zones = list(zones_cfg)
    n_per_zone = np.array([zones_cfg[z]["n_clients"] for z in zones])
    weights = n_per_zone / n_per_zone.sum()
    benign_id = list(class_names).index(BENIGN)
    attack_classes = [c for c in class_names if c != BENIGN]

    homes: dict = {c: None for c in attack_classes}
    for z in zones:
        for c in zones_cfg[z]["dominant"]:
            if c not in homes:
                raise ValueError(f"Zone {z}: unknown dominant class '{c}'")
            if homes[c] is not None:
                raise ValueError(f"Class '{c}' is dominant in two zones")
            homes[c] = z

    cls_idx = {c: np.flatnonzero(y == i) for i, c in enumerate(class_names)}
    counts = {c: len(cls_idx[c]) for c in attack_classes if len(cls_idx[c]) > 0}
    s = solve_home_share(counts, homes, zones, weights, dominant_fraction)
    dominance = _dominance(s, counts, homes, zones, weights)

    # 1) attack rows of each class -> zones
    zone_rows = {z: {} for z in zones}
    for c in counts:
        parts = _split_by_shares(rng.permutation(cls_idx[c]), _zone_shares(homes[c], zones, weights, s))
        for z, part in zip(zones, parts):
            zone_rows[z][c] = part

    # 2) zone -> clients with Dirichlet(alpha) per class
    client_attack, client_groups = [], []
    for z, k in zip(zones, n_per_zone):
        for _ in range(max_tries):
            per_client = [[] for _ in range(k)]
            for c, rows in zone_rows[z].items():
                props = rng.dirichlet(np.full(k, alpha))
                for j, part in enumerate(_split_by_shares(rows, props)):
                    per_client[j].append(part)
            per_client = [np.concatenate(p) if p else np.array([], dtype=int) for p in per_client]
            if min(len(p) for p in per_client) >= min_samples:
                break
        else:
            raise ValueError(f"Zone {z}: could not give every client >= {min_samples} attack rows "
                             f"in {max_tries} Dirichlet draws (alpha={alpha}).")
        client_attack.extend(per_client)
        client_groups.extend([z] * k)

    # 3) Benign: each client gets fraction b ~ U(lo, hi) of its data
    lo, hi = benign_fraction_range
    b = rng.uniform(lo, hi, size=len(client_attack))
    a = np.array([len(p) for p in client_attack], dtype=float)
    need = a * b / (1 - b)
    benign_pool = rng.permutation(cls_idx[BENIGN])
    keep_attack = 1.0
    if need.sum() > len(benign_pool):
        keep_attack = len(benign_pool) / need.sum()
        client_attack = [rng.choice(p, size=int(len(p) * keep_attack), replace=False)
                         for p in client_attack]
        a = np.array([len(p) for p in client_attack], dtype=float)
        need = a * b / (1 - b)
        if verbose:
            print(f"WARNING: not enough Benign rows; kept {keep_attack:.1%} of attack rows "
                  f"so every client still gets {lo:.0%}-{hi:.0%} Benign.")
    need = np.maximum(1, np.floor(need).astype(int))
    starts = np.concatenate([[0], np.cumsum(need)])
    client_benign = [benign_pool[starts[i]:starts[i + 1]] for i in range(len(need))]

    # 4) combine + local split
    client_idx, train_idx, val_idx = [], [], []
    for att, ben in zip(client_attack, client_benign):
        idx = np.sort(np.concatenate([att, ben]))
        tr, va = _local_split(idx, val_fraction, rng)
        client_idx.append(idx)
        train_idx.append(tr)
        val_idx.append(va)

    part = Partition(client_idx, train_idx, val_idx, client_groups, info={
        "scheme": "zone_dirichlet", "alpha": alpha, "home_share": float(s),
        "zone_dominance": {z: float(d) for z, d in zip(zones, dominance)},
        "attack_rows_kept": float(keep_attack),
        "benign_unused": int(len(benign_pool) - starts[-1]),
    })
    validate_partition(part, y, benign_id)
    return part


def validate_partition(part: Partition, y: np.ndarray, benign_id: int) -> None:
    """Raise if a client is empty, lacks Benign, or two clients share a row."""
    all_idx = np.concatenate(part.client_indices)
    if len(all_idx) != len(np.unique(all_idx)):
        raise ValueError("A pool row was assigned to more than one client.")
    for i, idx in enumerate(part.client_indices):
        if len(idx) == 0 or len(part.train_indices[i]) == 0:
            raise ValueError(f"Client {i} is empty.")
        if not (y[idx] == benign_id).any():
            raise ValueError(f"Client {i} has no Benign data.")


def partition_from_config(y: np.ndarray, class_names: Sequence[str], cfg: dict,
                          alpha: float | None = None, verbose: bool = True) -> Partition:
    """Run the partition scheme named in ``cfg['partition']['scheme']``."""
    p = cfg["partition"]
    if p["scheme"] != "zone_dirichlet":
        raise ValueError(f"Unknown partition scheme {p['scheme']}")
    return zone_dirichlet_partition(
        y, class_names, p["zones"],
        alpha=p["dirichlet_alpha"] if alpha is None else alpha,
        benign_fraction_range=(p["benign_fraction_min"], p["benign_fraction_max"]),
        dominant_fraction=p["zone_dominant_fraction"],
        min_samples=p["min_samples_per_client"],
        val_fraction=cfg["data"]["client_val_fraction"],
        seed=cfg["seed"], verbose=verbose,
    )


def client_class_table(part: Partition, y: np.ndarray, class_names: Sequence[str]) -> pd.DataFrame:
    """Per-client class counts, with zone, total and Benign fraction."""
    k = len(class_names)
    rows = [np.bincount(y[idx], minlength=k) for idx in part.client_indices]
    df = pd.DataFrame(rows, columns=list(class_names),
                      index=[f"client{i:02d}" for i in range(part.n_clients)])
    df.insert(0, "zone", part.client_groups)
    df["total"] = df[list(class_names)].sum(axis=1)
    df["benign_frac"] = (df[BENIGN] / df["total"]).round(3)
    return df


def zone_class_table(part: Partition, y: np.ndarray, class_names: Sequence[str]) -> pd.DataFrame:
    """Class counts summed per zone (to check dominance)."""
    t = client_class_table(part, y, class_names)
    return t.groupby("zone")[list(class_names) + ["total"]].sum()
