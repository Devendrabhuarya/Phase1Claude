"""Proposed cohort-aware two-stage screening (Architecture README 6.3–6.7).

Pipeline for the updates of the clients that passed ABAC:

    median-norm clipping (6.3)
      -> cohort formation on final-layer updates (6.4, every round)
      -> per cohort: Multi-Krum (6.5 A) then client-level FLTrust (6.5 B)
      -> inter-cohort: root-alignment check, trust-proportional weights,
         cap w_max = 0.5 (6.6)
      -> per-client reputation scores sᵢ for the ABAC loop (6.7)

Ablation variants (7.2) switch parts on:

| Variant | ABAC | clipping | MK + FLTrust | cohorts | cap + reputation |
|---------|------|----------|--------------|---------|------------------|
| V1      |  ✓   |          |              |         |                  |
| V2      |  ✓   |    ✓     |              |         |                  |
| V3      |  ✓   |    ✓     |  ✓ (global)  |         |                  |
| V4      |  ✓   |    ✓     |      ✓       |    ✓    |                  |
| V5 full |  ✓   |    ✓     |      ✓       |    ✓    |        ✓         |

(ABAC itself runs in the server before training: ``src/abac/admission.py``.)
"""

from __future__ import annotations

import numpy as np
import torch

from src.aggregators import register
from src.aggregators.base import Aggregator
from src.aggregators.cohorts import form_cohorts, median_norm_clip
from src.aggregators.robust import default_f, fltrust_combine, fltrust_scores, multikrum_select

VARIANTS = {
    "V1": dict(abac=True, clipping=False, screening=False, cohorts=False, cap=False, reputation=False),
    "V2": dict(abac=True, clipping=True, screening=False, cohorts=False, cap=False, reputation=False),
    "V3": dict(abac=True, clipping=True, screening=True, cohorts=False, cap=False, reputation=False),
    "V4": dict(abac=True, clipping=True, screening=True, cohorts=True, cap=False, reputation=False),
    "V5": dict(abac=True, clipping=True, screening=True, cohorts=True, cap=True, reputation=True),
}


def variant_flags(cfg: dict) -> dict:
    v = cfg.get("ablation", {}).get("variant", "V5")
    if v not in VARIANTS:
        raise ValueError(f"Unknown ablation variant '{v}'")
    return dict(VARIANTS[v])


def capped_weights(raw: np.ndarray, cap: float) -> np.ndarray:
    """Normalise ``raw`` to sum 1 so that no weight exceeds ``cap``.

    Water-filling: capped weights are fixed at ``cap`` and the remaining mass is
    shared by the others in proportion to their raw weights, repeated until no
    weight exceeds the cap. If the cap is infeasible (K · cap < 1, e.g. a single
    cohort) the weights are made equal, which is the closest feasible choice.
    """
    raw = np.asarray(raw, dtype=float)
    k = len(raw)
    if k == 0:
        return raw
    if k * cap < 1.0:
        return np.full(k, 1.0 / k)
    w = raw / raw.sum()
    fixed = np.zeros(k, dtype=bool)
    while (w > cap + 1e-12).any():
        fixed |= w > cap + 1e-12
        free = ~fixed
        remaining = 1.0 - cap * fixed.sum()
        w = np.where(fixed, cap, 0.0)
        if free.any():
            w[free] = remaining * raw[free] / raw[free].sum()
    return w


def screen_group(updates: torch.Tensor, g0: torch.Tensor) -> dict:
    """Multi-Krum then FLTrust inside one group (cohort or all clients).

    Returns positions (within the group) kept by Multi-Krum, their trust
    scores, the group update (or None if Σ TS = 0) and the total trust.
    """
    n = len(updates)
    f = default_f(n)
    keep = multikrum_select(updates, f, n - f) if n > 2 else list(range(n))
    ts, normalised = fltrust_scores(updates[keep], g0)
    return {"f": f, "kept": keep, "ts": ts, "update": fltrust_combine(ts, normalised),
            "total_trust": float(ts.sum())}


@register
class Proposed(Aggregator):
    """Cohort-aware Multi-Krum + FLTrust with capped inter-cohort weights (variant V1–V5)."""

    name = "proposed"

    def __init__(self, cfg: dict | None = None):
        super().__init__(cfg)
        self.flags = variant_flags(self.cfg) if self.cfg else dict(VARIANTS["V5"])

    def aggregate(self, updates, global_weights, context):
        ids = list(context.get("client_ids", range(len(updates))))
        n = len(updates)
        info = {"variant": self.cfg.get("ablation", {}).get("variant", "V5")}

        # 6.3 median-norm clipping
        if self.flags["clipping"]:
            updates, med, clipped = median_norm_clip(updates)
            info["clip_median_norm"], info["clipped"] = med, [ids[i] for i in clipped]

        # V1 / V2: no screening -> sample-weighted FedAvg of the (clipped) updates
        if not self.flags["screening"]:
            w = torch.as_tensor(context["n_samples"], dtype=updates.dtype)
            w = w / w.sum()
            info["selected"] = ids
            return (w[:, None] * updates).sum(dim=0), info

        g0 = context["root_update"]().to(updates.dtype)

        # 6.4 cohorts (V4, V5) or one global group (V3)
        if self.flags["cohorts"]:
            c = self.cfg["cohorts"]
            labels, cinfo = form_cohorts(updates[:, context["final_layer"]], c["k_candidates"],
                                         c["min_cohort_size"], c["linkage"])
            info.update({f"cohort_{k}": v for k, v in cinfo.items()})
        else:
            labels = np.zeros(n, dtype=int)
        info["cohorts"] = {ids[i]: int(labels[i]) for i in range(n)}

        # 6.5 intra-cohort screening
        scores = {cid: 0.0 for cid in ids}          # sᵢ = 0 unless kept by Multi-Krum
        trust, mk_kept, cohort_updates, cohort_trust, cohort_members = {}, {}, {}, {}, {}
        dropped = {}
        for k in sorted(set(labels.tolist())):
            pos = np.flatnonzero(labels == k)
            res = screen_group(updates[pos], g0)
            kept = [ids[pos[j]] for j in res["kept"]]
            mk_kept[k] = kept
            for cid, t in zip(kept, res["ts"].tolist()):
                trust[cid] = t
                scores[cid] = t
            if res["update"] is None:
                dropped[k] = "zero total trust"
                continue
            cohort_updates[k], cohort_trust[k] = res["update"], res["total_trust"]
            cohort_members[k] = [cid for cid in kept if trust[cid] > 0]

        # 6.6 inter-cohort aggregation
        if self.flags["cohorts"]:
            for k in list(cohort_updates):
                cos = float(torch.nn.functional.cosine_similarity(cohort_updates[k], g0, dim=0))
                if cos <= 0:
                    dropped[k] = f"root misaligned (cos={cos:.3f})"
                    del cohort_updates[k], cohort_trust[k], cohort_members[k]
        survivors = sorted(cohort_updates)
        if survivors:
            raw = np.array([cohort_trust[k] for k in survivors])
            w = (capped_weights(raw, self.cfg["inter_cohort"]["weight_cap"]) if self.flags["cap"]
                 else raw / raw.sum())
            new_update = sum(float(wk) * cohort_updates[k] for wk, k in zip(w, survivors))
            weights = {k: float(wk) for k, wk in zip(survivors, w)}
        else:
            new_update, weights = torch.zeros_like(global_weights), {}

        info.update({
            "selected": sorted(c for k in survivors for c in cohort_members[k]),
            "multikrum_kept": mk_kept, "trust_scores": trust, "dropped_cohorts": dropped,
            "cohort_weights": weights, "round_scores": scores,
        })
        return new_update, info
