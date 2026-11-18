"""Baseline robust aggregators (Architecture README 7.1).

All follow the common interface ``aggregate(updates, global_weights, context)
-> (new_update, info)`` (see ``base.py``). ``info["selected"]`` lists the
clients whose updates were used; it is ``None`` for coordinate-wise rules
(Median, Trimmed Mean) that do not accept/reject whole clients.

The building blocks :func:`krum_scores`, :func:`multikrum_select` and
:func:`fltrust_scores` are reused by the proposed method (Stage 8).
"""

from __future__ import annotations

import math

import torch

from src.aggregators import register
from src.aggregators.base import Aggregator


def default_f(n: int) -> int:
    """Assumed number of attackers f = floor((n − 3) / 2) (largest f with n > 2f + 2)."""
    return max(0, (n - 3) // 2)


# ------------------------------------------------------------------ coordinate-wise rules


@register
class Median(Aggregator):
    """Coordinate-wise median of the updates (Yin et al., 2018)."""

    name = "median"

    def aggregate(self, updates, global_weights, context):
        return torch.quantile(updates, 0.5, dim=0), {"selected": None}


def trimmed_mean(updates: torch.Tensor, beta: float) -> tuple:
    """Per coordinate: drop the k = floor(beta·n) largest and smallest values, average the rest."""
    n = len(updates)
    k = int(math.floor(beta * n))
    if n - 2 * k < 1:
        raise ValueError(f"Trimmed mean: beta={beta} removes all {n} updates")
    s, _ = torch.sort(updates, dim=0)
    return s[k:n - k].mean(dim=0), k


@register
class TrimmedMean(Aggregator):
    """Coordinate-wise trimmed mean (Yin et al., 2018)."""

    name = "trimmed_mean"

    def aggregate(self, updates, global_weights, context):
        beta = self.cfg.get("aggregator", {}).get("trimmed_mean_beta", 0.1)
        out, k = trimmed_mean(updates, beta)
        return out, {"selected": None, "trimmed_per_side": k}


# ------------------------------------------------------------------ Krum family


def krum_scores(updates: torch.Tensor, f: int) -> torch.Tensor:
    """Krum score of each update: sum of squared distances to its n − f − 2 nearest neighbours."""
    n = len(updates)
    k = max(1, n - f - 2)
    k = min(k, n - 1)
    d2 = torch.cdist(updates.double(), updates.double()) ** 2
    d2.fill_diagonal_(float("inf"))                    # exclude the update itself
    nearest, _ = torch.topk(d2, k, dim=1, largest=False)
    return nearest.sum(dim=1)


def multikrum_select(updates: torch.Tensor, f: int, m: int) -> list:
    """Positions of the ``m`` updates with the lowest Krum scores."""
    if len(updates) <= 1:
        return list(range(len(updates)))
    scores = krum_scores(updates, f)
    return sorted(torch.argsort(scores)[:m].tolist())


def _resolve_f(cfg: dict, n: int) -> int:
    f = cfg.get("aggregator", {}).get("krum_f")
    return default_f(n) if f is None else int(f)


@register
class Krum(Aggregator):
    """Krum (Blanchard et al., 2017): use the single update with the lowest score."""

    name = "krum"

    def aggregate(self, updates, global_weights, context):
        ids = list(context.get("client_ids", range(len(updates))))
        f = _resolve_f(self.cfg, len(updates))
        scores = krum_scores(updates, f)
        best = int(torch.argmin(scores))
        return updates[best].clone(), {"selected": [ids[best]], "f": f,
                                       "scores": dict(zip(ids, scores.tolist()))}


@register
class MultiKrum(Aggregator):
    """Multi-Krum: average of the m = n − f updates with the lowest Krum scores."""

    name = "multikrum"

    def aggregate(self, updates, global_weights, context):
        ids = list(context.get("client_ids", range(len(updates))))
        n = len(updates)
        f = _resolve_f(self.cfg, n)
        keep = multikrum_select(updates, f, n - f)
        return updates[keep].mean(dim=0), {"selected": [ids[i] for i in keep], "f": f}


# ------------------------------------------------------------------ FLTrust


def fltrust_scores(updates: torch.Tensor, g0: torch.Tensor) -> tuple:
    """FLTrust trust scores and magnitude-normalised updates.

    TSᵢ = ReLU(cos(Δᵢ, g₀));   Δᵢ' = Δᵢ · ‖g₀‖ / ‖Δᵢ‖
    """
    g0 = g0.to(updates.dtype)
    norms = updates.norm(dim=1).clamp_min(1e-12)
    cos = (updates @ g0) / (norms * g0.norm().clamp_min(1e-12))
    ts = torch.relu(cos)
    normalised = updates * (g0.norm() / norms)[:, None]
    return ts, normalised


def fltrust_combine(ts: torch.Tensor, normalised: torch.Tensor) -> torch.Tensor | None:
    """Σ TSᵢ Δᵢ' / Σ TSᵢ, or ``None`` if every trust score is zero."""
    total = ts.sum()
    if total <= 0:
        return None
    return (ts[:, None] * normalised).sum(dim=0) / total


@register
class FLTrust(Aggregator):
    """FLTrust (Cao et al., 2021) over all clients, using the server's root update g₀."""

    name = "fltrust"

    def aggregate(self, updates, global_weights, context):
        ids = list(context.get("client_ids", range(len(updates))))
        ts, normalised = fltrust_scores(updates, context["root_update"]())
        out = fltrust_combine(ts, normalised)
        if out is None:                                     # nobody trusted: keep the model
            out = torch.zeros_like(global_weights)
        return out, {"selected": [i for i, t in zip(ids, ts.tolist()) if t > 0],
                     "trust_scores": dict(zip(ids, ts.tolist()))}
