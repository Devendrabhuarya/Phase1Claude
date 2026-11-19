"""Median-norm clipping (6.3) and cohort formation (6.4).

Cohort formation, every round:

1. feature = flattened final-layer update of each admitted client;
2. cosine distance matrix between clients;
3. agglomerative clustering (average linkage) for K ∈ {2, 3}, the K with the
   best silhouette score wins;
4. clusters smaller than ``min_cohort_size`` (6) are merged into the nearest
   cohort (smallest average distance) until every cohort is large enough.

If there are fewer than 2 × min_size clients, everyone forms one cohort.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def median_norm_clip(updates: torch.Tensor) -> tuple:
    """Scale every update whose norm exceeds this round's median norm down to the median.

    Returns:
        (clipped_updates, median_norm, positions_that_were_clipped)
    """
    norms = updates.norm(dim=1)
    med = torch.quantile(norms, 0.5)
    factor = torch.clamp(med / norms.clamp_min(1e-12), max=1.0)
    clipped = (factor < 1.0).nonzero().flatten().tolist()
    return updates * factor[:, None], float(med), clipped


def cosine_distance_matrix(features: torch.Tensor) -> np.ndarray:
    """Pairwise cosine distance 1 − cos(a, b), clipped to [0, 2], zero diagonal."""
    f = features.double()
    f = f / f.norm(dim=1, keepdim=True).clamp_min(1e-12)
    d = (1.0 - f @ f.T).clamp(0.0, 2.0).numpy()
    np.fill_diagonal(d, 0.0)
    return (d + d.T) / 2


def _merge_small(labels: np.ndarray, dist: np.ndarray, min_size: int) -> np.ndarray:
    """Merge the smallest too-small cluster into its nearest cluster until all are >= min_size."""
    labels = labels.copy()
    while True:
        ids, counts = np.unique(labels, return_counts=True)
        if len(ids) <= 1 or counts.min() >= min_size:
            break
        small = ids[np.argmin(counts)]
        members = labels == small
        others = [c for c in ids if c != small]
        nearest = min(others, key=lambda c: dist[np.ix_(members, labels == c)].mean())
        labels[members] = nearest
    _, labels = np.unique(labels, return_inverse=True)       # relabel 0..K-1
    return labels


def form_cohorts(features: torch.Tensor, k_candidates: Sequence[int], min_size: int,
                 linkage: str = "average") -> tuple:
    """Cluster clients into cohorts.

    Returns:
        (labels, info) – labels[i] is the cohort of the i-th row; info has the
        chosen K, silhouette scores and the final number of cohorts.
    """
    n = len(features)
    if n < 2 * min_size:
        return np.zeros(n, dtype=int), {"k_chosen": 1, "silhouette": {}, "n_cohorts": 1,
                                        "reason": "too few clients for two cohorts"}
    dist = cosine_distance_matrix(features)
    best_k, best_s, best_labels, sil = None, -np.inf, None, {}
    for k in k_candidates:
        if not 2 <= k <= n - 1:
            continue
        labels = AgglomerativeClustering(n_clusters=k, metric="precomputed",
                                         linkage=linkage).fit_predict(dist)
        s = float(silhouette_score(dist, labels, metric="precomputed"))
        sil[k] = s
        if s > best_s:
            best_k, best_s, best_labels = k, s, labels
    labels = _merge_small(best_labels, dist, min_size)
    return labels, {"k_chosen": best_k, "silhouette": sil, "n_cohorts": int(labels.max() + 1)}
