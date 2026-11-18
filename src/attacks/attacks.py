"""Poisoning attacks (Architecture README 3, "Attacks evaluated").

Every attack is an object with

* ``malicious_ids`` – the attacking clients;
* ``poison_data(cid, X, y)`` – change a malicious client's local data once,
  before training (data poisoning);
* ``poison_updates(updates, malicious_mask, rng)`` – change the malicious
  rows of the stacked update matrix after local training (model poisoning).

The server loop (``src/fl/server.py``) calls both hooks. Inputs are never
modified in place.

| Attack       | What the malicious clients send                                   |
|--------------|-------------------------------------------------------------------|
| label_flip   | honest training on data whose attack labels are set to Benign     |
| sign_flip    | −scale · Δ                                                        |
| gaussian     | pure noise N(0, std²)                                             |
| scaling      | factor · Δ                                                        |
| alie         | μ − z·σ of the benign updates (Baruch et al., 2019), same for all |
| minmax       | μ − γ·p with the largest γ whose distance to every benign update  |
|              | stays within the largest benign pairwise distance                 |
|              | (Shejwalkar & Houmansadr, 2021), same for all                     |

ALIE and Min-Max are adaptive: they know the statistics of this round's
benign updates; colluding clients send the same vector.
"""

from __future__ import annotations

import math
from statistics import NormalDist
from typing import Optional, Sequence

import numpy as np
import torch


class Attack:
    """No-op base class (honest behaviour)."""

    name = "none"

    def __init__(self, malicious_ids: Sequence[int]):
        self.malicious_ids = sorted(int(i) for i in malicious_ids)

    def poison_data(self, cid: int, X: np.ndarray, y: np.ndarray):
        return X, y

    def poison_updates(self, updates: torch.Tensor, malicious_mask: torch.Tensor,
                       rng: np.random.Generator) -> torch.Tensor:
        return updates


# ------------------------------------------------------------------ data poisoning


class LabelFlip(Attack):
    """Data poisoning: every attack label becomes Benign (attacks look harmless)."""

    name = "label_flip"

    def __init__(self, malicious_ids, benign_id: int):
        super().__init__(malicious_ids)
        self.benign_id = benign_id

    def poison_data(self, cid, X, y):
        y = np.array(y, copy=True)
        y[y != self.benign_id] = self.benign_id
        return X, y


# ------------------------------------------------------------------ simple model poisoning


class SignFlip(Attack):
    """Model poisoning: send −scale · Δ (pushes the model the opposite way)."""

    name = "sign_flip"

    def __init__(self, malicious_ids, scale: float = 1.0):
        super().__init__(malicious_ids)
        self.scale = scale

    def poison_updates(self, updates, malicious_mask, rng):
        out = updates.clone()
        out[malicious_mask] = -self.scale * updates[malicious_mask]
        return out


class GaussianNoise(Attack):
    """Model poisoning: replace Δ by random noise N(0, std²)."""

    name = "gaussian"

    def __init__(self, malicious_ids, std: float = 1.0):
        super().__init__(malicious_ids)
        self.std = std

    def poison_updates(self, updates, malicious_mask, rng):
        out = updates.clone()
        k = int(malicious_mask.sum())
        noise = rng.normal(0.0, self.std, size=(k, updates.shape[1]))
        out[malicious_mask] = torch.as_tensor(noise, dtype=updates.dtype)
        return out


class Scaling(Attack):
    """Model poisoning: boost Δ by a large factor."""

    name = "scaling"

    def __init__(self, malicious_ids, factor: float = 10.0):
        super().__init__(malicious_ids)
        self.factor = factor

    def poison_updates(self, updates, malicious_mask, rng):
        out = updates.clone()
        out[malicious_mask] = self.factor * updates[malicious_mask]
        return out


# ------------------------------------------------------------------ adaptive attacks


def alie_z(n_clients: int, n_malicious: int) -> float:
    """ALIE z_max (Baruch et al.): s = floor(n/2 + 1) − f, z = Φ⁻¹((n − s) / n)."""
    s = math.floor(n_clients / 2 + 1) - n_malicious
    p = (n_clients - s) / n_clients
    p = min(max(p, 1e-6), 1 - 1e-6)
    return NormalDist().inv_cdf(p)


class ALIE(Attack):
    """"A Little Is Enough": small coordinated shift inside the benign variance."""

    name = "alie"

    def __init__(self, malicious_ids, z: Optional[float] = None):
        super().__init__(malicious_ids)
        self.z = z

    def poison_updates(self, updates, malicious_mask, rng):
        benign = updates[~malicious_mask]
        mu, sigma = benign.mean(dim=0), benign.std(dim=0, unbiased=False)
        z = self.z if self.z is not None else alie_z(len(updates), int(malicious_mask.sum()))
        out = updates.clone()
        out[malicious_mask] = mu - z * sigma
        self.last_z = z
        return out


def _perturbation(mu: torch.Tensor, benign: torch.Tensor, kind: str) -> torch.Tensor:
    if kind == "std":
        return benign.std(dim=0, unbiased=False)
    if kind == "unit_vec":
        return mu / (mu.norm() + 1e-12)
    if kind == "sign":
        return torch.sign(mu)
    raise ValueError(f"Unknown Min-Max perturbation '{kind}'")


class MinMax(Attack):
    """Min-Max: largest deviation whose max distance to benign updates stays
    within the largest benign pairwise distance (so distance rules can't tell)."""

    name = "minmax"

    def __init__(self, malicious_ids, gamma_init: float = 10.0, tol: float = 1e-5,
                 perturbation: str = "std"):
        super().__init__(malicious_ids)
        self.gamma_init, self.tol, self.perturbation = gamma_init, tol, perturbation

    def poison_updates(self, updates, malicious_mask, rng):
        benign = updates[~malicious_mask].double()
        mu = benign.mean(dim=0)
        p = _perturbation(mu, benign, self.perturbation)
        max_benign_dist = torch.cdist(benign, benign).max()

        def ok(gamma: float) -> bool:
            mal = mu - gamma * p
            return bool((benign - mal).norm(dim=1).max() <= max_benign_dist)

        # binary search from the paper (Algorithm 1)
        gamma, step, gamma_ok = self.gamma_init, self.gamma_init / 2, 0.0
        while abs(gamma_ok - gamma) > self.tol:
            if ok(gamma):
                gamma_ok = gamma
                gamma += step
            else:
                gamma -= step
            step /= 2
        out = updates.clone()
        out[malicious_mask] = (mu - gamma_ok * p).to(updates.dtype)
        self.last_gamma = gamma_ok
        return out


# ------------------------------------------------------------------ factory


def get_attack(cfg: dict, malicious_ids: Sequence[int], benign_id: int) -> Optional[Attack]:
    """Build the attack named in ``cfg['attacks']['type']`` (``None`` for no attack)."""
    a = cfg["attacks"]
    kind = a["type"]
    if kind in (None, "none") or not malicious_ids:
        return None
    if kind == "label_flip":
        return LabelFlip(malicious_ids, benign_id)
    if kind == "sign_flip":
        return SignFlip(malicious_ids, a["sign_flip_scale"])
    if kind == "gaussian":
        return GaussianNoise(malicious_ids, a["gaussian_std"])
    if kind == "scaling":
        return Scaling(malicious_ids, a["scaling_factor"])
    if kind == "alie":
        return ALIE(malicious_ids, a["alie_z"])
    if kind == "minmax":
        return MinMax(malicious_ids, a["minmax_gamma_init"], a["minmax_tol"], a["minmax_perturbation"])
    raise ValueError(f"Unknown attack '{kind}'")
