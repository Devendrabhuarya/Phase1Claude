"""Client-side local training (Architecture README 6.1).

A client receives the global weights, trains for ``local_epochs`` on its own
data with class-weighted cross-entropy and SGD (lr 0.01, momentum 0.9,
batch 256), and returns only its update  Δ = w_client − w_global.
Raw traffic never leaves the client.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.models.mlp import get_weights, set_weights


def class_weights(y: np.ndarray, n_classes: int, max_weight: float) -> torch.Tensor:
    """Balanced weights n / (k * n_c) over the classes present locally, capped at ``max_weight``.

    Classes absent from the client get weight 1 (they never appear in its loss).
    """
    counts = np.bincount(y, minlength=n_classes).astype(float)
    present = counts > 0
    w = np.ones(n_classes)
    w[present] = counts.sum() / (present.sum() * counts[present])
    return torch.tensor(np.minimum(w, max_weight), dtype=torch.float32)


def train_local(model: nn.Module, X: np.ndarray, y: np.ndarray, training_cfg: dict,
                n_classes: int, seed: int, epochs: int | None = None,
                batch_size: int | None = None) -> None:
    """Train ``model`` in place on (X, y) with the settings in ``training_cfg``."""
    t = training_cfg
    epochs = t["local_epochs"] if epochs is None else epochs
    batch_size = t["batch_size"] if batch_size is None else batch_size
    weight = class_weights(y, n_classes, t["class_weight_max"]) if t["class_weighted"] else None
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    if t["optimizer"] != "sgd":
        raise ValueError("Only SGD is implemented")
    opt = torch.optim.SGD(model.parameters(), lr=t["lr"], momentum=t["momentum"])

    Xt = torch.as_tensor(X, dtype=torch.float32)
    yt = torch.as_tensor(y, dtype=torch.long)
    gen = torch.Generator().manual_seed(int(seed))
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(Xt), generator=gen)
        for start in range(0, len(Xt), batch_size):
            batch = order[start:start + batch_size]
            opt.zero_grad()
            loss_fn(model(Xt[batch]), yt[batch]).backward()
            opt.step()


def client_update(model: nn.Module, global_weights: torch.Tensor, X: np.ndarray, y: np.ndarray,
                  training_cfg: dict, n_classes: int, seed: int, **kwargs) -> torch.Tensor:
    """Load the global weights into ``model``, train locally and return Δ = w_local − w_global."""
    set_weights(model, global_weights)
    torch.manual_seed(int(seed))          # dropout masks
    train_local(model, X, y, training_cfg, n_classes, seed, **kwargs)
    return get_weights(model) - global_weights
