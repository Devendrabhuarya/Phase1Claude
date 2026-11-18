"""Local IDS model: a small MLP (Architecture README 6.1).

    n_features -> 128 -> 64 -> n_classes, ReLU, dropout 0.2

Input and output sizes are read from the data, so the same code works with
39 or 37 features and any number of classes.

FL code works on **flat weight vectors** (1-D tensors): an update is simply
``w_client - w_global``. The helpers below convert between a model and its
flat vector.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class MLP(nn.Module):
    """Fully connected network: [Linear -> ReLU -> Dropout] * hidden + Linear."""

    def __init__(self, n_features: int, n_classes: int, hidden_sizes: Sequence[int],
                 dropout: float):
        super().__init__()
        layers, prev = [], n_features
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(n_features: int, n_classes: int, model_cfg: dict) -> MLP:
    """Create the MLP from the ``model`` section of the config."""
    if model_cfg.get("activation", "relu") != "relu":
        raise ValueError("Only ReLU is implemented")
    return MLP(n_features, n_classes, model_cfg["hidden_sizes"], model_cfg["dropout"])


def get_weights(model: nn.Module) -> torch.Tensor:
    """Flat copy of all model parameters."""
    return parameters_to_vector(model.parameters()).detach().clone()


def set_weights(model: nn.Module, flat: torch.Tensor) -> None:
    """Load a flat weight vector into ``model`` (in place)."""
    vector_to_parameters(flat.detach().clone(), model.parameters())


def final_layer_slice(model: nn.Module) -> slice:
    """Position of the last layer's weight + bias inside the flat vector.

    Used for cohort formation (6.4): the final-layer update is cheap and the
    most class-informative part of an update.
    """
    sizes = [p.numel() for p in model.parameters()]
    last = list(model.modules())[-1]
    n_last = sum(p.numel() for p in last.parameters())
    total = sum(sizes)
    return slice(total - n_last, total)
