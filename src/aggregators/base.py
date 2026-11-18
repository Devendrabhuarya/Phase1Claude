"""Common aggregator interface and FedAvg.

Every aggregator implements::

    aggregate(updates, global_weights, context) -> (new_update, info)

* ``updates``        – tensor (n_clients, n_params), one flat update Δᵢ per row
* ``global_weights`` – flat tensor of the current global model
* ``context``        – dict with round information, e.g.
    ``round``, ``client_ids`` (list), ``n_samples`` (tensor),
    ``root_update`` (callable returning the server's root update g₀),
    ``final_layer`` (slice of the last layer in the flat vector)
* ``new_update``     – flat tensor added to the global weights
* ``info``           – dict for logging/ledger: ``selected`` client ids,
    ``scores``, ``cohorts`` ... (whatever the rule produces)
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class Aggregator(ABC):
    """Base class; subclasses set ``name`` and implement :meth:`aggregate`."""

    name = "base"

    def __init__(self, cfg: dict | None = None):
        self.cfg = cfg or {}

    @abstractmethod
    def aggregate(self, updates: torch.Tensor, global_weights: torch.Tensor,
                  context: dict) -> tuple:
        """Return ``(new_update, info)``."""


class FedAvg(Aggregator):
    """Federated averaging: mean of updates weighted by local sample counts."""

    name = "fedavg"

    def aggregate(self, updates, global_weights, context):
        n = context.get("n_samples")
        w = torch.ones(len(updates)) if n is None else torch.as_tensor(n, dtype=torch.float32)
        w = w / w.sum()
        return (w[:, None] * updates).sum(dim=0), {
            "selected": list(context.get("client_ids", range(len(updates)))),
            "weights": w.tolist(),
        }
