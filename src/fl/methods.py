"""Build the server-side defence for a config: aggregator + optional ABAC admission."""

from __future__ import annotations

from src.abac.admission import AdmissionController
from src.aggregators import get_aggregator
from src.aggregators.proposed import variant_flags


def build_method(cfg: dict, n_clients: int):
    """Return ``(aggregator, admission)``.

    Baselines run without ABAC (``admission = None``). The proposed method
    uses ABAC in every variant; reputation feedback only in V5.
    """
    name = cfg["aggregator"]["name"]
    agg = get_aggregator(name, cfg)
    if name != "proposed" or not cfg["abac"]["enabled"]:
        return agg, None
    flags = variant_flags(cfg)
    use_rep = flags["reputation"] and cfg["reputation"]["enabled"]
    return agg, AdmissionController(cfg, n_clients, use_reputation=use_rep)
