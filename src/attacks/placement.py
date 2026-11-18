"""Which clients are malicious (Architecture README 3, "Placement").

* **spread** – malicious clients are taken from the zones in turn
  (A, B, C, A, ...), so they are spread evenly over the federation.
* **concentrated** – the chosen zone is filled first (cohort-capture test);
  if more attackers are needed than that zone has clients, the rest are
  spread over the other zones.

The number of attackers is ``round(fraction * n_clients)``
(0 %, 10 %, 20 %, 30 %, 40 % of 20 clients -> 0, 2, 4, 6, 8).
"""

from __future__ import annotations

from typing import Sequence

from src.utils.seed import make_rng


def _round_robin(pools: list, n: int) -> list:
    """Take one client from each pool in turn until ``n`` are picked."""
    picked, pools = [], [list(p) for p in pools]
    while len(picked) < n and any(pools):
        for p in pools:
            if p and len(picked) < n:
                picked.append(p.pop(0))
    return picked


def select_malicious(client_groups: Sequence[str], fraction: float, placement: str,
                     seed: int, concentrated_zone: str = "A") -> list:
    """Return the sorted ids of malicious clients.

    Args:
        client_groups: zone of every client (index = client id).
        fraction: share of malicious clients.
        placement: ``"spread"`` or ``"concentrated"``.
        seed: base seed (the choice inside a zone is random but reproducible).
        concentrated_zone: zone filled first for ``"concentrated"``.
    """
    n_clients = len(client_groups)
    n_mal = int(round(fraction * n_clients))
    if n_mal == 0:
        return []
    rng = make_rng(seed, 6)
    zones = sorted(set(client_groups))
    members = {z: [int(c) for c in rng.permutation(
        [i for i, g in enumerate(client_groups) if g == z])] for z in zones}

    if placement == "spread":
        chosen = _round_robin([members[z] for z in zones], n_mal)
    elif placement == "concentrated":
        if concentrated_zone not in members:
            raise ValueError(f"Unknown zone '{concentrated_zone}'")
        first = members[concentrated_zone][:n_mal]
        rest = _round_robin([members[z] for z in zones if z != concentrated_zone], n_mal - len(first))
        chosen = first + rest
    else:
        raise ValueError(f"Unknown placement '{placement}'")
    return sorted(chosen)
