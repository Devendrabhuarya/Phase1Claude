"""Reputation update and quarantine (Architecture README 6.7).

* Per-round score  sᵢ = TSᵢ if client i survived Multi-Krum, else 0.
* Reputation (EMA) rᵢ ← β·rᵢ + (1 − β)·sᵢ,  β = 0.8,  initial rᵢ = 0.5.
* If rᵢ < τ = 0.3 the client is quarantined for 5 rounds; afterwards it is
  re-admitted on probation with rᵢ = τ.

Reputation lives in the ABAC attribute registry, so the next round's
admission decision uses it (closed loop).
"""

from __future__ import annotations

from typing import Mapping

from src.abac.registry import AttributeRegistry


class ReputationManager:
    """Updates reputations and quarantine fields of an :class:`AttributeRegistry`."""

    def __init__(self, registry: AttributeRegistry, beta: float, threshold: float,
                 quarantine_rounds: int, probation_value: float):
        self.registry = registry
        self.beta, self.tau = beta, threshold
        self.quarantine_rounds, self.probation_value = quarantine_rounds, probation_value
        self._in_quarantine: set = set()
        self.events: list = []           # (round, client_id, "quarantined" | "released")

    def release_expired(self, rnd: int) -> list:
        """Start of round ``rnd``: clients whose quarantine ended return on probation (r = τ)."""
        released = []
        for cid in sorted(self._in_quarantine):
            rec = self.registry.get(cid)
            if rec.quarantined_until < rnd:
                rec.reputation = self.probation_value
                self._in_quarantine.discard(cid)
                released.append(cid)
                self.events.append((rnd, cid, "released"))
        return released

    def update(self, rnd: int, scores: Mapping[str, float]) -> list:
        """End of round ``rnd``: EMA update for every scored client; returns newly quarantined ids."""
        newly = []
        for cid, s in scores.items():
            rec = self.registry.get(cid)
            if rec is None:
                continue
            rec.reputation = self.beta * rec.reputation + (1 - self.beta) * float(s)
            if rec.reputation < self.tau and cid not in self._in_quarantine:
                rec.quarantined_until = rnd + self.quarantine_rounds
                self._in_quarantine.add(cid)
                newly.append(cid)
                self.events.append((rnd, cid, "quarantined"))
        return newly

    @property
    def quarantined(self) -> list:
        return sorted(self._in_quarantine)
