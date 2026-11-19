"""ABAC admission policy with PyCasbin (Architecture README 6.2).

Policy: **permit** if

    registered ∧ firmware ≥ min ∧ fresh ∧ reputation ≥ τ ∧ not quarantined

Casbin evaluates the matcher below on two attribute objects:

* ``r.sub`` – the client (registered, firmware, last_seen, reputation, quarantined_until)
* ``r.obj`` – the resource "fl_aggregation" in the current round
  (round, min_firmware, freshness, tau)

Casbin only answers permit/deny, so :func:`deny_reasons` re-checks each
condition separately to explain a denial for the audit ledger. A unit test
checks that both always agree.
"""

from __future__ import annotations

from types import SimpleNamespace

import casbin

from src.abac.registry import ClientAttributes, firmware_to_int

MODEL_TEXT = """
[request_definition]
r = sub, obj, act

[policy_definition]
p = sub, obj, act

[policy_effect]
e = some(where (p.eft == allow))

[matchers]
m = r.sub.role == p.sub && r.obj.name == p.obj && r.act == p.act \
 && r.sub.registered == True \
 && r.sub.firmware >= r.obj.min_firmware \
 && r.obj.round - r.sub.last_seen <= r.obj.freshness \
 && r.sub.reputation >= r.obj.tau \
 && r.sub.quarantined_until < r.obj.round
"""

RESOURCE = "fl_aggregation"
ACTION = "submit_update"


class AbacPolicy:
    """PyCasbin enforcer for the admission rule."""

    def __init__(self, min_firmware_version: str, freshness_window: int, tau: float):
        model = casbin.model.Model()
        model.load_model_from_text(MODEL_TEXT)
        self.enforcer = casbin.Enforcer(model)
        self.enforcer.add_policy("client", RESOURCE, ACTION)
        self.min_firmware = firmware_to_int(min_firmware_version)
        self.freshness = int(freshness_window)
        self.tau = float(tau)

    def _objects(self, attrs: ClientAttributes, rnd: int):
        sub = SimpleNamespace(role="client", registered=bool(attrs.registered),
                              firmware=firmware_to_int(attrs.firmware_version),
                              last_seen=int(attrs.last_seen), reputation=float(attrs.reputation),
                              quarantined_until=int(attrs.quarantined_until))
        obj = SimpleNamespace(name=RESOURCE, round=int(rnd), min_firmware=self.min_firmware,
                              freshness=self.freshness, tau=self.tau)
        return sub, obj

    def permit(self, attrs: ClientAttributes, rnd: int) -> bool:
        """Casbin decision for one client in round ``rnd``."""
        sub, obj = self._objects(attrs, rnd)
        return bool(self.enforcer.enforce(sub, obj, ACTION))

    def deny_reasons(self, attrs: ClientAttributes, rnd: int) -> list:
        """Human-readable list of failed conditions (empty list = permit)."""
        reasons = []
        if not attrs.registered:
            reasons.append("unregistered")
        if firmware_to_int(attrs.firmware_version) < self.min_firmware:
            reasons.append(f"firmware {attrs.firmware_version} below minimum")
        if rnd - attrs.last_seen > self.freshness:
            reasons.append(f"stale (last seen round {attrs.last_seen})")
        if attrs.reputation < self.tau:
            reasons.append(f"reputation {attrs.reputation:.3f} < {self.tau}")
        if attrs.quarantined_until >= rnd:
            reasons.append(f"quarantined until round {attrs.quarantined_until}")
        return reasons

    def decide(self, attrs: ClientAttributes, rnd: int) -> dict:
        """``{"permit": bool, "reason": str}`` for the ledger."""
        ok = self.permit(attrs, rnd)
        reasons = self.deny_reasons(attrs, rnd)
        return {"permit": ok, "reason": "ok" if ok else "; ".join(reasons) or "policy"}
