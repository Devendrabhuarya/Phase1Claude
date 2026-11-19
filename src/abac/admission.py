"""Server-side admission controller: ABAC + unauthorized-join injection + reputation feedback.

Used by the FL server (``run_federated(..., admission=...)``):

* :meth:`AdmissionController.admit` – start of a round: release finished
  quarantines, record check-ins of the registered clients, inject the
  unauthorized join attempts, and ask the ABAC policy about everyone.
  Only admitted clients train and send updates.
* :meth:`AdmissionController.after_round` – end of a round: feed the
  screening scores back into the reputations (closed loop), if enabled.

Unauthorized identities injected every round (``abac.n_unauthorized_clients``),
cycling through the types:

* ``unregistered``   – unknown identity (spoofed / Sybil)
* ``stale_firmware`` – registered device with firmware below the minimum
* ``expired``        – registered identity that has not checked in for a long time
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.abac.policy import AbacPolicy
from src.abac.registry import AttributeRegistry
from src.reputation.reputation import ReputationManager


def client_name(cid: int) -> str:
    """Registry identity of FL client ``cid`` (e.g. 3 -> 'client03')."""
    return f"client{cid:02d}"


@dataclass
class AdmissionResult:
    admitted: list                                  # admitted FL client ids (ints)
    intruders_admitted: list                        # injected identities that got through
    decisions: dict = field(default_factory=dict)   # identity -> {"permit", "reason"}
    n_intruders: int = 0
    released: list = field(default_factory=list)

    @property
    def intruders_blocked(self) -> int:
        return self.n_intruders - len(self.intruders_admitted)


class AdmissionController:
    """Holds the attribute registry, the PyCasbin policy and the reputation manager."""

    def __init__(self, cfg: dict, n_clients: int, use_reputation: bool = True):
        a, r = cfg["abac"], cfg["reputation"]
        self.cfg = cfg
        self.registry = AttributeRegistry()
        self.policy = AbacPolicy(a["min_firmware_version"], a["freshness_window"],
                                 a["reputation_threshold"])
        frac = r.get("max_quarantine_fraction")
        self.reputation = ReputationManager(
            self.registry, r["beta"], r["threshold"], r["quarantine_rounds"],
            r["probation_value"],
            max_quarantined=None if frac is None else int(frac * n_clients))
        self.use_reputation = use_reputation
        fw, dev = a["registered_firmware_versions"], a["device_types"]
        for cid in range(n_clients):
            self.registry.register(client_name(cid), device_type=dev[cid % len(dev)],
                                   firmware_version=fw[cid % len(fw)], last_seen=0,
                                   reputation=r["initial"], quarantined_until=0)
        self.n_clients = n_clients
        self.inject = bool(a["inject_unauthorized"])
        self.n_unauth = int(a["n_unauthorized_clients"]) if self.inject else 0
        self.unauth_types = list(a["unauthorized_types"])
        self.totals = {"intruders": 0, "intruders_blocked": 0}

    # -------------------------------------------------------------- intruders

    def _inject(self, rnd: int) -> list:
        """Create this round's unauthorized identities in the registry; return their ids."""
        a = self.cfg["abac"]
        ids = []
        for k in range(self.n_unauth):
            kind = self.unauth_types[k % len(self.unauth_types)]
            iid = f"intruder-r{rnd}-{k}-{kind}"
            if kind == "stale_firmware":
                self.registry.register(iid, firmware_version=a["stale_firmware_version"],
                                       last_seen=rnd, reputation=self.cfg["reputation"]["initial"])
            elif kind == "expired":
                self.registry.register(iid, firmware_version=a["registered_firmware_versions"][0],
                                       last_seen=rnd - a["freshness_window"] - 10,
                                       reputation=self.cfg["reputation"]["initial"])
            elif kind != "unregistered":
                raise ValueError(f"Unknown unauthorized type '{kind}'")
            ids.append(iid)                         # 'unregistered' stays unknown to the registry
        return ids

    # -------------------------------------------------------------- round hooks

    def admit(self, rnd: int, client_ids) -> AdmissionResult:
        released = self.reputation.release_expired(rnd)
        for cid in client_ids:                      # authenticated heartbeats
            self.registry.touch(client_name(cid), rnd)
        decisions, admitted = {}, []
        for cid in client_ids:
            d = self.policy.decide(self.registry.lookup(client_name(cid)), rnd)
            decisions[client_name(cid)] = d
            if d["permit"]:
                admitted.append(cid)
        intruders = self._inject(rnd)
        got_in = []
        for iid in intruders:
            d = self.policy.decide(self.registry.lookup(iid), rnd)
            decisions[iid] = d
            if d["permit"]:
                got_in.append(iid)
        self.totals["intruders"] += len(intruders)
        self.totals["intruders_blocked"] += len(intruders) - len(got_in)
        return AdmissionResult(admitted, got_in, decisions, len(intruders), released)

    def intruder_update(self, rnd: int, k: int, n_params: int):
        """Noise update an admitted intruder would send (only used if ABAC fails)."""
        rng = np.random.default_rng([self.cfg["seed"], 8, rnd, k])
        return rng.normal(0.0, self.cfg["abac"]["unauthorized_update_std"], size=n_params)

    def after_round(self, rnd: int, info: dict) -> list:
        """Reputation feedback from the aggregator's ``round_scores`` (client id -> sᵢ)."""
        scores = info.get("round_scores")
        if not self.use_reputation or not scores:
            return []
        return self.reputation.update(rnd, {client_name(c): s for c, s in scores.items()
                                            if isinstance(c, (int, np.integer))})

    def block_rate(self) -> float:
        t = self.totals
        return t["intruders_blocked"] / t["intruders"] if t["intruders"] else float("nan")

    def reputations(self) -> dict:
        return {cid: round(self.registry.get(client_name(cid)).reputation, 4)
                for cid in range(self.n_clients)}
