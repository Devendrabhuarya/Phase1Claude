"""Client attribute registry (Architecture README 6.2).

The server keeps one record of attributes per known client identity. The
ABAC policy (``policy.py``) decides admission only from these attributes.

| Attribute           | Meaning                                                  |
|---------------------|----------------------------------------------------------|
| registered          | identity enrolled by the operator                        |
| device_type         | camera, thermostat, gateway ...                          |
| firmware_version    | "major.minor.patch"; must be >= the minimum version      |
| last_seen           | round of the last authenticated check-in (freshness)     |
| reputation          | in [0, 1], updated by the reputation module (6.7)        |
| quarantined_until   | client is quarantined while round <= quarantined_until   |
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Optional


def firmware_to_int(version: str) -> int:
    """'2.1.3' -> 20103, so versions can be compared as numbers inside the policy."""
    parts = [int(p) for p in str(version).split(".")] + [0, 0, 0]
    major, minor, patch = parts[:3]
    return major * 10_000 + minor * 100 + patch


@dataclass
class ClientAttributes:
    client_id: str
    registered: bool = True
    device_type: str = "gateway"
    firmware_version: str = "2.0.0"
    last_seen: int = 0
    reputation: float = 0.5
    quarantined_until: int = 0


class AttributeRegistry:
    """In-memory registry: client id -> :class:`ClientAttributes`."""

    def __init__(self):
        self._records: Dict[str, ClientAttributes] = {}

    def register(self, client_id: str, **attrs) -> ClientAttributes:
        rec = ClientAttributes(client_id=client_id, **attrs)
        self._records[client_id] = rec
        return rec

    def get(self, client_id: str) -> Optional[ClientAttributes]:
        return self._records.get(client_id)

    def lookup(self, client_id: str) -> ClientAttributes:
        """Record for ``client_id``; an unknown identity gets an *unregistered* record."""
        rec = self.get(client_id)
        return rec if rec is not None else ClientAttributes(client_id=client_id, registered=False,
                                                            firmware_version="0.0.0",
                                                            last_seen=-10**9, reputation=0.0)

    def touch(self, client_id: str, rnd: int) -> None:
        """Record an authenticated check-in in round ``rnd`` (updates freshness)."""
        if client_id in self._records:
            self._records[client_id].last_seen = rnd

    def ids(self) -> list:
        return list(self._records)

    def snapshot(self) -> dict:
        """Plain-dict copy of all records (for logging)."""
        return {cid: asdict(rec) for cid, rec in self._records.items()}

    def __contains__(self, client_id: str) -> bool:
        return client_id in self._records
