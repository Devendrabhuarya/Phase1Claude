"""Turn one FL round's state into a ledger record (README 6.8 block fields).

Client ids are written as names (``client03``; intruders keep their own
ids) so the ledger is readable on its own.
"""

from __future__ import annotations

from src.abac.admission import client_name
from src.ledger.merkle import DEFAULT_HASH, hash_tensor

#: Keys of the aggregator ``info`` dict copied into the ``screening`` field.
SCREENING_KEYS = ("selected", "multikrum_kept", "trust_scores", "dropped_cohorts",
                  "cohort_weights", "clipped", "clip_median_norm", "round_scores")


def _name(cid) -> str:
    return client_name(cid) if isinstance(cid, int) else str(cid)


def _rename(obj):
    """Replace int client ids by names in the values / keys used by ``info``."""
    if isinstance(obj, dict):
        return {(_name(k) if isinstance(k, int) else k): obj[k] for k in obj}
    return obj


def screening_record(info: dict) -> dict:
    """Screening part of a block: selections, Multi-Krum kept, trust, dropped cohorts."""
    out = {}
    for key in SCREENING_KEYS:
        if key not in info:
            continue
        v = info[key]
        if key in ("selected", "clipped"):
            v = [_name(c) for c in v]
        elif key == "multikrum_kept":
            v = {str(k): [_name(c) for c in members] for k, members in v.items()}
        elif isinstance(v, dict):
            v = _rename(v)
        out[key] = v
    return out


def round_record(rnd: int, ids: list, updates, info: dict, global_weights,
                 adm=None, admission=None, algo: str = DEFAULT_HASH) -> dict:
    """Ledger record for round ``rnd``.

    Args:
        ids: client ids whose updates reached the aggregator (ints, or
            intruder id strings).
        updates: matching update vectors (only their hashes are stored).
        info: the aggregator's ``info`` dict.
        global_weights: the new global weights after this round.
        adm / admission: ABAC result and controller (``None`` for baselines).
    """
    rec = {
        "round": rnd,
        "abac_decisions": dict(adm.decisions) if adm is not None else {},
        "update_hashes": {_name(c): hash_tensor(u, algo) for c, u in zip(ids, updates)},
        "cohorts": _rename(info.get("cohorts", {})),
        "screening": screening_record(info),
        "global_model_hash": hash_tensor(global_weights, algo),
    }
    if admission is not None:
        rec["reputation"] = {
            "scores": {client_name(c): r for c, r in admission.reputations().items()},
            "quarantined": list(admission.reputation.quarantined),
        }
    else:
        rec["reputation"] = {}
    return rec
