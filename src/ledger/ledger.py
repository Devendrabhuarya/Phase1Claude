"""Hash-chained audit ledger (Architecture README 6.8).

One block per FL round. Each block stores the round's decisions (ABAC,
cohorts, screening, reputation) and *hashes* of the data (client updates,
new global model); raw traffic and full update vectors stay off-chain.

    block_hash = H(canonical JSON of every other field, incl. prev_hash)

Because every block contains the previous block's hash, changing any past
record changes its hash and breaks the link to the next block, so
:meth:`HashChainLedger.verify` detects it.

The FL code only uses the small :class:`Ledger` interface (``append``,
``verify``, ``query``) so the hash chain could later be swapped for
Hyperledger Fabric without changing the server loop. The ledger only
records decisions; it never decides whether an update is benign.
"""

from __future__ import annotations

import abc
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.ledger.merkle import DEFAULT_HASH, hash_bytes, merkle_root

GENESIS_PREV_HASH = "0" * 64

#: Fields covered by ``block_hash`` (README 6.8 table + ``update_hashes``).
BLOCK_FIELDS = ("index", "round", "timestamp", "prev_hash", "abac_decisions", "update_hashes",
                "update_merkle_root", "cohorts", "screening", "reputation", "global_model_hash")


def to_jsonable(obj):
    """Convert numpy / torch values and non-string keys to plain JSON types.

    Dict keys become strings, so a block hashes the same before and after a
    JSON save/load round trip.
    """
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        items = sorted(obj, key=str) if isinstance(obj, set) else obj
        return [to_jsonable(v) for v in items]
    if isinstance(obj, torch.Tensor):
        return to_jsonable(obj.detach().cpu().tolist())
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, float) and obj != obj:          # NaN is not valid JSON
        return None
    return obj


def canonical_json(obj) -> bytes:
    """Deterministic encoding: sorted keys, no whitespace, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def compute_block_hash(block: dict, algo: str = DEFAULT_HASH) -> str:
    """Hash of all fields in :data:`BLOCK_FIELDS`."""
    return hash_bytes(canonical_json({k: block[k] for k in BLOCK_FIELDS}), algo)


class Ledger(abc.ABC):
    """Interface the FL server talks to (replaceable by a real blockchain)."""

    @abc.abstractmethod
    def append(self, record: dict) -> dict:
        """Add one round's record; return the stored block."""

    @abc.abstractmethod
    def verify(self) -> dict:
        """Check the whole chain; return ``{"valid": bool, "errors": [...], ...}``."""

    @abc.abstractmethod
    def query(self, round: int | None = None, client: str | None = None) -> list:
        """Return blocks (or one client's entries) matching the filter."""


class HashChainLedger(Ledger):
    """In-memory hash chain, optionally written to a JSON-lines file.

    Args:
        algo: hashlib algorithm name (``ledger.hash`` in the config).
        path: optional ``.jsonl`` file; each appended block is written as one line.
    """

    def __init__(self, algo: str = DEFAULT_HASH, path: str | Path | None = None):
        self.algo = algo
        self.blocks: list = []
        self.path = Path(path) if path else None
        self.timings = {"append_s": [], "verify_s": []}
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    # ------------------------------------------------------------- write

    def append(self, record: dict) -> dict:
        """Create the next block from ``record`` and link it to the chain.

        ``record`` holds ``round``, ``abac_decisions``, ``update_hashes``
        (client -> hash), ``cohorts``, ``screening``, ``reputation`` and
        ``global_model_hash``; ``index``, ``timestamp``, ``prev_hash``, the
        Merkle root and ``block_hash`` are filled in here.
        """
        t0 = time.perf_counter()
        update_hashes = to_jsonable(record.get("update_hashes", {}))
        block = {
            "index": len(self.blocks),
            "round": int(record["round"]),
            "timestamp": time.time(),
            "prev_hash": self.blocks[-1]["block_hash"] if self.blocks else GENESIS_PREV_HASH,
            "abac_decisions": to_jsonable(record.get("abac_decisions", {})),
            "update_hashes": update_hashes,
            "update_merkle_root": merkle_root(list(update_hashes.values()), self.algo),
            "cohorts": to_jsonable(record.get("cohorts", {})),
            "screening": to_jsonable(record.get("screening", {})),
            "reputation": to_jsonable(record.get("reputation", {})),
            "global_model_hash": record["global_model_hash"],
        }
        block["block_hash"] = compute_block_hash(block, self.algo)
        self.blocks.append(block)
        if self.path:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(canonical_json(block).decode("utf-8") + "\n")
        self.timings["append_s"].append(time.perf_counter() - t0)
        return copy.deepcopy(block)

    # ------------------------------------------------------------- check

    def verify(self) -> dict:
        """Recompute every block hash, Merkle root and link.

        Returns ``{"valid", "n_blocks", "first_invalid", "errors"}`` where
        ``errors`` lists every problem found (``first_invalid`` is the index
        of the earliest bad block or ``None``).
        """
        t0 = time.perf_counter()
        errors = []
        prev = GENESIS_PREV_HASH
        for i, b in enumerate(self.blocks):
            missing = [k for k in BLOCK_FIELDS + ("block_hash",) if k not in b]
            if missing:
                errors.append((i, f"missing fields {missing}"))
                prev = b.get("block_hash", "")
                continue
            if b["index"] != i:
                errors.append((i, f"index {b['index']} != position {i}"))
            if b["prev_hash"] != prev:
                errors.append((i, "prev_hash does not match previous block"))
            if merkle_root(list(b["update_hashes"].values()), self.algo) != b["update_merkle_root"]:
                errors.append((i, "update_merkle_root does not match update_hashes"))
            if compute_block_hash(b, self.algo) != b["block_hash"]:
                errors.append((i, "block_hash does not match block contents"))
            prev = b["block_hash"]
        self.timings["verify_s"].append(time.perf_counter() - t0)
        return {"valid": not errors, "n_blocks": len(self.blocks),
                "first_invalid": errors[0][0] if errors else None, "errors": errors}

    # ------------------------------------------------------------- read

    def query(self, round: int | None = None, client: str | None = None) -> list:
        """Blocks of one round (or all), or one client's history.

        With ``client`` (e.g. ``"client03"``) each result is that client's
        entry per round: ABAC decision, update hash, cohort, reputation and
        whether screening selected it.
        """
        blocks = [b for b in self.blocks if round is None or b["round"] == round]
        if client is None:
            return copy.deepcopy(blocks)
        out = []
        for b in blocks:
            s = b["screening"]
            out.append({
                "round": b["round"],
                "abac": b["abac_decisions"].get(client),
                "update_hash": b["update_hashes"].get(client),
                "cohort": b["cohorts"].get(client),
                "selected": client in s.get("selected", []),
                "trust_score": s.get("trust_scores", {}).get(client),
                "reputation": b["reputation"].get("scores", {}).get(client),
                "quarantined": client in b["reputation"].get("quarantined", []),
            })
        return out

    def head(self) -> str:
        """Hash of the last block (the value an external anchor would store)."""
        return self.blocks[-1]["block_hash"] if self.blocks else GENESIS_PREV_HASH

    def __len__(self) -> int:
        return len(self.blocks)

    # ------------------------------------------------------------- files

    @classmethod
    def load(cls, path: str | Path, algo: str = DEFAULT_HASH) -> "HashChainLedger":
        """Read a JSON-lines ledger file (no verification; call :meth:`verify`)."""
        led = cls(algo=algo)
        with Path(path).open(encoding="utf-8") as fh:
            led.blocks = [json.loads(line) for line in fh if line.strip()]
        return led

    def save(self, path: str | Path) -> None:
        """Write all blocks to a JSON-lines file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for b in self.blocks:
                fh.write(canonical_json(b).decode("utf-8") + "\n")


def build_ledger(cfg: dict, path: str | Path | None = None) -> HashChainLedger | None:
    """Ledger from the ``ledger`` config section, or ``None`` if disabled."""
    lc = cfg.get("ledger", {})
    if not lc.get("enabled", False):
        return None
    return HashChainLedger(algo=lc.get("hash", DEFAULT_HASH), path=path)
