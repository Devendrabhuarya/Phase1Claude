"""Hashing helpers and a binary Merkle tree (Architecture README 6.8).

A Merkle root commits to a list of leaf hashes with one hash: changing,
adding, removing or reordering any leaf changes the root. A Merkle proof
(the sibling hashes on the path from a leaf to the root) lets anyone check
that one client's update hash is part of a round's root without seeing
the other leaves.

Tree rule used here (as in Bitcoin): leaves are hashed pairwise level by
level; if a level has an odd number of nodes the last one is paired with
itself. The root of an empty list is the hash of the empty string.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch

DEFAULT_HASH = "sha256"


def hash_bytes(data: bytes, algo: str = DEFAULT_HASH) -> str:
    """Hex digest of ``data`` with the hashlib algorithm ``algo``."""
    return hashlib.new(algo, data).hexdigest()


def hash_tensor(t, algo: str = DEFAULT_HASH) -> str:
    """Hash of a weight / update vector.

    The vector is converted to little-endian float32 bytes first, so the
    same values always give the same hash regardless of device or dtype.
    """
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu().numpy()
    arr = np.ascontiguousarray(np.asarray(t), dtype="<f4")
    return hash_bytes(arr.tobytes(), algo)


def _parent(left: str, right: str, algo: str) -> str:
    return hash_bytes(bytes.fromhex(left) + bytes.fromhex(right), algo)


def merkle_root(leaves: list, algo: str = DEFAULT_HASH) -> str:
    """Merkle root of a list of hex leaf hashes."""
    if not leaves:
        return hash_bytes(b"", algo)
    level = list(leaves)
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        level = [_parent(level[i], level[i + 1], algo) for i in range(0, len(level), 2)]
    return level[0]


def merkle_proof(leaves: list, index: int, algo: str = DEFAULT_HASH) -> list:
    """Sibling hashes from leaf ``index`` up to the root.

    Each step is ``(sibling_hash, sibling_is_right)``.
    """
    if not 0 <= index < len(leaves):
        raise IndexError(f"leaf index {index} out of range for {len(leaves)} leaves")
    proof, level, i = [], list(leaves), index
    while len(level) > 1:
        if len(level) % 2:
            level.append(level[-1])
        sib = i + 1 if i % 2 == 0 else i - 1
        proof.append((level[sib], i % 2 == 0))
        level = [_parent(level[j], level[j + 1], algo) for j in range(0, len(level), 2)]
        i //= 2
    return proof


def verify_merkle_proof(leaf: str, proof: list, root: str, algo: str = DEFAULT_HASH) -> bool:
    """True if ``leaf`` with ``proof`` hashes up to ``root``."""
    h = leaf
    for sibling, sibling_is_right in proof:
        h = _parent(h, sibling, algo) if sibling_is_right else _parent(sibling, h, algo)
    return h == root
