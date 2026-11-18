"""CICIoT2023 label inspection and keyword-based mapping to 8 classes.

The raw dataset has ~34 attack labels whose exact spelling we do not assume
(e.g. ``DDoS-ICMP_Flood``, ``Backdoor_Malware``, ``BenignTraffic``). Each raw
label is normalised (upper-case, ``-``, ``_`` and spaces removed) and the
keyword list from the config is checked **in order**; the first keyword that
appears inside the normalised label decides the class. Order matters:
``DDOS`` must be checked before ``DOS`` because "DDOS" contains "DOS".

Any label that matches no keyword raises an error — data is never silently
dropped or mislabelled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pandas as pd


class UnmappedLabelError(ValueError):
    """Raised when one or more raw labels match no keyword."""


def normalize_label(label: object) -> str:
    """Upper-case a raw label and remove '-', '_' and spaces."""
    text = str(label).upper()
    for ch in ("-", "_", " "):
        text = text.replace(ch, "")
    return text.strip()


def map_label(label: object, keywords: Sequence[Sequence[str]]) -> str | None:
    """Return the class for one raw label, or ``None`` if no keyword matches.

    Args:
        label: raw label text.
        keywords: ordered list of ``[KEYWORD, ClassName]`` pairs.
    """
    norm = normalize_label(label)
    for keyword, class_name in keywords:
        if keyword in norm:
            return class_name
    return None


def build_label_map(raw_labels: Iterable[object],
                    keywords: Sequence[Sequence[str]]) -> dict:
    """Map every distinct raw label to its class; raise if any is unmapped.

    Returns:
        dict raw_label -> class name.

    Raises:
        UnmappedLabelError: listing every raw label that matched no keyword.
    """
    mapping, unmapped = {}, []
    for raw in sorted(set(raw_labels), key=str):
        cls = map_label(raw, keywords)
        if cls is None:
            unmapped.append(raw)
        else:
            mapping[raw] = cls
    if unmapped:
        raise UnmappedLabelError(
            f"{len(unmapped)} label(s) match no keyword in the config "
            f"(data.label_keywords): {unmapped}")
    return mapping


def map_labels(labels: pd.Series, keywords: Sequence[Sequence[str]]) -> pd.Series:
    """Vectorised mapping of a label column (only distinct values are checked)."""
    mapping = build_label_map(labels.unique(), keywords)
    return labels.map(mapping)


def inspect_labels(files: Sequence[Path | str], label_column: str = "Label",
                   chunk_size: int = 100_000, verbose: bool = True) -> pd.Series:
    """Count every raw label across CSV files, reading only the label column.

    Returns:
        Series raw_label -> count, sorted by count (descending).
    """
    counts: dict = {}
    for f in files:
        for chunk in pd.read_csv(f, usecols=[label_column], chunksize=chunk_size):
            for lab, n in chunk[label_column].value_counts(dropna=False).items():
                counts[lab] = counts.get(lab, 0) + int(n)
    result = pd.Series(counts, name="count", dtype="int64").sort_values(ascending=False)
    if verbose:
        print(f"{len(result)} unique raw labels in {len(files)} file(s):")
        print(result.to_string())
    return result


def summarize_mapping(raw_counts: Mapping[object, int],
                      keywords: Sequence[Sequence[str]]) -> pd.DataFrame:
    """Table raw_label | class | count, to check the mapping by eye."""
    mapping = build_label_map(raw_counts.keys(), keywords)
    rows = [{"raw_label": k, "class": mapping[k], "count": int(v)} for k, v in raw_counts.items()]
    return pd.DataFrame(rows).sort_values(["class", "count"], ascending=[True, False],
                                          ignore_index=True)
