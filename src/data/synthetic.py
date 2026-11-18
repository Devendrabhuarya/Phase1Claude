"""Small synthetic CICIoT2023-like data for tests and local smoke runs.

The CSV files use the real 39 CICIoT2023 column names and a set of realistic
raw labels with mixed spellings. Each class has a different feature mean so
that a model can actually learn something (needed by later stages' tests).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import numpy as np
import pandas as pd

CICIOT_FEATURES = [
    "Header_Length", "Protocol Type", "Time_To_Live", "Rate",
    "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
    "ack_flag_number", "ece_flag_number", "cwr_flag_number",
    "ack_count", "syn_count", "fin_count", "rst_count",
    "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC", "TCP", "UDP",
    "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC",
    "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number", "Variance",
]

#: raw label -> expected 8-class label (mixed spellings on purpose)
SYNTHETIC_LABELS = {
    "BenignTraffic": "Benign",
    "DDoS-ICMP_Flood": "DDoS",
    "DDOS_SYN_FLOOD": "DDoS",
    "DoS-UDP_Flood": "DoS",
    "dos tcp flood": "DoS",
    "Mirai-greeth_flood": "Mirai",
    "Recon-PortScan": "Recon",
    "VulnerabilityScan": "Recon",
    "DNS_Spoofing": "Spoofing",
    "MITM-ArpSpoofing": "Spoofing",
    "DictionaryBruteForce": "BruteForce",
    "SqlInjection": "Web",
    "XSS": "Web",
    "CommandInjection": "Web",
    "Backdoor_Malware": "Web",
    "Uploading_Attack": "Web",
    "BrowserHijacking": "Web",
}

CLASS_ORDER = ["Benign", "DDoS", "DoS", "Recon", "Spoofing", "Mirai", "Web", "BruteForce"]


def make_class_features(cls_id: int, n: int, rng: np.random.Generator,
                        n_features: int = len(CICIOT_FEATURES), separation: float = 2.0,
                        noise: float = 1.0) -> np.ndarray:
    """Features for ``n`` rows of class ``cls_id``: mean ~ separation*cls_id + one strong feature."""
    X = rng.normal(loc=separation * cls_id, scale=noise, size=(n, n_features))
    X[:, cls_id % n_features] += 2.5 * separation
    return X


def make_synthetic_frame(rows_per_label: Mapping[str, int], seed: int = 0,
                         n_features: int = len(CICIOT_FEATURES)) -> pd.DataFrame:
    """DataFrame with feature columns + ``Label`` for the given raw-label row counts."""
    rng = np.random.default_rng(seed)
    parts = []
    for raw, n in rows_per_label.items():
        cls_id = CLASS_ORDER.index(SYNTHETIC_LABELS.get(raw, "Benign"))
        df = pd.DataFrame(make_class_features(cls_id, n, rng, n_features),
                          columns=CICIOT_FEATURES[:n_features])
        df["Label"] = raw
        parts.append(df)
    return pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=seed)


def write_synthetic_csvs(out_dir: Path | str, n_files: int = 3, rows_per_label: int = 200,
                         rare_rows: int = 20, seed: int = 0,
                         extra_labels: Optional[Mapping[str, int]] = None) -> list:
    """Write ``Merged01.csv``... with all synthetic labels; Web/BruteForce labels are rare.

    Also injects a few inf/NaN values and exact duplicate rows to exercise cleaning.
    Returns the list of written paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rare = {"Web", "BruteForce"}
    paths = []
    for i in range(n_files):
        counts = {lab: (rare_rows if cls in rare else rows_per_label)
                  for lab, cls in SYNTHETIC_LABELS.items()}
        if extra_labels:
            counts.update(extra_labels)
        df = make_synthetic_frame(counts, seed=seed + i)
        df.iloc[0, 3] = np.inf
        df.iloc[1, 4] = np.nan
        df = pd.concat([df, df.iloc[:5]], ignore_index=True)  # 5 exact duplicates
        p = out_dir / f"Merged{i + 1:02d}.csv"
        df.to_csv(p, index=False)
        paths.append(p)
    return paths


def make_synthetic_sample(class_counts: Mapping[str, int], seed: int = 0,
                          separation: float = 0.6, noise: float = 1.0):
    """In-memory :class:`SampledData` (no CSV) with the 8 CICIoT2023 classes.

    The default ``separation`` makes classes overlap, so the task is learnable
    but not trivial (a poisoned model visibly loses accuracy).
    """
    from src.data.ciciot2023 import SampledData  # local import avoids a cycle

    rng = np.random.default_rng(seed)
    X_parts, y_parts = [], []
    for c, n in class_counts.items():
        cid = CLASS_ORDER.index(c)
        X_parts.append(make_class_features(cid, n, rng, separation=separation, noise=noise))
        y_parts.append(np.full(n, cid, dtype=np.int64))
    seen = pd.Series({c: class_counts.get(c, 0) for c in CLASS_ORDER}, name="seen")
    return SampledData(X=np.concatenate(X_parts).astype(np.float32), y=np.concatenate(y_parts),
                       feature_names=list(CICIOT_FEATURES), class_names=list(CLASS_ORDER),
                       raw_label_counts=seen.rename("count"), class_rows_seen=seen)


#: default class sizes for synthetic FL runs (ratio similar to the real capped sample)
SYNTHETIC_FL_COUNTS = {"Benign": 4000, "DDoS": 1000, "DoS": 1000, "Recon": 1000,
                       "Spoofing": 1000, "Mirai": 1000, "Web": 500, "BruteForce": 300}
