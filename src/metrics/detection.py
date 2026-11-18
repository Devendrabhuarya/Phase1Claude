"""Detection metrics (Architecture README 7.4): accuracy, macro-F1, MCC,
per-class recall, plus the label-flip attack success rate."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, recall_score


@torch.no_grad()
def predict(model: torch.nn.Module, X: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    """Predicted class indices for X (evaluation mode, no dropout)."""
    model.eval()
    Xt = torch.as_tensor(X, dtype=torch.float32)
    out = [model(Xt[i:i + batch_size]).argmax(dim=1) for i in range(0, len(Xt), batch_size)]
    return torch.cat(out).numpy()


def detection_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                      class_names: Sequence[str]) -> dict:
    """Accuracy, macro-F1, MCC and recall of every class (key ``recall_<class>``)."""
    labels = list(range(len(class_names)))
    rec = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    out.update({f"recall_{c}": float(r) for c, r in zip(class_names, rec)})
    return out


def attack_success_rate(y_true: np.ndarray, y_pred: np.ndarray, benign_id: int) -> float:
    """Fraction of attack samples classified as Benign (goal of label flipping)."""
    attack = y_true != benign_id
    return float((y_pred[attack] == benign_id).mean()) if attack.any() else 0.0


def majority_baseline(y_train: np.ndarray, y_test: np.ndarray, class_names: Sequence[str]) -> dict:
    """Metrics of always predicting the most frequent training class."""
    majority = np.bincount(y_train).argmax()
    return detection_metrics(y_test, np.full_like(y_test, majority), class_names)
