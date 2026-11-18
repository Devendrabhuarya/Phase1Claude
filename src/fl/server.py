"""Sequential federated-learning simulation (plain PyTorch, no Flower/Ray).

One round:

1. every participating client trains locally from the current global
   weights and returns its update Δᵢ (malicious clients may poison their
   data before training or their update after training — Stage 6);
2. the aggregator turns the stacked updates into one update;
3. the server adds it to the global weights and evaluates on the test set.

The server also offers the **root update** g₀ (training the current global
model on the server's small clean root set), computed only if an aggregator
asks for it (FLTrust, proposed method).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
import torch

from src.data.partition import Partition
from src.data.preprocess import DatasetBundle
from src.fl.client import client_update, steps_per_round
from src.metrics.detection import attack_success_rate, detection_metrics, predict
from src.models.mlp import build_model, final_layer_slice, get_weights, set_weights
from src.utils.seed import make_rng


@dataclass
class FLData:
    """Per-client training data plus the server-side test and root sets."""

    client_X: list
    client_y: list
    client_groups: list           # zone of each client (evaluation only)
    X_test: np.ndarray
    y_test: np.ndarray
    X_root: np.ndarray
    y_root: np.ndarray
    class_names: list

    @property
    def n_clients(self) -> int:
        return len(self.client_X)

    @property
    def n_features(self) -> int:
        return self.X_test.shape[1]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def benign_id(self) -> int:
        return self.class_names.index("Benign")


def make_fl_data(bundle: DatasetBundle, partition: Partition) -> FLData:
    """Cut the client pool into per-client local training sets."""
    return FLData(
        client_X=[bundle.X_pool[i] for i in partition.train_indices],
        client_y=[bundle.y_pool[i] for i in partition.train_indices],
        client_groups=list(partition.client_groups),
        X_test=bundle.X_test, y_test=bundle.y_test,
        X_root=bundle.X_root, y_root=bundle.y_root,
        class_names=list(bundle.class_names),
    )


def _client_seed(seed: int, rnd: int, cid: int) -> int:
    return int(make_rng(seed, 4, rnd, cid).integers(2**31 - 1))


def run_federated(
    data: FLData,
    cfg: dict,
    aggregator,
    attack=None,
    rounds: Optional[int] = None,
    round_hook: Optional[Callable[[int, dict], None]] = None,
    verbose: bool = True,
    admission=None,
) -> dict:
    """Run the FL simulation and return ``{"history": [...], "final": {...}, "weights": tensor}``.

    Args:
        data: per-client data and server test/root sets.
        cfg: full config.
        aggregator: object with ``aggregate(updates, global_weights, context)``.
        attack: optional attack object (Stage 6) with ``malicious_ids``,
            ``poison_data(cid, X, y)`` and ``poison_updates(updates, malicious_mask, rng)``.
        rounds: override ``cfg['fl']['rounds']``.
        round_hook: called as ``round_hook(round, row)`` after every round.
        admission: optional ABAC admission controller (Stage 8) with
            ``admit(round, client_ids)`` and ``after_round(round, info)``.
            Only admitted clients train; intruders that get through send noise.
    """
    seed = cfg["seed"]
    torch.manual_seed(seed)
    model = build_model(data.n_features, data.n_classes, cfg["model"])
    global_w = get_weights(model)
    rounds = cfg["fl"]["rounds"] if rounds is None else rounds
    t_cfg = cfg["training"]
    client_ids = list(range(data.n_clients))
    malicious = set(attack.malicious_ids) if attack is not None else set()

    # Data poisoning happens once, on the malicious clients' local data.
    client_X, client_y = list(data.client_X), list(data.client_y)
    if attack is not None:
        for cid in malicious:
            client_X[cid], client_y[cid] = attack.poison_data(cid, client_X[cid], client_y[cid])
    n_samples = torch.tensor([len(y) for y in client_y], dtype=torch.float32)
    f_cfg = cfg["fltrust"]
    if f_cfg["root_steps"] == "match_clients":
        root_steps = int(round(np.mean([steps_per_round(len(y), t_cfg) for y in client_y])))
    else:
        root_steps = int(f_cfg["root_steps"])

    history = []
    for rnd in range(1, rounds + 1):
        # --- ABAC admission (before training)
        t_abac = time.perf_counter()
        adm = admission.admit(rnd, client_ids) if admission is not None else None
        active = adm.admitted if adm is not None else client_ids
        t_abac = time.perf_counter() - t_abac

        t0 = time.perf_counter()
        updates = [client_update(model, global_w, client_X[c], client_y[c], t_cfg, data.n_classes,
                                 _client_seed(seed, rnd, c)) for c in active]
        t_clients = time.perf_counter() - t0

        ids = list(active)
        if updates and attack is not None and malicious:
            updates = torch.stack(updates)
            mask = torch.tensor([c in malicious for c in ids])
            if mask.any() and not mask.all():     # adaptive attacks need some benign updates
                updates = attack.poison_updates(updates, mask, make_rng(seed, 6, rnd))
            updates = list(updates)
        if adm is not None:                       # intruders that slipped through ABAC
            for k, iid in enumerate(adm.intruders_admitted):
                noise = admission.intruder_update(rnd, k, len(global_w))
                updates.append(torch.as_tensor(noise, dtype=global_w.dtype))
                ids.append(iid)

        root_cache: dict = {}

        def root_update(_w=global_w, _rnd=rnd) -> torch.Tensor:
            if "g0" not in root_cache:
                root_cache["g0"] = client_update(
                    model, _w, data.X_root, data.y_root, t_cfg, data.n_classes,
                    _client_seed(seed, _rnd, 10_000), steps=root_steps,
                    batch_size=f_cfg["root_batch_size"])
            return root_cache["g0"]

        sizes = [float(n_samples[c]) if isinstance(c, int) else float(np.median(n_samples))
                 for c in ids]
        context = {"round": rnd, "client_ids": ids, "n_samples": torch.tensor(sizes),
                   "root_update": root_update, "final_layer": final_layer_slice(model)}
        t1 = time.perf_counter()
        if updates:
            new_update, info = aggregator.aggregate(torch.stack(updates), global_w, context)
        else:                                     # nobody admitted: keep the model
            new_update, info = torch.zeros_like(global_w), {"selected": []}
        global_w = global_w + new_update
        newly_quarantined = admission.after_round(rnd, info) if admission is not None else []
        t_server = time.perf_counter() - t1 + t_abac

        row = {"round": rnd, "time_clients_s": t_clients, "time_server_s": t_server}
        if adm is not None:
            row.update(abac_row(adm, newly_quarantined, admission))
        if "cohorts" in info:
            row["cohort_ari"] = cohort_ari(info["cohorts"], data.client_groups)
        if rnd % cfg["fl"]["eval_every"] == 0 or rnd == rounds:
            row.update(evaluate(model, global_w, data, cfg))
        selected = info.get("selected")
        if selected is not None and malicious:
            row.update(screening_rates(selected, client_ids, malicious))
        row["info"] = info
        history.append(row)
        if round_hook:
            round_hook(rnd, row)
        if verbose and "accuracy" in row:
            print(f"round {rnd:3d}  acc={row['accuracy']:.4f}  macroF1={row['macro_f1']:.4f}  "
                  f"mcc={row['mcc']:.4f}  ({t_clients:.1f}s clients, {t_server:.2f}s server)")

    final = {k: v for k, v in history[-1].items() if k != "info"}
    return {"history": history, "final": final, "weights": global_w}


def abac_row(adm, newly_quarantined, admission) -> dict:
    """Per-round ABAC / reputation statistics."""
    return {
        "n_admitted": len(adm.admitted),
        "denied_clients": sorted(set(range(admission.n_clients)) - set(adm.admitted)),
        "intruders": adm.n_intruders,
        "intruders_blocked": adm.intruders_blocked,
        "newly_quarantined": [int(c[-2:]) for c in newly_quarantined],
        "released": [int(c[-2:]) for c in adm.released],
        "abac_decisions": adm.decisions,
        "reputation": admission.reputations(),
    }


def cohort_ari(cohorts: dict, client_groups) -> float:
    """Adjusted Rand Index between discovered cohorts and true zones (FL clients only)."""
    from sklearn.metrics import adjusted_rand_score
    pairs = [(client_groups[c], k) for c, k in cohorts.items() if isinstance(c, int)]
    if len(pairs) < 2:
        return float("nan")
    true, found = zip(*pairs)
    return float(adjusted_rand_score(true, found))


def evaluate(model, weights: torch.Tensor, data: FLData, cfg: dict) -> dict:
    """Test-set metrics of the model with ``weights`` (+ label-flip attack success rate)."""
    set_weights(model, weights)
    y_pred = predict(model, data.X_test, cfg["fl"]["eval_batch_size"])
    m = detection_metrics(data.y_test, y_pred, data.class_names)
    m["attack_success_rate"] = attack_success_rate(data.y_test, y_pred, data.benign_id)
    return m


def screening_rates(selected: Sequence[int], client_ids: Sequence[int], malicious: set) -> dict:
    """Malicious-client detection TPR / FPR for one round (rejected = flagged)."""
    selected = set(selected)
    benign = [c for c in client_ids if c not in malicious]
    tpr = np.mean([c not in selected for c in malicious]) if malicious else 0.0
    fpr = np.mean([c not in selected for c in benign]) if benign else 0.0
    return {"screen_tpr": float(tpr), "screen_fpr": float(fpr)}
