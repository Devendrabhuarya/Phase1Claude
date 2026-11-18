"""Data- and model-poisoning attacks and malicious-client placement (Stage 6)."""

from src.attacks.attacks import (ALIE, Attack, GaussianNoise, LabelFlip, MinMax, Scaling,  # noqa: F401
                                 SignFlip, get_attack)
from src.attacks.placement import select_malicious  # noqa: F401


def attack_from_config(cfg: dict, client_groups, benign_id: int):
    """Choose malicious clients (fraction + placement from config) and build the attack."""
    a = cfg["attacks"]
    mal = select_malicious(client_groups, a["malicious_fraction"], a["placement"], cfg["seed"],
                           a["concentrated_zone"])
    return get_attack(cfg, mal, benign_id), mal
