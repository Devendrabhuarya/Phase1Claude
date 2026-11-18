"""Stage 0 tests: configs load with the README values, seeding is reproducible,
result helpers work, and every package in the skeleton is importable."""

import importlib
import json
import random

import numpy as np
import pytest

from src.utils.config import CONFIG_DIR, deep_merge, load_config, save_config, load_yaml
from src.utils.io import load_json, run_is_done, save_json, save_rows_csv
from src.utils.seed import make_rng, set_seed

PACKAGES = ["src.data", "src.models", "src.fl", "src.attacks", "src.aggregators",
            "src.abac", "src.reputation", "src.ledger", "src.metrics", "src.utils"]


@pytest.mark.parametrize("pkg", PACKAGES)
def test_packages_importable(pkg):
    importlib.import_module(pkg)


def test_deep_merge_nested_and_does_not_mutate():
    base = {"a": 1, "b": {"x": 1, "y": 2}, "l": [1, 2]}
    out = deep_merge(base, {"b": {"y": 3}, "l": [9]})
    assert out == {"a": 1, "b": {"x": 1, "y": 3}, "l": [9]}
    assert base["b"]["y"] == 2  # input unchanged


@pytest.mark.parametrize("dataset", ["ciciot2023"])
def test_load_config(dataset):
    cfg = load_config(dataset)
    assert cfg["dataset"] == dataset
    assert cfg["paths"]["data_dir"]  # dataset-specific path merged in
    assert cfg["paths"]["results_dir"]  # base path kept


def test_unknown_dataset_raises():
    with pytest.raises(ValueError):
        load_config("mnist")


def test_readme_hyperparameters():
    """Spot-check that the config holds the exact values of Architecture_README.md."""
    c = load_config("ciciot2023")
    assert c["model"]["hidden_sizes"] == [128, 64] and c["model"]["dropout"] == 0.2
    t = c["training"]
    assert (t["lr"], t["momentum"], t["local_epochs"], t["batch_size"]) == (0.01, 0.9, 2, 256)
    assert c["fl"]["rounds"] == 40
    assert c["partition"]["n_clients"] == 20
    assert c["partition"]["dirichlet_alpha"] == 0.5 and c["partition"]["dirichlet_alpha_hard"] == 0.1
    zones = c["partition"]["zones"]
    assert [zones[z]["n_clients"] for z in "ABC"] == [7, 7, 6]
    assert sum(zones[z]["n_clients"] for z in "ABC") == c["partition"]["n_clients"]
    assert c["data"]["test_fraction"] == 0.2 and c["data"]["root_size"] == 500
    assert c["abac"]["reputation_threshold"] == 0.3
    assert c["cohorts"]["min_cohort_size"] == 6 and c["cohorts"]["k_candidates"] == [2, 3]
    assert c["inter_cohort"]["weight_cap"] == 0.5
    r = c["reputation"]
    assert (r["beta"], r["initial"], r["threshold"], r["quarantine_rounds"]) == (0.8, 0.5, 0.3, 5)
    assert c["experiment"]["malicious_fractions"] == [0.0, 0.1, 0.2, 0.3, 0.4]
    assert c["experiment"]["n_seeds"] == 3
    assert len(c["data"]["classes"]) == 8


def test_label_keyword_order_ddos_before_dos():
    kws = [k for k, _ in load_config("ciciot2023")["data"]["label_keywords"]]
    assert kws.index("DDOS") < kws.index("DOS")


def test_nbaiot_not_supported():
    with pytest.raises(ValueError):
        load_config("nbaiot")


def test_overrides_and_save_roundtrip(tmp_path):
    cfg = load_config("ciciot2023", overrides={"attacks": {"type": "sign_flip"}})
    assert cfg["attacks"]["type"] == "sign_flip"
    assert cfg["attacks"]["placement"] == "spread"  # untouched sibling key
    save_config(cfg, tmp_path / "cfg.yaml")
    assert load_yaml(tmp_path / "cfg.yaml") == cfg


def test_set_seed_reproducible():
    set_seed(123)
    a = (random.random(), np.random.rand())
    set_seed(123)
    b = (random.random(), np.random.rand())
    assert a == b


def test_set_seed_torch_reproducible():
    torch = pytest.importorskip("torch")
    set_seed(7)
    x = torch.rand(3)
    set_seed(7)
    assert torch.equal(x, torch.rand(3))


def test_make_rng_independent_streams():
    assert make_rng(1, 0).random() == make_rng(1, 0).random()
    assert make_rng(1, 0).random() != make_rng(1, 1).random()


def test_io_helpers(tmp_path):
    p = tmp_path / "run" / "result.json"
    assert not run_is_done(p)
    save_json({"acc": 0.9}, p)
    assert run_is_done(p) and load_json(p) == {"acc": 0.9}
    save_rows_csv([{"round": 1, "acc": 0.5}, {"round": 2, "acc": 0.6}], tmp_path / "m.csv")
    assert (tmp_path / "m.csv").read_text().splitlines()[0] == "round,acc"


def test_notebook_has_required_cells():
    nb_path = CONFIG_DIR.parent / "notebooks" / "colab_main.ipynb"
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    assert "drive.mount" in code
    assert "'git', 'clone'" in code
    assert "pytest" in code
