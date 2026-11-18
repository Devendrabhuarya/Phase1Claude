"""Logging and result-file helpers (JSON / CSV, resumable runs)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd


def get_logger(name: str = "fl_ids", level: int = logging.INFO) -> logging.Logger:
    """Return a console logger with a short, readable format (configured once)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s",
                                               datefmt="%H:%M:%S"))
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger


def save_json(obj: Any, path: Path | str) -> None:
    """Write ``obj`` as pretty JSON, creating parent folders if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def load_json(path: Path | str) -> Any:
    """Read a JSON file."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_rows_csv(rows: Iterable[Mapping[str, Any]], path: Path | str) -> None:
    """Write a list of dicts (e.g. per-round metrics) to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False)


def run_is_done(result_path: Path | str) -> bool:
    """Return True if a run's result file already exists (used to resume grids)."""
    return Path(result_path).is_file()
