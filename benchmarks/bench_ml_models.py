"""
benchmarks/bench_ml_models.py
------------------------------
Benchmark all six ML model trainers on a 3 000-row synthetic dataset.

Usage
-----
    python -m benchmarks.bench_ml_models          # from project root
    python benchmarks/bench_ml_models.py          # direct

Output
------
A table printed to stdout:

    ┌────────────────┬──────────┬────────────────────┐
    │ Model          │ Time (s) │ Notes              │
    ├────────────────┼──────────┼────────────────────┤
    │ Ridge          │    0.123 │ GridSearchCV cv=5  │
    ...

Each trainer is called once with small hyper-parameters (n_estimators=50,
cv=5, etc.) to give a meaningful timing without taking minutes.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.ml_models import (
    train_ridge,
    train_lasso,
    train_svr,
    train_random_forest,
    train_lightgbm,
    train_xgboost,
)

# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

_N_TRAIN   = 2_400
_N_TEST    =   600
_N_FEATURES =    7
_RNG       = np.random.default_rng(42)

def _make_data():
    X_train = pd.DataFrame(
        _RNG.normal(0.0, 1.0, (_N_TRAIN, _N_FEATURES)),
        columns=[f"f{i}" for i in range(_N_FEATURES)],
    )
    X_test = pd.DataFrame(
        _RNG.normal(0.0, 1.0, (_N_TEST, _N_FEATURES)),
        columns=[f"f{i}" for i in range(_N_FEATURES)],
    )
    # Simple linear target with noise
    weights  = _RNG.normal(0.0, 1.0, _N_FEATURES)
    y_train  = pd.Series(X_train.values @ weights + _RNG.normal(0, 0.5, _N_TRAIN))
    y_test   = pd.Series(X_test.values  @ weights + _RNG.normal(0, 0.5, _N_TEST))
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Benchmark config — balanced between speed and realism
# ---------------------------------------------------------------------------

BENCH_CONFIG = {
    "models": {
        "ridge":         {"alpha": 1.0},
        "lasso":         {"alpha": 1.0},
        "svr":           {"kernel": "rbf", "C": 1.0},
        "random_forest": {"n_estimators": 50, "random_state": 42},
        "lightgbm":      {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 5},
        "xgboost":       {
            "n_estimators":  50,
            "learning_rate": 0.1,
            "max_depth":     5,
            "random_state":  42,
        },
    },
}

# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_TRAINERS: list[tuple[str, Callable, str]] = [
    ("Ridge",        train_ridge,         "GridSearchCV cv=5"),
    ("Lasso",        train_lasso,         "GridSearchCV cv=5"),
    ("SVR",          train_svr,           "10 % subsample of train"),
    ("RandomForest", train_random_forest, "n_estimators=50"),
    ("LightGBM",     train_lightgbm,      "n_estimators=50"),
    ("XGBoost",      train_xgboost,       "n_estimators=50"),
]


def run(verbose: bool = True) -> list[dict]:
    """Run all six trainers and return a list of timing records.

    Parameters
    ----------
    verbose:
        When True, print progress and the final table.

    Returns
    -------
    list[dict]
        One dict per model: ``{"model", "time_s", "mse", "mae", "r2"}``.
    """
    X_train, X_test, y_train, y_test = _make_data()

    records: list[dict] = []
    for name, trainer, notes in _TRAINERS:
        if verbose:
            print(f"  Benchmarking {name:<14} ... ", end="", flush=True)

        t0     = time.perf_counter()
        result = trainer(X_train, y_train, X_test, y_test, BENCH_CONFIG)
        elapsed = time.perf_counter() - t0

        record = {
            "model":  name,
            "time_s": round(elapsed, 4),
            "mse":    round(result["mse"], 4),
            "mae":    round(result["mae"], 4),
            "r2":     round(result["r2"],  4),
            "notes":  notes,
        }
        records.append(record)
        if verbose:
            print(f"{elapsed:6.2f} s")

    if verbose:
        _print_table(records)

    return records


def _print_table(records: list[dict]) -> None:
    df = (
        pd.DataFrame(records)
        .set_index("model")
        [["time_s", "mse", "mae", "r2", "notes"]]
        .sort_values("time_s")
    )
    sep = "─" * 72
    print(f"\n{sep}")
    print("ML MODEL BENCHMARK  (3 000 rows, 7 features)")
    print(sep)
    print(df.to_string())
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running ML model benchmarks …\n")
    run(verbose=True)
