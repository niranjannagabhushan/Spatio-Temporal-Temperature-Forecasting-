"""
benchmarks/bench_kriging.py
----------------------------
Benchmark Ordinary Kriging and Universal Kriging interpolation on a
synthetic 80-station, 50×50-grid dataset.

Usage
-----
    python -m benchmarks.bench_kriging
    python benchmarks/bench_kriging.py

Requires pykrige.  Skips gracefully if pykrige is not installed.

Output
------
    ─────────────────────────────────────────────
    KRIGING BENCHMARK  (80 stations, 50×50 grid)
    ─────────────────────────────────────────────
    Ordinary Kriging :  0.84 s  (best of 3 runs)
    Universal Kriging:  1.12 s  (best of 3 runs)
    ─────────────────────────────────────────────
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

_N_STATIONS      =  80
_GRID_RESOLUTION =  50
_N_REPEATS       =   3
_RNG             = np.random.default_rng(5)


def _make_kriging_inputs():
    """Return (x, y, z, grid_x, grid_y) for Aosta-Valley-like coordinates."""
    x = _RNG.uniform(7.2,  7.8, _N_STATIONS)
    y = _RNG.uniform(45.5, 45.9, _N_STATIONS)
    # Simple spatial gradient: temperature decreases with altitude / latitude
    z = 22.0 - 8.0 * (y - 45.5) + _RNG.normal(0, 0.8, _N_STATIONS)

    grid_x = np.linspace(x.min(), x.max(), _GRID_RESOLUTION)
    grid_y = np.linspace(y.min(), y.max(), _GRID_RESOLUTION)
    return x, y, z, grid_x, grid_y


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(verbose: bool = True) -> list[dict]:
    """Benchmark OK and UK; return list of timing records.

    Returns
    -------
    list[dict]
        ``[{"method", "time_s", "succeeded"}, ...]``
    """
    try:
        from src.models.kriging import run_ordinary_kriging, run_universal_kriging
    except ImportError as exc:
        print(f"Cannot import kriging module: {exc}")
        return []

    try:
        import pykrige  # noqa: F401
    except ImportError:
        print("pykrige is not installed — skipping kriging benchmark.")
        return []

    x, y, z, gx, gy = _make_kriging_inputs()

    records: list[dict] = []
    for label, fn in [
        ("Ordinary Kriging",  run_ordinary_kriging),
        ("Universal Kriging", run_universal_kriging),
    ]:
        best_time = float("inf")
        succeeded = False
        import warnings
        for _ in range(_N_REPEATS):
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pred, _ = fn(x, y, z, gx, gy, variogram_model="linear")
            elapsed = time.perf_counter() - t0
            best_time = min(best_time, elapsed)
            if pred is not None:
                succeeded = True

        record = {
            "method":    label,
            "time_s":    round(best_time, 4),
            "succeeded": succeeded,
        }
        records.append(record)

        if verbose:
            status = "✓" if succeeded else "✗ (numerical failure)"
            print(f"  {label:<20}: {best_time:6.3f} s  (best of {_N_REPEATS})  {status}")

    if verbose:
        _print_table(records)

    return records


def _print_table(records: list[dict]) -> None:
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"KRIGING BENCHMARK  ({_N_STATIONS} stations, {_GRID_RESOLUTION}×{_GRID_RESOLUTION} grid)")
    print(sep)
    for r in records:
        status = "succeeded" if r["succeeded"] else "FAILED (LinAlgError)"
        print(f"  {r['method']:<22}: {r['time_s']:6.3f} s   [{status}]")
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running Kriging benchmarks …\n")
    run(verbose=True)
