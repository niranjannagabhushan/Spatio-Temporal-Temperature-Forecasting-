"""
benchmarks/run_all.py
----------------------
Master benchmark runner.  Executes all three benchmark modules in sequence
and prints a single consolidated report.

Usage
-----
    python -m benchmarks.run_all          # from project root
    python benchmarks/run_all.py          # direct

Exit code
---------
0 — all benchmarks completed (individual failures are reported, not raised).
1 — a benchmark module itself crashed with an unhandled exception.
"""

from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Import benchmark modules
# ---------------------------------------------------------------------------

import benchmarks.bench_ml_models    as _ml
import benchmarks.bench_kriging      as _krig
import benchmarks.bench_preprocessor as _prep


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

_DIVIDER = "═" * 72


def _header(title: str) -> None:
    print(f"\n{_DIVIDER}")
    print(f"  {title}")
    print(_DIVIDER)


def _section(label: str, module, verbose: bool = True) -> list[dict]:
    """Run one benchmark module, catch exceptions, return records."""
    _header(label)
    try:
        t0      = time.perf_counter()
        records = module.run(verbose=verbose)
        elapsed = time.perf_counter() - t0
        print(f"\n  ✓  Completed in {elapsed:.2f} s")
        return records
    except Exception:
        print("\n  ✗  Benchmark raised an exception:")
        traceback.print_exc()
        return []


def main() -> int:
    wall_start = time.perf_counter()
    print(f"\n{'═'*72}")
    print("  FULL BENCHMARK SUITE — Spatio-Temporal Temperature Forecasting")
    print(f"{'═'*72}")

    ml_records   = _section("ML MODEL BENCHMARKS",       _ml)
    krig_records = _section("KRIGING BENCHMARKS",         _krig)
    prep_records = _section("PREPROCESSOR BENCHMARKS",    _prep)

    # ── Consolidated summary ────────────────────────────────────────────────
    all_records: list[dict] = []

    for r in ml_records:
        all_records.append({
            "suite":    "ML Models",
            "name":     r["model"],
            "time_s":   r["time_s"],
            "r2":       r.get("r2", "—"),
        })
    for r in krig_records:
        all_records.append({
            "suite":    "Kriging",
            "name":     r["method"],
            "time_s":   r["time_s"],
            "r2":       "—",
        })
    for r in prep_records:
        all_records.append({
            "suite":    "Preprocessor",
            "name":     r["function"],
            "time_s":   r["time_s"],
            "r2":       "—",
        })

    wall_elapsed = time.perf_counter() - wall_start

    print(f"\n{_DIVIDER}")
    print("  CONSOLIDATED SUMMARY")
    print(_DIVIDER)
    if all_records:
        df = (
            pd.DataFrame(all_records)
            .set_index(["suite", "name"])
            [["time_s", "r2"]]
        )
        print(df.to_string())
    else:
        print("  No records collected.")

    print(_DIVIDER)
    print(f"  Total wall time: {wall_elapsed:.2f} s")
    print(_DIVIDER)
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sys.exit(main())
