"""
benchmarks/bench_preprocessor.py
----------------------------------
Benchmark the two CPU-intensive helpers inside src.data.preprocessor:

  _add_temporal_features  — tested at 100 000 rows
  _distance_to_nearest    — tested at 2 000 points vs a 50-segment river

Usage
-----
    python -m benchmarks.bench_preprocessor
    python benchmarks/bench_preprocessor.py

Output example
--------------
    ─────────────────────────────────────────────────────
    PREPROCESSOR BENCHMARK
    ─────────────────────────────────────────────────────
    _add_temporal_features  (100 000 rows): 0.031 s
    _distance_to_nearest    (2 000 pts  ): 0.412 s
    ─────────────────────────────────────────────────────
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.preprocessor import _add_temporal_features, _distance_to_nearest

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

_N_TEMPORAL =  100_000   # rows for _add_temporal_features
_N_POINTS   =   2_000    # station points for _distance_to_nearest
_N_SEGMENTS =      50    # segments in the synthetic river linestring
_RNG        = np.random.default_rng(3)


def _make_temporal_df(n: int = _N_TEMPORAL) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2020-01-01", periods=n, freq="min"),
        "value":     _RNG.normal(15.0, 5.0, n),
    })


def _make_point_gdf(n: int = _N_POINTS) -> gpd.GeoDataFrame:
    lats = _RNG.uniform(45.5, 45.9, n)
    lons = _RNG.uniform( 7.2,  7.8, n)
    return gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)],
        crs="EPSG:32632",
    ).to_crs("EPSG:32632")


def _make_river_gdf(n_segments: int = _N_SEGMENTS) -> gpd.GeoDataFrame:
    """A single multi-segment linestring spanning the bounding box."""
    xs = np.linspace(7.2, 7.8, n_segments + 1)
    ys = np.linspace(45.5, 45.9, n_segments + 1)
    line = LineString(list(zip(xs, ys)))
    gdf = gpd.GeoDataFrame(geometry=[line], crs="EPSG:4326")
    return gdf.to_crs("EPSG:32632")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run(verbose: bool = True) -> list[dict]:
    """Benchmark preprocessor helpers; return list of timing records.

    Returns
    -------
    list[dict]
        ``[{"function", "n", "time_s"}, ...]``
    """
    records: list[dict] = []

    # ── _add_temporal_features ──────────────────────────────────────────────
    df = _make_temporal_df()
    if verbose:
        print(f"  _add_temporal_features ({_N_TEMPORAL:,} rows) … ", end="", flush=True)
    t0 = time.perf_counter()
    _add_temporal_features(df)
    t_temporal = time.perf_counter() - t0
    if verbose:
        print(f"{t_temporal:.4f} s")

    records.append({
        "function": "_add_temporal_features",
        "n":        _N_TEMPORAL,
        "time_s":   round(t_temporal, 4),
    })

    # ── _distance_to_nearest ────────────────────────────────────────────────
    pts = _make_point_gdf()
    river = _make_river_gdf()
    if verbose:
        print(f"  _distance_to_nearest    ({_N_POINTS:,} pts)  … ", end="", flush=True)
    t0 = time.perf_counter()
    _distance_to_nearest(pts, river)
    t_dist = time.perf_counter() - t0
    if verbose:
        print(f"{t_dist:.4f} s")

    records.append({
        "function": "_distance_to_nearest",
        "n":        _N_POINTS,
        "time_s":   round(t_dist, 4),
    })

    if verbose:
        _print_table(records)

    return records


def _print_table(records: list[dict]) -> None:
    sep = "─" * 60
    print(f"\n{sep}")
    print("PREPROCESSOR BENCHMARK")
    print(sep)
    for r in records:
        print(f"  {r['function']:<32} n={r['n']:>8,}  →  {r['time_s']:.4f} s")
    print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running preprocessor benchmarks …\n")
    run(verbose=True)
