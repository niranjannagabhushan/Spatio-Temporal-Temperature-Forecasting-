"""
tests/conftest.py
-----------------
Shared pytest fixtures used across the entire test suite.

Fixtures
--------
minimal_config     (session) — smallest valid config dict; uses tiny model
                               settings so trainers finish in < 1 s.
synthetic_station_gdf (session) — 200-row GeoDataFrame that mirrors the
                               output of build_pipeline(), usable by every
                               test that needs a realistic feature matrix.
xy_kriging         (session) — (x, y, z, grid_x, grid_y) tuple for kriging
                               tests; purely synthetic station coordinates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point

# ── reproducible randomness ──────────────────────────────────────────────────
_RNG = np.random.default_rng(0)
_N   = 200  # rows in the synthetic GeoDataFrame


# ─────────────────────────────────────────────────────────────────────────────
# 1.  minimal_config
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def minimal_config() -> dict:
    """Config dict with tiny hyper-parameter values for fast test execution.

    All ML trainers respect their config keys, so using n_estimators=10,
    epochs=2, etc. keeps the suite under a few seconds even on CI runners.
    """
    return {
        "data": {
            "weather_data": "weather.csv",
            "station_metadata": "stations.csv",
            "lakes_shapefile":  "lakes.shp",
            "rivers_shapefile": "rivers.shp",
        },
        "crs": {
            "input":  "EPSG:4326",
            "target": "EPSG:32632",
        },
        "preprocessing": {
            "target_col": "temperature",
        },
        "features": {
            "target":    "temperature",
            # Columns dropped when building X; must exist in synthetic_station_gdf
            "drop_cols": ["temperature", "geometry", "station_id"],
        },
        "models": {
            "ridge":   {"alpha": 1.0},
            "lasso":   {"alpha": 1.0},
            "svr":     {"kernel": "rbf", "C": 1.0},
            "random_forest": {"n_estimators": 10, "random_state": 42},
            "lightgbm": {
                "n_estimators":  10,
                "learning_rate": 0.1,
                "max_depth":     3,
            },
            "xgboost": {
                "n_estimators":  10,
                "learning_rate": 0.1,
                "max_depth":     3,
                "random_state":  42,
            },
            "kriging": {
                "variogram_model": "linear",
                "grid_resolution": 10,
            },
            "transformer": {
                "d_model":    16,
                "num_heads":   2,
                "num_layers":  1,
                "dropout":     0.0,
                "epochs":      2,
                "batch_size": 32,
            },
            "tcn": {
                "filters":     8,
                "kernel_size": 2,
                "num_blocks":  1,
                "dropout":     0.0,
                "epochs":      2,
                "batch_size": 32,
            },
            "keras_tuner_mlp": {
                "max_trials":        1,
                "epochs":            2,
                "validation_split":  0.2,
            },
            "gcn": {
                "n_subsample":  100,
                "k_neighbours":   3,
                "hidden_dim":    16,
                "lr":         1e-3,
                "epochs":        2,
                "seed":         42,
            },
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2.  synthetic_station_gdf
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def synthetic_station_gdf() -> gpd.GeoDataFrame:
    """200-row GeoDataFrame matching the schema produced by build_pipeline().

    Columns
    -------
    station_id, temperature, altitude, latitude, longitude,
    distance_to_lake, distance_to_river, hour, month, geometry (EPSG:4326).
    """
    n    = _N
    rng  = _RNG
    lats = rng.uniform(45.50, 45.90, n)
    lons = rng.uniform( 7.20,  7.80, n)

    data = {
        "station_id":         np.arange(n, dtype=int),
        "temperature":        rng.normal(15.0,  5.0, n).astype(float),
        "altitude":           rng.uniform(500.0, 2000.0, n),
        "latitude":           lats,
        "longitude":          lons,
        "distance_to_lake":   rng.uniform(100.0, 10_000.0, n),
        "distance_to_river":  rng.uniform( 50.0,  5_000.0, n),
        "hour":               rng.integers(0, 24, n).astype(int),
        "month":              rng.integers(1, 13, n).astype(int),
    }

    gdf = gpd.GeoDataFrame(
        data,
        geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)],
        crs="EPSG:4326",
    )
    return gdf


# ─────────────────────────────────────────────────────────────────────────────
# 3.  xy_kriging
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def xy_kriging():
    """Synthetic kriging inputs: (x, y, z, grid_x, grid_y).

    30 station points with a simple linear temperature gradient used to
    verify that run_ordinary_kriging / run_universal_kriging return
    arrays of the expected shape when pykrige is available.
    """
    rng    = np.random.default_rng(7)
    x      = rng.uniform(7.2, 7.8, 30)
    y      = rng.uniform(45.5, 45.9, 30)
    # Simple deterministic signal: temperature decreases with latitude
    z      = 20.0 - 10.0 * (y - 45.5) + rng.normal(0, 0.5, 30)
    grid_x = np.linspace(x.min(), x.max(), 10)
    grid_y = np.linspace(y.min(), y.max(), 10)
    return x, y, z, grid_x, grid_y
