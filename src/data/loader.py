"""
src/data/loader.py
──────────────────
Low-level I/O helpers.  Every path and setting is driven by config.yaml so
that the notebooks (and any future scripts) never hard-code file names.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import geopandas as gpd
import pandas as pd
import yaml


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> dict:
    """Load and return the YAML config as a dict.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.  Defaults to ``config.yaml``
        in the current working directory (i.e. the project root).

    Returns
    -------
    dict
        Parsed configuration dictionary.
    """
    with open(Path(config_path), "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ── Tabular loaders ───────────────────────────────────────────────────────────

def load_weather_data(path: str) -> pd.DataFrame:
    """Load weather station measurements CSV.

    Expects a CSV file with at least the columns:
    ``station_id``, ``temperature``, ``timestamp``.

    Parameters
    ----------
    path:
        File path to the weather measurements CSV.

    Returns
    -------
    pd.DataFrame
        Raw weather measurements with columns: station_id, temperature,
        timestamp.
    """
    return pd.read_csv(Path(path))


def load_station_metadata(path: str) -> pd.DataFrame:
    """Load station metadata CSV and normalise the primary-key column name.

    The raw CSV uses ``id`` as the station identifier; this function renames
    it to ``station_id`` so that downstream merge operations work without
    extra boilerplate in every caller.

    Expects columns (after rename):
    ``station_id``, ``name``, ``altitude``, ``latitude``, ``longitude``.

    Parameters
    ----------
    path:
        File path to the station metadata CSV.

    Returns
    -------
    pd.DataFrame
        Station metadata with ``id`` renamed to ``station_id``.
    """
    df = pd.read_csv(Path(path))
    df.rename(columns={"id": "station_id"}, inplace=True)
    return df


# ── Spatial loaders ───────────────────────────────────────────────────────────

def load_shapefiles(
    lakes_path: str,
    rivers_path: str,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load hydrographic shapefiles for the Aosta Valley region.

    Parameters
    ----------
    lakes_path:
        Path to the lakes polygon shapefile (``idrografia_laghi.shp``).
    rivers_path:
        Path to the rivers line shapefile (``idrografia_lineare.shp``).

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        ``(aosta_lakes, aosta_rivers)`` — one GeoDataFrame per layer.
    """
    aosta_lakes = gpd.read_file(Path(lakes_path))
    aosta_rivers = gpd.read_file(Path(rivers_path))
    return aosta_lakes, aosta_rivers
