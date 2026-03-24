"""
src/data/loader.py
------------------
Raw I/O helpers: reading CSVs, shapefiles, and the central YAML config.

All paths passed to these functions should come from config.yaml so that
nothing is hard-coded in the modelling notebooks.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    """Load and return the YAML config as a dict.

    Parameters
    ----------
    config_path:
        Path to the YAML configuration file.  Defaults to ``"config.yaml"``
        which is the file at the project root.

    Returns
    -------
    dict
        Parsed YAML content.
    """
    with open(Path(config_path), "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Tabular loaders
# ---------------------------------------------------------------------------

def load_weather_data(path: str) -> pd.DataFrame:
    """Load weather station measurements CSV.

    Expects the CSV to contain at minimum the columns:
    ``station_id``, ``temperature``, ``timestamp``.

    Parameters
    ----------
    path:
        Path to the measurements CSV file
        (e.g. ``"weather_station_data_202406202154.csv"``).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: ``station_id``, ``temperature``, ``timestamp``.
    """
    return pd.read_csv(Path(path))


def load_station_metadata(path: str) -> pd.DataFrame:
    """Load station metadata CSV and normalise the station identifier column.

    The raw CSV uses ``"id"`` as the station identifier; this function renames
    it to ``"station_id"`` so it aligns with the measurements table before
    merging.

    Parameters
    ----------
    path:
        Path to the station metadata CSV file
        (e.g. ``"weather_stations_202406211148.csv"``).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ``station_id``, ``name``, ``altitude``, ``latitude``, ``longitude``.
    """
    df = pd.read_csv(Path(path))
    df = df.rename(columns={"id": "station_id"})
    return df


# ---------------------------------------------------------------------------
# Geospatial loaders
# ---------------------------------------------------------------------------

def load_shapefiles(
    lakes_path: str,
    rivers_path: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load and return the lakes and rivers GeoDataFrames from shapefiles.

    Parameters
    ----------
    lakes_path:
        Path to the lakes shapefile (e.g. ``"idrografia_laghi.shp"``).
    rivers_path:
        Path to the rivers shapefile (e.g. ``"idrografia_lineare.shp"``).

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]
        ``(aosta_lakes, aosta_rivers)`` — one GeoDataFrame per layer.
    """
    aosta_lakes = gpd.read_file(Path(lakes_path))
    aosta_rivers = gpd.read_file(Path(rivers_path))
    return aosta_lakes, aosta_rivers
