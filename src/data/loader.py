"""
src/data/loader.py
------------------
Low-level I/O helpers.  Every function accepts explicit file paths so that
callers (notebooks or pipeline code) can resolve paths however they like.
"""

from __future__ import annotations

import os
from typing import Tuple

import geopandas as gpd
import pandas as pd
import yaml


def load_config(config_path: str) -> dict:
    """Load a YAML configuration file and return it as a plain dict.

    Parameters
    ----------
    config_path:
        Absolute or relative path to ``config.yaml``.
    """
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_weather_data(path: str) -> pd.DataFrame:
    """Read the raw weather-station measurement CSV.

    Expected columns: ``station_id``, ``temperature``, ``timestamp``.

    Parameters
    ----------
    path:
        Path to the weather-station CSV file.
    """
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def load_station_metadata(path: str) -> pd.DataFrame:
    """Read the weather-station metadata CSV.

    Expected columns: ``id`` (or ``station_id``), ``name``, ``altitude``,
    ``latitude``, ``longitude``.

    Parameters
    ----------
    path:
        Path to the station-metadata CSV file.
    """
    df = pd.read_csv(path)
    # Normalise the primary key column name used throughout the pipeline.
    if "id" in df.columns and "station_id" not in df.columns:
        df = df.rename(columns={"id": "station_id"})
    return df


def load_shapefiles(
    lakes_path: str,
    rivers_path: str,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Load the lake-polygon and river-linestring shapefiles.

    Parameters
    ----------
    lakes_path:
        Path to the lakes shapefile (e.g. ``idrografia_laghi.shp``).
    rivers_path:
        Path to the rivers shapefile (e.g. ``idrografia_lineare.shp``).

    Returns
    -------
    tuple
        ``(aosta_lakes, aosta_rivers)`` as :class:`geopandas.GeoDataFrame`
        objects.
    """
    aosta_lakes = gpd.read_file(lakes_path)
    aosta_rivers = gpd.read_file(rivers_path)
    return aosta_lakes, aosta_rivers
