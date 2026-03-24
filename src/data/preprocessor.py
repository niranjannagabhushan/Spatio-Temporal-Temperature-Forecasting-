"""
src/data/preprocessor.py
------------------------
Feature-engineering pipeline that turns raw CSVs + shapefiles into a
spatially-enriched GeoDataFrame ready for modelling.

Key public symbol
-----------------
build_pipeline(config, base_path='.')
    Accepts an optional *base_path* so that notebooks living in a
    ``notebooks/`` subdirectory can resolve project-root paths from
    ``config.yaml`` by passing ``base_path='..'``.
"""

from __future__ import annotations

import os

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.ops import nearest_points

from .loader import (
    load_weather_data,
    load_station_metadata,
    load_shapefiles,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _distance_to_nearest(
    point_gdf: gpd.GeoDataFrame,
    reference_gdf: gpd.GeoDataFrame,
) -> pd.Series:
    """Return the distance (metres) from each point to the nearest geometry
    in *reference_gdf*, both assumed to be in the same projected CRS."""
    ref_union = reference_gdf.geometry.unary_union
    distances = point_gdf.geometry.apply(
        lambda geom: geom.distance(ref_union)
    )
    return distances


def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract ``hour`` and ``month`` from the ``timestamp`` column."""
    if "timestamp" not in df.columns:
        return df
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.copy()
    df["hour"] = ts.dt.hour
    df["month"] = ts.dt.month
    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_pipeline(config: dict, base_path: str = ".") -> gpd.GeoDataFrame:
    """Load, merge, and engineer features for the full dataset.

    Parameters
    ----------
    config:
        Dictionary loaded from ``config.yaml`` (via
        :func:`src.data.loader.load_config`).
    base_path:
        Root directory used to resolve relative paths stored in *config*.
        Notebooks in ``notebooks/`` should pass ``base_path='..'``; scripts
        at the project root can use the default ``'.'``.

    Returns
    -------
    geopandas.GeoDataFrame
        One row per weather reading with columns including:
        ``station_id``, ``temperature``, ``altitude``,
        ``distance_to_lake``, ``distance_to_river``, ``hour``, ``month``,
        and a ``geometry`` point column (EPSG:4326 → projected for distance
        calculations, then stored in EPSG:4326).
    """
    data_cfg = config["data"]

    # ------------------------------------------------------------------
    # 1. Load raw inputs
    # ------------------------------------------------------------------
    weather_path = os.path.join(base_path, data_cfg["weather_data"])
    station_path = os.path.join(base_path, data_cfg["station_data"])
    lakes_path = os.path.join(base_path, data_cfg["lakes_shp"])
    rivers_path = os.path.join(base_path, data_cfg["rivers_shp"])

    weather_df = load_weather_data(weather_path)
    station_df = load_station_metadata(station_path)
    aosta_lakes, aosta_rivers = load_shapefiles(lakes_path, rivers_path)

    # ------------------------------------------------------------------
    # 2. Merge weather readings with station metadata
    # ------------------------------------------------------------------
    merged = pd.merge(weather_df, station_df, on="station_id", how="inner")

    # ------------------------------------------------------------------
    # 3. Build GeoDataFrame (geographic CRS)
    # ------------------------------------------------------------------
    gdf = gpd.GeoDataFrame(
        merged,
        geometry=gpd.points_from_xy(merged["longitude"], merged["latitude"]),
        crs="EPSG:4326",
    )

    # ------------------------------------------------------------------
    # 4. Project to a metric CRS for distance calculations
    # ------------------------------------------------------------------
    projected_crs = "EPSG:32632"  # UTM zone 32N — appropriate for Aosta Valley
    gdf_proj = gdf.to_crs(projected_crs)
    lakes_proj = aosta_lakes.to_crs(projected_crs)
    rivers_proj = aosta_rivers.to_crs(projected_crs)

    # ------------------------------------------------------------------
    # 5. Compute proximity features
    # ------------------------------------------------------------------
    gdf_proj["distance_to_lake"] = _distance_to_nearest(gdf_proj, lakes_proj)
    gdf_proj["distance_to_river"] = _distance_to_nearest(gdf_proj, rivers_proj)

    # ------------------------------------------------------------------
    # 6. Temporal features
    # ------------------------------------------------------------------
    gdf_proj = _add_temporal_features(gdf_proj)

    # ------------------------------------------------------------------
    # 7. Drop rows missing the regression target
    # ------------------------------------------------------------------
    gdf_proj = gdf_proj.dropna(subset=["temperature"]).reset_index(drop=True)

    # Return in geographic CRS for compatibility with mapping cells
    return gdf_proj.to_crs("EPSG:4326")
