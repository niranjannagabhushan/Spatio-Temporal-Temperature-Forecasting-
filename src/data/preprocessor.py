"""
src/data/preprocessor.py
------------------------
Preprocessing pipeline: merging, GeoDataFrame construction, CRS reprojection,
spatial feature engineering, temporal feature engineering, and target cleaning.

Every step is exposed as a standalone function so individual stages can be
unit-tested or re-used independently.  ``build_pipeline`` wires them all
together using settings from ``config.yaml``.

Note on SettingWithCopyWarning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
All ``.dt`` accessor assignments use ``.loc[]`` (e.g.
``gdf.loc[:, 'hour'] = gdf['timestamp'].dt.hour``) to avoid the
``SettingWithCopyWarning`` that was present in both source notebooks.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd

from src.data.loader import (
    load_config,
    load_shapefiles,
    load_station_metadata,
    load_weather_data,
)


# ---------------------------------------------------------------------------
# Step 1 — merge tabular tables
# ---------------------------------------------------------------------------

def merge_station_data(
    weather_data: pd.DataFrame,
    station_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Right-merge weather measurements with station metadata on ``station_id``.

    A *right* merge ensures every station in the metadata table is present in
    the result even when it has no corresponding measurement rows.

    Parameters
    ----------
    weather_data:
        Measurements DataFrame with at least columns:
        ``station_id``, ``temperature``, ``timestamp``.
    station_metadata:
        Metadata DataFrame with at least columns:
        ``station_id``, ``name``, ``altitude``, ``latitude``, ``longitude``.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame keyed on ``station_id``.
    """
    return pd.merge(weather_data, station_metadata, on="station_id", how="right")


# ---------------------------------------------------------------------------
# Step 2 — build GeoDataFrame
# ---------------------------------------------------------------------------

def build_geodataframe(
    merged_data: pd.DataFrame,
    input_crs: str,
) -> gpd.GeoDataFrame:
    """Convert a merged DataFrame to a GeoDataFrame using lon/lat columns.

    Parameters
    ----------
    merged_data:
        DataFrame that must contain ``longitude`` and ``latitude`` columns.
    input_crs:
        EPSG string for the coordinate reference system of the raw coordinates
        (e.g. ``"EPSG:4326"`` for WGS-84 geographic).

    Returns
    -------
    gpd.GeoDataFrame
        Point GeoDataFrame with ``geometry`` derived from
        ``longitude`` / ``latitude`` and CRS set to *input_crs*.
    """
    return gpd.GeoDataFrame(
        merged_data,
        geometry=gpd.points_from_xy(merged_data.longitude, merged_data.latitude),
        crs=input_crs,
    )


# ---------------------------------------------------------------------------
# Step 3 — reproject to metric CRS
# ---------------------------------------------------------------------------

def reproject(
    gdf: gpd.GeoDataFrame,
    lakes: gpd.GeoDataFrame,
    rivers: gpd.GeoDataFrame,
    target_crs: str,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Reproject station GeoDataFrame, lakes, and rivers to *target_crs*.

    A metric (projected) CRS such as UTM is required before running
    ``sjoin_nearest`` so that distances are expressed in metres.

    Parameters
    ----------
    gdf:
        Station point GeoDataFrame in its original CRS.
    lakes:
        Lakes polygon GeoDataFrame.
    rivers:
        Rivers linestring GeoDataFrame.
    target_crs:
        Target EPSG string (e.g. ``"EPSG:32632"`` for UTM zone 32N).

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]
        ``(gdf, lakes, rivers)`` all reprojected to *target_crs*.
    """
    gdf = gdf.to_crs(target_crs)
    lakes = lakes.to_crs(target_crs)
    rivers = rivers.to_crs(target_crs)
    return gdf, lakes, rivers


# ---------------------------------------------------------------------------
# Step 4 — spatial feature engineering
# ---------------------------------------------------------------------------

def add_spatial_features(
    station_gdf: gpd.GeoDataFrame,
    lakes: gpd.GeoDataFrame,
    rivers: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Enrich stations with distance-to-lake and distance-to-river features.

    Performs two sequential ``sjoin_nearest`` operations and cleans up the
    ``index_right`` column produced by each join before running the next one.

    Parameters
    ----------
    station_gdf:
        Station point GeoDataFrame (must be in a projected metric CRS).
    lakes:
        Lakes polygon GeoDataFrame (same CRS as *station_gdf*).
    rivers:
        Rivers linestring GeoDataFrame (same CRS as *station_gdf*).

    Returns
    -------
    gpd.GeoDataFrame
        Enriched GeoDataFrame with additional columns:
        ``distance_to_lake`` and ``distance_to_river``.
    """
    # --- 1. Nearest lake ---
    result = gpd.sjoin_nearest(
        station_gdf, lakes, how="left", distance_col="distance_to_lake"
    )
    if "index_right" in result.columns:
        result = result.drop(columns=["index_right"])

    # --- 2. Nearest river ---
    result = gpd.sjoin_nearest(
        result, rivers, how="left", distance_col="distance_to_river"
    )
    if "index_right" in result.columns:
        result = result.drop(columns=["index_right"])

    return result


# ---------------------------------------------------------------------------
# Step 5 — temporal feature engineering
# ---------------------------------------------------------------------------

def add_temporal_features(station_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Parse the ``timestamp`` column and extract cyclic temporal features.

    Fixes the ``SettingWithCopyWarning`` from the notebooks by using
    ``.loc[]`` for every assignment.

    Parameters
    ----------
    station_gdf:
        GeoDataFrame with a ``timestamp`` column (string or datetime).

    Returns
    -------
    gpd.GeoDataFrame
        Same GeoDataFrame with four additional columns:
        ``hour``, ``day_of_week``, ``day_of_year``, ``month``.
    """
    # Ensure the column is a proper datetime type before using .dt accessors
    station_gdf.loc[:, "timestamp"] = pd.to_datetime(station_gdf["timestamp"])

    station_gdf.loc[:, "hour"] = station_gdf["timestamp"].dt.hour
    station_gdf.loc[:, "day_of_week"] = station_gdf["timestamp"].dt.dayofweek
    station_gdf.loc[:, "day_of_year"] = station_gdf["timestamp"].dt.dayofyear
    station_gdf.loc[:, "month"] = station_gdf["timestamp"].dt.month

    return station_gdf


# ---------------------------------------------------------------------------
# Step 6 — drop rows with missing target
# ---------------------------------------------------------------------------

def drop_missing_targets(
    station_gdf: gpd.GeoDataFrame,
    target_col: str = "temperature",
) -> gpd.GeoDataFrame:
    """Drop rows where the target column is NaN.

    Parameters
    ----------
    station_gdf:
        GeoDataFrame that must contain *target_col*.
    target_col:
        Name of the target variable column.  Defaults to ``"temperature"``.

    Returns
    -------
    gpd.GeoDataFrame
        Cleaned GeoDataFrame with no NaN values in *target_col*.
    """
    return station_gdf.dropna(subset=[target_col])


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------

def build_pipeline(config: dict) -> gpd.GeoDataFrame:
    """Run the full preprocessing pipeline end-to-end from a config dict.

    This is the single call a notebook needs to replace ~15 boilerplate cells:

    .. code-block:: python

        from src.data.loader import load_config
        from src.data.preprocessor import build_pipeline

        config = load_config("config.yaml")
        station_gdf = build_pipeline(config)

    Pipeline stages
    ---------------
    1. ``load_weather_data``     — read measurements CSV
    2. ``load_station_metadata`` — read & normalise metadata CSV
    3. ``load_shapefiles``       — read lakes & rivers shapefiles
    4. ``merge_station_data``    — right-merge on ``station_id``
    5. ``build_geodataframe``    — create point GeoDataFrame (EPSG:4326)
    6. ``reproject``             — reproject all layers to UTM (EPSG:32632)
    7. ``add_spatial_features``  — distance to nearest lake / river
    8. ``add_temporal_features`` — hour, day_of_week, day_of_year, month
    9. ``drop_missing_targets``  — remove rows with NaN temperature

    Parameters
    ----------
    config:
        Dictionary loaded from ``config.yaml`` via :func:`load_config`.

    Returns
    -------
    gpd.GeoDataFrame
        Fully preprocessed station GeoDataFrame ready for modelling.
    """
    # -- Resolve paths and settings from config --
    data_cfg = config["data"]
    crs_cfg = config["crs"]
    target_col = config.get("model", {}).get("target", "temperature")

    # -- Load raw data --
    weather_data = load_weather_data(data_cfg["weather_data"])
    station_metadata = load_station_metadata(data_cfg["station_metadata"])
    lakes, rivers = load_shapefiles(
        data_cfg["lakes_shapefile"],
        data_cfg["rivers_shapefile"],
    )

    # -- Tabular preprocessing --
    merged = merge_station_data(weather_data, station_metadata)

    # -- Spatial construction --
    station_gdf = build_geodataframe(merged, input_crs=crs_cfg["input"])
    station_gdf, lakes, rivers = reproject(
        station_gdf, lakes, rivers, target_crs=crs_cfg["target"]
    )

    # -- Feature engineering --
    station_gdf = add_spatial_features(station_gdf, lakes, rivers)
    station_gdf = add_temporal_features(station_gdf)

    # -- Clean target --
    station_gdf = drop_missing_targets(station_gdf, target_col=target_col)

    return station_gdf
