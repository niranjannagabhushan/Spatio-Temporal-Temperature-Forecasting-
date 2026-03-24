"""
src/data/preprocessor.py
────────────────────────
Full preprocessing pipeline extracted from the duplicated notebook cells in
``krigmain.ipynb`` and ``mlkricat.ipynb``.

Pipeline order (also reflected in ``build_pipeline``):
    1.  load_config           – read config.yaml (via loader.py)
    2.  load_weather_data     – raw measurements CSV
    3.  load_station_metadata – station metadata CSV (renames 'id' → 'station_id')
    4.  load_shapefiles       – lakes and rivers GeoDataFrames
    5.  merge_station_data    – right-merge on station_id
    6.  build_geodataframe    – DataFrame → GeoDataFrame (EPSG:4326)
    7.  reproject             – reproject all layers to EPSG:32632 for metric ops
    8.  add_spatial_features  – nearest-join distances to lake and river
    9.  add_temporal_features – parse timestamp, extract hour/dow/doy/month
    10. drop_missing_targets  – drop rows where temperature is NaN

Fixes applied vs. the original notebooks
-----------------------------------------
* All ``dt`` accessor assignments use ``.loc[]`` to suppress the
  ``SettingWithCopyWarning`` that appeared in both notebooks.
* ``index_right`` is dropped after *each* spatial join (not only the first).
* ``pd.to_datetime()`` is explicitly called before extracting temporal features.
"""

from __future__ import annotations

from typing import Tuple

import geopandas as gpd
import pandas as pd

from src.data.loader import (
    load_config,
    load_shapefiles,
    load_station_metadata,
    load_weather_data,
)


# ── Step 5 ────────────────────────────────────────────────────────────────────

def merge_station_data(
    weather_data: pd.DataFrame,
    station_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Right-merge weather measurements with station metadata on ``station_id``.

    A *right* merge is used so that every station in the metadata is kept even
    if it has no measurements (those rows will have NaN temperatures and are
    later removed by ``drop_missing_targets``).

    Parameters
    ----------
    weather_data:
        DataFrame with columns: station_id, temperature, timestamp.
    station_metadata:
        DataFrame with columns: station_id, name, altitude, latitude,
        longitude.

    Returns
    -------
    pd.DataFrame
        Merged DataFrame combining both inputs.
    """
    return pd.merge(weather_data, station_metadata, on="station_id", how="right")


# ── Step 6 ────────────────────────────────────────────────────────────────────

def build_geodataframe(
    merged_data: pd.DataFrame,
    input_crs: str,
) -> gpd.GeoDataFrame:
    """Convert a merged DataFrame to a point GeoDataFrame.

    Geometry is derived from the ``longitude`` and ``latitude`` columns, and
    the CRS is set to *input_crs* (typically ``EPSG:4326``).

    Parameters
    ----------
    merged_data:
        Output of ``merge_station_data``.
    input_crs:
        EPSG string for the source geographic CRS, e.g. ``"EPSG:4326"``.

    Returns
    -------
    gpd.GeoDataFrame
        Point GeoDataFrame in *input_crs*.
    """
    return gpd.GeoDataFrame(
        merged_data,
        geometry=gpd.points_from_xy(merged_data.longitude, merged_data.latitude),
        crs=input_crs,
    )


# ── Step 7 ────────────────────────────────────────────────────────────────────

def reproject(
    gdf: gpd.GeoDataFrame,
    lakes: gpd.GeoDataFrame,
    rivers: gpd.GeoDataFrame,
    target_crs: str,
) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Reproject stations, lakes, and rivers to a common projected CRS.

    All three layers must share the same CRS before spatial nearest-joins and
    metric distance computations.  The target is typically ``EPSG:32632``
    (UTM zone 32 N), which gives distances in metres.

    Parameters
    ----------
    gdf:
        Station GeoDataFrame (output of ``build_geodataframe``).
    lakes:
        Lakes GeoDataFrame (output of ``load_shapefiles``).
    rivers:
        Rivers GeoDataFrame (output of ``load_shapefiles``).
    target_crs:
        EPSG string for the projected CRS, e.g. ``"EPSG:32632"``.

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]
        ``(gdf, lakes, rivers)`` all reprojected to *target_crs*.
    """
    gdf = gdf.to_crs(target_crs)
    lakes = lakes.to_crs(target_crs)
    rivers = rivers.to_crs(target_crs)
    return gdf, lakes, rivers


# ── Step 8 ────────────────────────────────────────────────────────────────────
def _distance_to_nearest(
    points_gdf: gpd.GeoDataFrame,
    reference_gdf: gpd.GeoDataFrame,
) -> pd.Series:
    """Private helper: distance from each point to the nearest reference geometry.

    Returns a ``pd.Series`` of distances aligned to *points_gdf*'s index.
    Both GeoDataFrames must share the same CRS.
    """
    union = reference_gdf.geometry.union_all()
    return points_gdf.geometry.distance(union)


def add_spatial_features(
    station_gdf: gpd.GeoDataFrame,
    lakes: gpd.GeoDataFrame,
    rivers: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """Enrich stations with nearest-lake and nearest-river distances.

    Two sequential ``sjoin_nearest`` calls add:
    * ``distance_to_lake``  – metric distance to the closest lake polygon.
    * ``distance_to_river`` – metric distance to the closest river line.

    The ``index_right`` column introduced by each join is dropped immediately
    after that join to avoid name conflicts in the second join.

    Parameters
    ----------
    station_gdf:
        Reprojected station GeoDataFrame (output of ``reproject``).
    lakes:
        Reprojected lakes GeoDataFrame.
    rivers:
        Reprojected rivers GeoDataFrame.

    Returns
    -------
    gpd.GeoDataFrame
        Station GeoDataFrame with ``distance_to_lake`` and
        ``distance_to_river`` columns added.
    """
    # ── 1. Nearest lake ───────────────────────────────────────────────────────
    station_gdf = gpd.sjoin_nearest(
        station_gdf,
        lakes,
        how="left",
        distance_col="distance_to_lake",
    )
    if "index_right" in station_gdf.columns:
        station_gdf.drop(columns=["index_right"], inplace=True)

    # ── 2. Nearest river ──────────────────────────────────────────────────────
    station_gdf = gpd.sjoin_nearest(
        station_gdf,
        rivers,
        how="left",
        distance_col="distance_to_river",
    )
    if "index_right" in station_gdf.columns:
        station_gdf.drop(columns=["index_right"], inplace=True)

    return station_gdf



# ── Step 9 ────────────────────────────────────────────────────────────────────
def _add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Private helper: add temporal columns to a plain DataFrame.

    Returns a copy — the original is never mutated.
    If no ``timestamp`` column is present, returns the copy unchanged.
    Bad timestamps are coerced to NaT rather than raising.
    """
    df = df.copy()
    if "timestamp" not in df.columns:
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["day_of_year"] = df["timestamp"].dt.dayofyear
    df["month"]       = df["timestamp"].dt.month
    return df

def add_temporal_features(station_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Parse the timestamp column and extract cyclic temporal features.

    The ``timestamp`` column is parsed with ``pd.to_datetime()`` before
    feature extraction.  All assignments use ``.loc[]`` to avoid the
    ``SettingWithCopyWarning`` that was present in the original notebooks.

    New columns added:
    * ``hour``        – hour of the day (0–23)
    * ``day_of_week`` – day of the week (0 = Monday … 6 = Sunday)
    * ``day_of_year`` – ordinal day of the year (1–366)
    * ``month``       – month of the year (1–12)

    Parameters
    ----------
    station_gdf:
        GeoDataFrame with a ``timestamp`` column (string or datetime).

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with the four new temporal feature columns appended.
    """
    # Ensure timestamp is a proper datetime dtype before using .dt accessor
    station_gdf.loc[:, "timestamp"] = pd.to_datetime(station_gdf["timestamp"])

    station_gdf.loc[:, "hour"]        = station_gdf["timestamp"].dt.hour
    station_gdf.loc[:, "day_of_week"] = station_gdf["timestamp"].dt.dayofweek
    station_gdf.loc[:, "day_of_year"] = station_gdf["timestamp"].dt.dayofyear
    station_gdf.loc[:, "month"]       = station_gdf["timestamp"].dt.month

    return station_gdf


# ── Step 10 ───────────────────────────────────────────────────────────────────

def drop_missing_targets(
    station_gdf: gpd.GeoDataFrame,
    target_col: str = "temperature",
) -> gpd.GeoDataFrame:
    """Drop rows where the modelling target is NaN.

    Stations that have no measurements after the right-merge will have NaN
    temperatures; they cannot be used for training or evaluation and are
    removed here.

    Parameters
    ----------
    station_gdf:
        Enriched station GeoDataFrame.
    target_col:
        Name of the target column to check for missing values.
        Defaults to ``"temperature"``.

    Returns
    -------
    gpd.GeoDataFrame
        Cleaned GeoDataFrame with no NaN values in *target_col*.
    """
    return station_gdf.dropna(subset=[target_col])


# ── Convenience entry-point ───────────────────────────────────────────────────

def build_pipeline(config: dict) -> gpd.GeoDataFrame:
    """Run the full data loading and preprocessing pipeline from a config dict.

    This is the single function that notebooks can call instead of repeating
    15+ cells of identical setup code:

    .. code-block:: python

        from src.data.preprocessor import build_pipeline
        from src.data.loader import load_config

        config = load_config("config.yaml")
        station_gdf = build_pipeline(config)

    Pipeline steps executed in order
    ---------------------------------
    1.  ``load_weather_data``     – raw measurements CSV
    2.  ``load_station_metadata`` – station metadata CSV
    3.  ``load_shapefiles``       – lakes and rivers GeoDataFrames
    4.  ``merge_station_data``    – right-merge on station_id
    5.  ``build_geodataframe``    – DataFrame → GeoDataFrame (input CRS)
    6.  ``reproject``             – reproject all layers to target CRS
    7.  ``add_spatial_features``  – nearest lake/river distances
    8.  ``add_temporal_features`` – hour, day_of_week, day_of_year, month
    9.  ``drop_missing_targets``  – remove rows with NaN temperature

    Parameters
    ----------
    config:
        Dictionary loaded from ``config.yaml`` via ``load_config()``.

    Returns
    -------
    gpd.GeoDataFrame
        Fully preprocessed station GeoDataFrame, ready for modelling.
    """
    data_cfg = config["data"]
    crs_cfg  = config["crs"]
    pre_cfg  = config["preprocessing"]

    # ── I/O ──────────────────────────────────────────────────────────────────
    weather_data     = load_weather_data(data_cfg["weather_data"])
    station_metadata = load_station_metadata(data_cfg["station_metadata"])
    lakes, rivers    = load_shapefiles(
        data_cfg["lakes_shapefile"],
        data_cfg["rivers_shapefile"],
    )

    # ── Preprocessing ────────────────────────────────────────────────────────
    merged_data  = merge_station_data(weather_data, station_metadata)
    station_gdf  = build_geodataframe(merged_data, input_crs=crs_cfg["input"])
    station_gdf, lakes, rivers = reproject(
        station_gdf, lakes, rivers, target_crs=crs_cfg["target"]
    )
    station_gdf  = add_spatial_features(station_gdf, lakes, rivers)
    station_gdf  = add_temporal_features(station_gdf)
    station_gdf  = drop_missing_targets(station_gdf, target_col=pre_cfg["target_col"])

    return station_gdf

# Private aliases used by tests
_add_temporal_features = add_temporal_features

def _distance_to_nearest(points_gdf, reference_gdf):
    return points_gdf.geometry.apply(
        lambda pt: reference_gdf.geometry.distance(pt).min()
    )
