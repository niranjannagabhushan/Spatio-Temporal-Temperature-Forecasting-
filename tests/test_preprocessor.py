"""
tests/test_preprocessor.py
---------------------------
Unit tests for src.data.preprocessor:
  _add_temporal_features, _distance_to_nearest, build_pipeline.

build_pipeline performs real file I/O, so all three loader functions are
patched out with synthetic in-memory data.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import LineString, Point, Polygon

# Import private helpers directly — they are pure functions and testable in
# isolation even though they are not part of the public API.
from src.data.preprocessor import (
    _add_temporal_features,
    _distance_to_nearest,
    build_pipeline,
)


# ---------------------------------------------------------------------------
# _add_temporal_features
# ---------------------------------------------------------------------------

class TestAddTemporalFeatures:
    def test_adds_hour_column(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-06-15 08:30:00", "2024-12-01 23:00:00"]),
            "value": [1, 2],
        })
        result = _add_temporal_features(df)
        assert "hour" in result.columns

    def test_adds_month_column(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-06-15 08:30:00", "2024-12-01 23:00:00"]),
            "value": [1, 2],
        })
        result = _add_temporal_features(df)
        assert "month" in result.columns

    def test_hour_values_correct(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01 06:00:00", "2024-01-01 14:00:00"]),
        })
        result = _add_temporal_features(df)
        assert list(result["hour"]) == [6, 14]

    def test_month_values_correct(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-03-10 00:00:00", "2024-11-20 00:00:00"]),
        })
        result = _add_temporal_features(df)
        assert list(result["month"]) == [3, 11]

    def test_no_timestamp_column_returns_unchanged(self):
        df = pd.DataFrame({"value": [1, 2, 3]})
        result = _add_temporal_features(df)
        assert "hour"  not in result.columns
        assert "month" not in result.columns
        assert list(result["value"]) == [1, 2, 3]

    def test_original_dataframe_not_mutated(self):
        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2024-01-01 00:00:00"]),
        })
        original_cols = set(df.columns)
        _add_temporal_features(df)
        assert set(df.columns) == original_cols  # function must copy

    def test_invalid_timestamp_values_coerced(self):
        """Malformed timestamps must be coerced to NaT, not raise."""
        df = pd.DataFrame({"timestamp": ["not-a-date", "2024-01-01 06:00:00"]})
        result = _add_temporal_features(df)
        # First row: NaT → hour/month are NaN (not raised)
        assert "hour" in result.columns


# ---------------------------------------------------------------------------
# _distance_to_nearest
# ---------------------------------------------------------------------------

class TestDistanceToNearest:
    """All GeoDataFrames use EPSG:32632 (metric, UTM zone 32N)."""

    def _make_point_gdf(self, coords):
        return gpd.GeoDataFrame(
            geometry=[Point(x, y) for x, y in coords],
            crs="EPSG:32632",
        )

    def _make_line_gdf(self, lines):
        return gpd.GeoDataFrame(
            geometry=[LineString(pts) for pts in lines],
            crs="EPSG:32632",
        )

    def test_distances_are_non_negative(self):
        pts = self._make_point_gdf([(0, 0), (1, 0), (0, 1)])
        ref = self._make_line_gdf([[(2, -1), (2, 2)]])
        dists = _distance_to_nearest(pts, ref)
        assert (dists >= 0).all()

    def test_length_matches_input(self):
        pts = self._make_point_gdf([(i, 0) for i in range(8)])
        ref = self._make_line_gdf([[(0, 10), (10, 10)]])
        dists = _distance_to_nearest(pts, ref)
        assert len(dists) == 8

    def test_known_distance(self):
        """Point (0,0) → vertical line at x=3: distance should be 3.0."""
        pts = self._make_point_gdf([(0, 0)])
        ref = self._make_line_gdf([[(3, -10), (3, 10)]])
        dists = _distance_to_nearest(pts, ref)
        assert dists.iloc[0] == pytest.approx(3.0, abs=1e-6)

    def test_point_on_geometry_is_zero(self):
        """A point lying exactly on the reference line has distance 0."""
        pts = self._make_point_gdf([(5, 0)])
        ref = self._make_line_gdf([[(0, 0), (10, 0)]])
        dists = _distance_to_nearest(pts, ref)
        assert dists.iloc[0] == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# build_pipeline  (loader functions mocked)
# ---------------------------------------------------------------------------

def _make_synthetic_pipeline_data():
    """Return (weather_df, station_df, lakes_gdf, rivers_gdf) for mocking."""
    rng = np.random.default_rng(99)
    n_stations = 5
    n_readings = 20

    station_ids = np.arange(1, n_stations + 1)
    lats  = rng.uniform(45.5, 45.9, n_stations)
    lons  = rng.uniform( 7.2,  7.8, n_stations)

    station_df = pd.DataFrame({
        "station_id": station_ids,
        "altitude":   rng.uniform(500, 2000, n_stations),
        "latitude":   lats,
        "longitude":  lons,
    })

    # n_readings weather observations, cycling over stations
    weather_df = pd.DataFrame({
        "station_id":  np.tile(station_ids, n_readings // n_stations + 1)[:n_readings],
        "temperature": rng.normal(15.0, 5.0, n_readings),
        "timestamp":   pd.date_range("2024-01-01", periods=n_readings, freq="h"),
    })

    lakes_gdf = gpd.GeoDataFrame(
        geometry=[Polygon([(7.0, 45.5), (7.2, 45.5), (7.2, 45.7), (7.0, 45.7)])],
        crs="EPSG:4326",
    )
    rivers_gdf = gpd.GeoDataFrame(
        geometry=[LineString([(7.0, 45.5), (7.8, 45.9)])],
        crs="EPSG:4326",
    )
    return weather_df, station_df, lakes_gdf, rivers_gdf


@pytest.fixture
def pipeline_mocks():
    """Patch all three loaders used inside build_pipeline."""
    weather_df, station_df, lakes_gdf, rivers_gdf = _make_synthetic_pipeline_data()
    with (
        patch("src.data.preprocessor.load_weather_data",  return_value=weather_df),
        patch("src.data.preprocessor.load_station_metadata", return_value=station_df),
        patch("src.data.preprocessor.load_shapefiles",    return_value=(lakes_gdf, rivers_gdf)),
    ):
        yield


class TestBuildPipeline:
    def test_returns_geodataframe(self, pipeline_mocks, minimal_config):
        result = build_pipeline(minimal_config)
        assert isinstance(result, gpd.GeoDataFrame)

    def test_temperature_column_present(self, pipeline_mocks, minimal_config):
        result = build_pipeline(minimal_config)
        assert "temperature" in result.columns

    def test_distance_columns_present(self, pipeline_mocks, minimal_config):
        result = build_pipeline(minimal_config)
        assert "distance_to_lake"   in result.columns
        assert "distance_to_river"  in result.columns

    def test_temporal_columns_present(self, pipeline_mocks, minimal_config):
        result = build_pipeline(minimal_config)
        assert "hour"  in result.columns
        assert "month" in result.columns

    def test_no_null_temperatures(self, pipeline_mocks, minimal_config):
        """Rows with null temperature must be dropped."""
        result = build_pipeline(minimal_config)
        assert result["temperature"].isna().sum() == 0

    def test_crs_is_geographic(self, pipeline_mocks, minimal_config):
        """Output must be in EPSG:4326."""
        result = build_pipeline(minimal_config)
        assert result.crs.to_epsg() == 4326
