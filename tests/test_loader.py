"""
tests/test_loader.py
--------------------
Unit tests for src.data.loader:
  load_config, load_weather_data, load_station_metadata, load_shapefiles.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.loader import (
    load_config,
    load_weather_data,
    load_station_metadata,
    load_shapefiles,
)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_returns_dict(self, tmp_path: Path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("key: value\nnested:\n  a: 1\n")
        result = load_config(str(cfg_file))
        assert isinstance(result, dict)

    def test_values_parsed(self, tmp_path: Path):
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text("alpha: 1.5\nn_estimators: 100\nlabel: hello\n")
        cfg = load_config(str(cfg_file))
        assert cfg["alpha"] == pytest.approx(1.5)
        assert cfg["n_estimators"] == 100
        assert cfg["label"] == "hello"

    def test_nested_dict(self, tmp_path: Path):
        content = textwrap.dedent("""\
            models:
              ridge:
                alpha: 10
              random_forest:
                n_estimators: 50
        """)
        cfg_file = tmp_path / "config.yaml"
        cfg_file.write_text(content)
        cfg = load_config(str(cfg_file))
        assert cfg["models"]["ridge"]["alpha"] == 10
        assert cfg["models"]["random_forest"]["n_estimators"] == 50

    def test_file_not_found_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.yaml"))


# ---------------------------------------------------------------------------
# load_weather_data
# ---------------------------------------------------------------------------

class TestLoadWeatherData:
    def _write_csv(self, tmp_path: Path, content: str) -> str:
        p = tmp_path / "weather.csv"
        p.write_text(content)
        return str(p)

    def test_returns_dataframe(self, tmp_path):
        path = self._write_csv(
            tmp_path,
            "station_id,temperature,timestamp\n1,14.5,2024-01-01 00:00:00\n",
        )
        df = load_weather_data(path)
        assert isinstance(df, pd.DataFrame)

    def test_columns_present(self, tmp_path):
        path = self._write_csv(
            tmp_path,
            "station_id,temperature,timestamp\n1,14.5,2024-01-01 00:00:00\n",
        )
        df = load_weather_data(path)
        assert "station_id" in df.columns
        assert "temperature" in df.columns

    def test_timestamp_parsed_to_datetime(self, tmp_path):
        path = self._write_csv(
            tmp_path,
            "station_id,temperature,timestamp\n1,14.5,2024-06-15 12:30:00\n",
        )
        df = load_weather_data(path)
        assert pd.api.types.is_datetime64_any_dtype(df["timestamp"])

    def test_no_timestamp_column_ok(self, tmp_path):
        """File without a timestamp column should load without error."""
        path = self._write_csv(tmp_path, "station_id,temperature\n1,14.5\n2,16.2\n")
        df = load_weather_data(path)
        assert "timestamp" not in df.columns
        assert len(df) == 2

    def test_multiple_rows_loaded(self, tmp_path):
        rows = "\n".join(
            f"{i},{10.0 + i},2024-01-{i+1:02d} 00:00:00" for i in range(5)
        )
        path = self._write_csv(tmp_path, f"station_id,temperature,timestamp\n{rows}\n")
        df = load_weather_data(path)
        assert len(df) == 5


# ---------------------------------------------------------------------------
# load_station_metadata
# ---------------------------------------------------------------------------

class TestLoadStationMetadata:
    def test_returns_dataframe(self, tmp_path):
        p = tmp_path / "stations.csv"
        p.write_text("station_id,altitude,latitude,longitude\n1,800,45.7,7.3\n")
        df = load_station_metadata(str(p))
        assert isinstance(df, pd.DataFrame)

    def test_station_id_column_preserved(self, tmp_path):
        p = tmp_path / "stations.csv"
        p.write_text("station_id,altitude\n1,800\n2,1200\n")
        df = load_station_metadata(str(p))
        assert "station_id" in df.columns

    def test_id_column_renamed_to_station_id(self, tmp_path):
        """If the CSV uses 'id' instead of 'station_id', it must be renamed."""
        p = tmp_path / "stations.csv"
        p.write_text("id,altitude\n1,800\n2,1200\n")
        df = load_station_metadata(str(p))
        assert "station_id" in df.columns
        assert "id" not in df.columns

    def test_id_not_renamed_when_station_id_exists(self, tmp_path):
        """If both 'id' and 'station_id' exist, neither is renamed."""
        p = tmp_path / "stations.csv"
        p.write_text("id,station_id,altitude\n10,1,800\n")
        df = load_station_metadata(str(p))
        assert "station_id" in df.columns
        assert "id" in df.columns  # 'id' kept as-is

    def test_row_count_preserved(self, tmp_path):
        rows = "\n".join(f"{i},{500 + i * 100}" for i in range(10))
        p = tmp_path / "stations.csv"
        p.write_text(f"station_id,altitude\n{rows}\n")
        df = load_station_metadata(str(p))
        assert len(df) == 10


# ---------------------------------------------------------------------------
# load_shapefiles
# ---------------------------------------------------------------------------

class TestLoadShapefiles:
    @patch("src.data.loader.gpd.read_file")
    def test_returns_two_geodataframes(self, mock_read_file):
        import geopandas as gpd
        mock_lakes  = MagicMock(spec=gpd.GeoDataFrame)
        mock_rivers = MagicMock(spec=gpd.GeoDataFrame)
        mock_read_file.side_effect = [mock_lakes, mock_rivers]

        lakes, rivers = load_shapefiles("lakes.shp", "rivers.shp")

        assert lakes  is mock_lakes
        assert rivers is mock_rivers

    @patch("src.data.loader.gpd.read_file")
    def test_called_with_correct_paths(self, mock_read_file):
        import geopandas as gpd
        mock_read_file.return_value = MagicMock(spec=gpd.GeoDataFrame)

        load_shapefiles("/data/lakes.shp", "/data/rivers.shp")

        calls = [call[0][0] for call in mock_read_file.call_args_list]
        assert "/data/lakes.shp"  in calls
        assert "/data/rivers.shp" in calls
