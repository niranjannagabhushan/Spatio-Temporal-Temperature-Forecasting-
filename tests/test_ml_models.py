"""
tests/test_ml_models.py
-----------------------
Unit tests for src.models.ml_models:
  build_preprocessor, prepare_features, all 6 individual trainers,
  and run_all_models.

All tests use the synthetic_station_gdf + minimal_config fixtures from
conftest.py so no real data files are needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from src.models.ml_models import (
    build_preprocessor,
    prepare_features,
    train_ridge,
    train_lasso,
    train_svr,
    train_random_forest,
    train_lightgbm,
    train_xgboost,
    run_all_models,
)


# ---------------------------------------------------------------------------
# Shared result-dict checker
# ---------------------------------------------------------------------------

RESULT_KEYS = {"model", "mse", "mae", "r2"}

def _assert_valid_result(result: dict, model_name: str) -> None:
    assert isinstance(result, dict),        "result must be a dict"
    assert set(result.keys()) == RESULT_KEYS
    assert result["model"] == model_name
    assert result["mse"]   >= 0.0,          "MSE must be non-negative"
    assert result["mae"]   >= 0.0,          "MAE must be non-negative"
    assert isinstance(result["r2"], float), "R² must be float"


# ---------------------------------------------------------------------------
# build_preprocessor
# ---------------------------------------------------------------------------

class TestBuildPreprocessor:
    def test_numeric_only_has_scaler(self):
        X = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        pre = build_preprocessor(X)
        names = [name for name, _, _ in pre.transformers]
        assert "num" in names

    def test_categorical_only_has_encoder(self):
        X = pd.DataFrame({"cat": ["a", "b"], "other": ["x", "y"]})
        pre = build_preprocessor(X)
        names = [name for name, _, _ in pre.transformers]
        assert "cat" in names

    def test_mixed_has_both(self):
        X = pd.DataFrame({"num": [1.0, 2.0], "cat": ["a", "b"]})
        pre = build_preprocessor(X)
        names = [name for name, _, _ in pre.transformers]
        assert "num" in names
        assert "cat" in names

    def test_returns_column_transformer(self):
        X = pd.DataFrame({"a": [1.0, 2.0]})
        pre = build_preprocessor(X)
        assert isinstance(pre, ColumnTransformer)

    def test_fit_transform_works(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        pre = build_preprocessor(X)
        transformed = pre.fit_transform(X)
        assert transformed.shape == (3, 2)

    def test_numeric_scaler_is_standard(self):
        X = pd.DataFrame({"val": [1.0, 2.0, 3.0]})
        pre = build_preprocessor(X)
        num_transformers = [t for name, t, _ in pre.transformers if name == "num"]
        assert len(num_transformers) == 1
        assert isinstance(num_transformers[0], StandardScaler)


# ---------------------------------------------------------------------------
# prepare_features
# ---------------------------------------------------------------------------

class TestPrepareFeatures:
    def test_returns_four_objects(self, synthetic_station_gdf, minimal_config):
        result = prepare_features(synthetic_station_gdf, minimal_config)
        assert len(result) == 4

    def test_train_test_size_80_20(self, synthetic_station_gdf, minimal_config):
        X_train, X_test, y_train, y_test = prepare_features(
            synthetic_station_gdf, minimal_config
        )
        n_total = len(X_train) + len(X_test)
        assert len(X_test)  == pytest.approx(n_total * 0.20, abs=2)
        assert len(X_train) == pytest.approx(n_total * 0.80, abs=2)

    def test_X_train_does_not_contain_target(self, synthetic_station_gdf, minimal_config):
        X_train, _, _, _ = prepare_features(synthetic_station_gdf, minimal_config)
        assert "temperature" not in X_train.columns

    def test_no_index_overlap_between_train_and_test(
        self, synthetic_station_gdf, minimal_config
    ):
        X_train, X_test, _, _ = prepare_features(synthetic_station_gdf, minimal_config)
        shared = set(X_train.index) & set(X_test.index)
        assert len(shared) == 0, "Train and test indices must not overlap"

    def test_y_series_aligned_with_X(self, synthetic_station_gdf, minimal_config):
        X_train, X_test, y_train, y_test = prepare_features(
            synthetic_station_gdf, minimal_config
        )
        assert set(X_train.index) == set(y_train.index)
        assert set(X_test.index)  == set(y_test.index)

    def test_no_null_values_in_numeric_X(self, synthetic_station_gdf, minimal_config):
        X_train, X_test, _, _ = prepare_features(synthetic_station_gdf, minimal_config)
        num_cols_tr = X_train.select_dtypes(include="number").columns
        num_cols_te = X_test.select_dtypes(include="number").columns
        assert X_train[num_cols_tr].isna().sum().sum() == 0
        assert X_test[num_cols_te].isna().sum().sum()  == 0


# ---------------------------------------------------------------------------
# Individual trainers — shared pattern
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def split_data(synthetic_station_gdf, minimal_config):
    """Pre-computed train/test split reused by all trainer tests."""
    return prepare_features(synthetic_station_gdf, minimal_config)


class TestTrainRidge:
    def test_result_keys_and_types(self, split_data, minimal_config):
        X_train, X_test, y_train, y_test = split_data
        result = train_ridge(X_train, y_train, X_test, y_test, minimal_config)
        _assert_valid_result(result, "Ridge")

    def test_r2_is_float(self, split_data, minimal_config):
        X_train, X_test, y_train, y_test = split_data
        result = train_ridge(X_train, y_train, X_test, y_test, minimal_config)
        assert isinstance(result["r2"], float)


class TestTrainLasso:
    def test_result_keys_and_types(self, split_data, minimal_config):
        X_train, X_test, y_train, y_test = split_data
        result = train_lasso(X_train, y_train, X_test, y_test, minimal_config)
        _assert_valid_result(result, "Lasso")


class TestTrainSVR:
    def test_result_keys_and_types(self, split_data, minimal_config):
        X_train, X_test, y_train, y_test = split_data
        result = train_svr(X_train, y_train, X_test, y_test, minimal_config)
        _assert_valid_result(result, "SVR")


class TestTrainRandomForest:
    def test_result_keys_and_types(self, split_data, minimal_config):
        X_train, X_test, y_train, y_test = split_data
        result = train_random_forest(X_train, y_train, X_test, y_test, minimal_config)
        _assert_valid_result(result, "RandomForest")

    def test_mse_non_negative(self, split_data, minimal_config):
        X_train, X_test, y_train, y_test = split_data
        result = train_random_forest(X_train, y_train, X_test, y_test, minimal_config)
        assert result["mse"] >= 0.0


class TestTrainLightGBM:
    def test_result_keys_and_types(self, split_data, minimal_config):
        X_train, X_test, y_train, y_test = split_data
        result = train_lightgbm(X_train, y_train, X_test, y_test, minimal_config)
        _assert_valid_result(result, "LightGBM")


class TestTrainXGBoost:
    def test_result_keys_and_types(self, split_data, minimal_config):
        X_train, X_test, y_train, y_test = split_data
        result = train_xgboost(X_train, y_train, X_test, y_test, minimal_config)
        _assert_valid_result(result, "XGBoost")


# ---------------------------------------------------------------------------
# run_all_models
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def all_models_results(synthetic_station_gdf, minimal_config):
    """Run all 6 ML models once and cache the result for the whole module."""
    return run_all_models(synthetic_station_gdf, minimal_config)


class TestRunAllModels:
    def test_returns_list(self, all_models_results):
        assert isinstance(all_models_results, list)

    def test_six_results(self, all_models_results):
        assert len(all_models_results) == 6

    def test_all_results_have_correct_keys(self, all_models_results):
        for r in all_models_results:
            assert set(r.keys()) == RESULT_KEYS

    def test_model_names_are_distinct(self, all_models_results):
        names = [r["model"] for r in all_models_results]
        assert len(names) == len(set(names)), "All model names must be unique"

    def test_expected_model_names_present(self, all_models_results):
        names_set = {r["model"] for r in all_models_results}
        expected  = {"Ridge", "Lasso", "SVR", "RandomForest", "LightGBM", "XGBoost"}
        assert names_set == expected
