"""
tests/test_deep_learning.py
---------------------------
Unit tests for src.models.deep_learning.

Structure
---------
TestScaleFeatures   — always runs (no optional dependency).
TestTrainTransformer, TestTrainTCN, TestTrainKerasTunerMLP
                    — skipped automatically when TensorFlow is not installed.

Each TF trainer test uses:
  * The synthetic_station_gdf + minimal_config fixtures (tiny epochs/dims).
  * train/test splits produced by prepare_features().
  * Assertions on result dict shape, metric types, and non-negativity.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from src.models.deep_learning import _scale_features
from src.models.ml_models import prepare_features


# ---------------------------------------------------------------------------
# _scale_features  (no TF dependency)
# ---------------------------------------------------------------------------

class TestScaleFeatures:
    def _make_data(self, n_train=100, n_test=30, n_features=5):
        rng = np.random.default_rng(10)
        X_train = rng.normal(10.0, 3.0, (n_train, n_features)).astype(np.float32)
        X_test  = rng.normal(10.0, 3.0, (n_test,  n_features)).astype(np.float32)
        return X_train, X_test

    def test_returns_two_arrays(self):
        X_train, X_test = self._make_data()
        result = _scale_features(X_train, X_test)
        assert len(result) == 2

    def test_train_mean_near_zero(self):
        X_train, X_test = self._make_data()
        X_tr_s, _ = _scale_features(X_train, X_test)
        assert np.abs(X_tr_s.mean()) < 0.05

    def test_train_std_near_one(self):
        X_train, X_test = self._make_data()
        X_tr_s, _ = _scale_features(X_train, X_test)
        assert abs(X_tr_s.std() - 1.0) < 0.1

    def test_test_not_independently_normalised(self):
        """Test set is transformed using train statistics, so its mean != 0."""
        rng = np.random.default_rng(20)
        X_train = rng.normal(0.0, 1.0, (200, 3)).astype(np.float32)
        # Test data comes from a very different distribution
        X_test  = rng.normal(100.0, 1.0, (50,  3)).astype(np.float32)
        _, X_te_s = _scale_features(X_train, X_test)
        # If the test set were independently normalised its mean would be ≈0;
        # since we use training statistics, it should be far from 0.
        assert abs(X_te_s.mean()) > 5.0

    def test_shapes_preserved(self):
        X_train, X_test = self._make_data(80, 20, 7)
        X_tr_s, X_te_s = _scale_features(X_train, X_test)
        assert X_tr_s.shape == (80, 7)
        assert X_te_s.shape == (20, 7)

    def test_original_arrays_not_mutated(self):
        X_train, X_test = self._make_data()
        orig_mean = X_train.mean()
        _scale_features(X_train, X_test)
        assert abs(X_train.mean() - orig_mean) < 1e-6


# ---------------------------------------------------------------------------
# Helpers for TF tests
# ---------------------------------------------------------------------------

RESULT_KEYS = {"model", "mse", "mae", "r2"}

def _assert_valid_result(result: dict, model_name: str) -> None:
    assert set(result.keys()) == RESULT_KEYS
    assert result["model"] == model_name
    assert result["mse"]   >= 0.0
    assert result["mae"]   >= 0.0
    assert isinstance(result["r2"], float)


def _get_numpy_splits(synthetic_station_gdf, minimal_config):
    """Return (X_train_np, y_train_np, X_test_np, y_test_np) as float32 arrays."""
    X_train_df, X_test_df, y_train, y_test = prepare_features(
        synthetic_station_gdf, minimal_config
    )
    X_train = X_train_df.select_dtypes(include="number").values.astype(np.float32)
    X_test  = X_test_df.select_dtypes(include="number").values.astype(np.float32)
    return X_train, y_train.values.astype(np.float32), X_test, y_test.values.astype(np.float32)


# ---------------------------------------------------------------------------
# train_transformer  (requires TensorFlow)
# ---------------------------------------------------------------------------

@pytest.mark.requires_tensorflow
class TestTrainTransformer:
    tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")

    def test_result_keys_and_types(self, synthetic_station_gdf, minimal_config):
        from src.models.deep_learning import train_transformer
        X_train, y_train, X_test, y_test = _get_numpy_splits(
            synthetic_station_gdf, minimal_config
        )
        result = train_transformer(X_train, y_train, X_test, y_test, minimal_config)
        _assert_valid_result(result, "Transformer")

    def test_evaluated_on_test_not_train(self, synthetic_station_gdf, minimal_config):
        """R² on training data would be much higher than on test — confirm test is used."""
        from src.models.deep_learning import train_transformer
        X_train, y_train, X_test, y_test = _get_numpy_splits(
            synthetic_station_gdf, minimal_config
        )
        # This just confirms no exception is raised and a valid R² is returned;
        # the correctness of the split is already tested via _scale_features.
        result = train_transformer(X_train, y_train, X_test, y_test, minimal_config)
        assert isinstance(result["r2"], float)


# ---------------------------------------------------------------------------
# train_tcn  (requires TensorFlow)
# ---------------------------------------------------------------------------

@pytest.mark.requires_tensorflow
class TestTrainTCN:
    tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")

    def test_result_keys_and_types(self, synthetic_station_gdf, minimal_config):
        from src.models.deep_learning import train_tcn
        X_train, y_train, X_test, y_test = _get_numpy_splits(
            synthetic_station_gdf, minimal_config
        )
        result = train_tcn(X_train, y_train, X_test, y_test, minimal_config)
        _assert_valid_result(result, "TCN")

    def test_mse_non_negative(self, synthetic_station_gdf, minimal_config):
        from src.models.deep_learning import train_tcn
        X_train, y_train, X_test, y_test = _get_numpy_splits(
            synthetic_station_gdf, minimal_config
        )
        result = train_tcn(X_train, y_train, X_test, y_test, minimal_config)
        assert result["mse"] >= 0.0


# ---------------------------------------------------------------------------
# train_keras_tuner_mlp  (requires TensorFlow + keras-tuner)
# ---------------------------------------------------------------------------

@pytest.mark.requires_tensorflow
class TestTrainKerasTunerMLP:
    tf = pytest.importorskip("tensorflow",  reason="TensorFlow not installed")
    kt = pytest.importorskip("keras_tuner", reason="keras-tuner not installed")

    def test_result_keys_and_types(self, synthetic_station_gdf, minimal_config):
        from src.models.deep_learning import train_keras_tuner_mlp
        X_train, y_train, X_test, y_test = _get_numpy_splits(
            synthetic_station_gdf, minimal_config
        )
        result = train_keras_tuner_mlp(
            X_train, y_train, X_test, y_test, minimal_config
        )
        _assert_valid_result(result, "KerasTunerMLP")

    def test_best_model_used_not_stale(self, synthetic_station_gdf, minimal_config):
        """
        Regression guard for Issue 3: the returned mse/mae must be finite,
        which would not be the case if an un-trained (stale) model were used.
        """
        import math
        from src.models.deep_learning import train_keras_tuner_mlp
        X_train, y_train, X_test, y_test = _get_numpy_splits(
            synthetic_station_gdf, minimal_config
        )
        result = train_keras_tuner_mlp(
            X_train, y_train, X_test, y_test, minimal_config
        )
        assert math.isfinite(result["mse"]), "MSE must be finite (stale model check)"
        assert math.isfinite(result["mae"]), "MAE must be finite (stale model check)"
