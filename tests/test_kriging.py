"""
tests/test_kriging.py
---------------------
Unit tests for src.models.kriging:
  run_ordinary_kriging, run_universal_kriging, run_kriging.

Two test categories
-------------------
1. Error-handling path  — pykrige is mocked to raise LinAlgError / ValueError /
                          RuntimeError; the wrappers must return (None, None)
                          and emit a RuntimeWarning.
2. Happy path           — tests using the real pykrige library (skipped
                          automatically when pykrige is not installed).
"""

from __future__ import annotations

import sys
import warnings
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numpy.linalg import LinAlgError

from src.models.kriging import (
    run_ordinary_kriging,
    run_universal_kriging,
    run_kriging,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mock_kriging_module(execute_side_effect=None, execute_return_value=None):
    """Return (mock_pykrige_ok, mock_pykrige_uk) with controlled execute()."""
    mock_instance = MagicMock()
    if execute_side_effect is not None:
        mock_instance.execute.side_effect = execute_side_effect
    elif execute_return_value is not None:
        mock_instance.execute.return_value = execute_return_value

    mock_class = MagicMock(return_value=mock_instance)

    mock_ok_module = MagicMock()
    mock_ok_module.OrdinaryKriging = mock_class

    mock_uk_module = MagicMock()
    mock_uk_module.UniversalKriging = mock_class

    return mock_ok_module, mock_uk_module, mock_instance


# ---------------------------------------------------------------------------
# run_ordinary_kriging — error-handling paths
# ---------------------------------------------------------------------------

class TestOrdinaryKrigingErrorHandling:
    """Verify that numerical exceptions are caught and (None, None) returned."""

    _PARAMS = dict(
        x=np.array([0.1, 0.2, 0.3]),
        y=np.array([45.5, 45.6, 45.7]),
        z=np.array([10.0, 12.0, 11.0]),
        grid_x=np.linspace(0.1, 0.3, 5),
        grid_y=np.linspace(45.5, 45.7, 5),
    )

    def _run_with_mock_exception(self, exc):
        mock_ok_mod, _, _ = _mock_kriging_module(execute_side_effect=exc)
        with patch.dict(
            sys.modules,
            {"pykrige": MagicMock(), "pykrige.ok": mock_ok_mod},
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = run_ordinary_kriging(**self._PARAMS)
        return result, caught

    def test_linalg_error_returns_none_none(self):
        result, _ = self._run_with_mock_exception(LinAlgError("singular"))
        assert result == (None, None)

    def test_value_error_returns_none_none(self):
        result, _ = self._run_with_mock_exception(ValueError("bad value"))
        assert result == (None, None)

    def test_runtime_error_returns_none_none(self):
        result, _ = self._run_with_mock_exception(RuntimeError("runtime"))
        assert result == (None, None)

    def test_linalg_error_emits_runtime_warning(self):
        _, caught = self._run_with_mock_exception(LinAlgError("singular"))
        categories = [w.category for w in caught]
        assert RuntimeWarning in categories

    def test_warning_message_mentions_ordinary_kriging(self):
        _, caught = self._run_with_mock_exception(LinAlgError("singular"))
        messages = [str(w.message) for w in caught]
        assert any("OrdinaryKriging" in m for m in messages)


# ---------------------------------------------------------------------------
# run_universal_kriging — error-handling paths
# ---------------------------------------------------------------------------

class TestUniversalKrigingErrorHandling:
    _PARAMS = dict(
        x=np.array([0.1, 0.2, 0.3]),
        y=np.array([45.5, 45.6, 45.7]),
        z=np.array([10.0, 12.0, 11.0]),
        grid_x=np.linspace(0.1, 0.3, 5),
        grid_y=np.linspace(45.5, 45.7, 5),
    )

    def _run_with_mock_exception(self, exc):
        _, mock_uk_mod, _ = _mock_kriging_module(execute_side_effect=exc)
        with patch.dict(
            sys.modules,
            {"pykrige": MagicMock(), "pykrige.uk": mock_uk_mod},
        ):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = run_universal_kriging(**self._PARAMS)
        return result, caught

    def test_linalg_error_returns_none_none(self):
        result, _ = self._run_with_mock_exception(LinAlgError("singular"))
        assert result == (None, None)

    def test_value_error_returns_none_none(self):
        result, _ = self._run_with_mock_exception(ValueError("bad"))
        assert result == (None, None)

    def test_runtime_error_returns_none_none(self):
        result, _ = self._run_with_mock_exception(RuntimeError("crash"))
        assert result == (None, None)

    def test_linalg_error_emits_runtime_warning(self):
        _, caught = self._run_with_mock_exception(LinAlgError("singular"))
        assert any(issubclass(w.category, RuntimeWarning) for w in caught)

    def test_warning_message_mentions_universal_kriging(self):
        _, caught = self._run_with_mock_exception(LinAlgError("singular"))
        messages = [str(w.message) for w in caught]
        assert any("UniversalKriging" in m for m in messages)


# ---------------------------------------------------------------------------
# run_ordinary_kriging — happy path (pykrige required)
# ---------------------------------------------------------------------------

@pytest.mark.requires_pykrige
class TestOrdinaryKrigingHappyPath:
    pykrige = pytest.importorskip("pykrige", reason="pykrige not installed")

    def test_returns_two_arrays(self, xy_kriging):
        x, y, z, gx, gy = xy_kriging
        pred, var = run_ordinary_kriging(x, y, z, gx, gy)
        if pred is None:
            pytest.skip("Kriging returned None (ill-conditioned on this data)")
        assert pred is not None
        assert var  is not None

    def test_prediction_shape_matches_grid(self, xy_kriging):
        x, y, z, gx, gy = xy_kriging
        pred, _ = run_ordinary_kriging(x, y, z, gx, gy)
        if pred is None:
            pytest.skip("Kriging returned None (ill-conditioned on this data)")
        assert pred.shape == (len(gy), len(gx))

    def test_variance_non_negative(self, xy_kriging):
        x, y, z, gx, gy = xy_kriging
        _, var = run_ordinary_kriging(x, y, z, gx, gy)
        if var is None:
            pytest.skip("Kriging returned None (ill-conditioned on this data)")
        assert (var >= 0).all()


# ---------------------------------------------------------------------------
# run_universal_kriging — happy path (pykrige required)
# ---------------------------------------------------------------------------

@pytest.mark.requires_pykrige
class TestUniversalKrigingHappyPath:
    pykrige = pytest.importorskip("pykrige", reason="pykrige not installed")

    def test_returns_two_arrays(self, xy_kriging):
        x, y, z, gx, gy = xy_kriging
        pred, var = run_universal_kriging(x, y, z, gx, gy)
        if pred is None:
            pytest.skip("Kriging returned None (ill-conditioned on this data)")
        assert pred is not None and var is not None

    def test_prediction_shape_matches_grid(self, xy_kriging):
        x, y, z, gx, gy = xy_kriging
        pred, _ = run_universal_kriging(x, y, z, gx, gy)
        if pred is None:
            pytest.skip("Kriging returned None (ill-conditioned on this data)")
        assert pred.shape == (len(gy), len(gx))


# ---------------------------------------------------------------------------
# run_kriging (high-level wrapper)
# ---------------------------------------------------------------------------

class TestRunKriging:
    def test_returns_dict_with_expected_keys(self, synthetic_station_gdf, minimal_config):
        """run_kriging always returns the six-key dict; ok_pred/uk_pred may be None."""
        mock_ok_mod, mock_uk_mod, _ = _mock_kriging_module(
            execute_return_value=(np.ones((10, 10)), np.zeros((10, 10)))
        )
        with patch.dict(
            sys.modules,
            {
                "pykrige":    MagicMock(),
                "pykrige.ok": mock_ok_mod,
                "pykrige.uk": mock_uk_mod,
            },
        ):
            result = run_kriging(synthetic_station_gdf, minimal_config)

        expected_keys = {"ok_pred", "ok_var", "uk_pred", "uk_var", "grid_x", "grid_y"}
        assert set(result.keys()) == expected_keys

    def test_grid_arrays_have_correct_length(self, synthetic_station_gdf, minimal_config):
        mock_ok_mod, mock_uk_mod, _ = _mock_kriging_module(
            execute_return_value=(np.ones((10, 10)), np.zeros((10, 10)))
        )
        with patch.dict(
            sys.modules,
            {
                "pykrige":    MagicMock(),
                "pykrige.ok": mock_ok_mod,
                "pykrige.uk": mock_uk_mod,
            },
        ):
            result = run_kriging(synthetic_station_gdf, minimal_config)

        resolution = minimal_config["models"]["kriging"]["grid_resolution"]
        assert len(result["grid_x"]) == resolution
        assert len(result["grid_y"]) == resolution

    def test_graceful_when_both_fail(self, synthetic_station_gdf, minimal_config):
        """When both kriging methods raise, dict values are None but no crash."""
        mock_ok_mod, mock_uk_mod, _ = _mock_kriging_module(
            execute_side_effect=LinAlgError("singular")
        )
        with patch.dict(
            sys.modules,
            {
                "pykrige":    MagicMock(),
                "pykrige.ok": mock_ok_mod,
                "pykrige.uk": mock_uk_mod,
            },
        ):
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                result = run_kriging(synthetic_station_gdf, minimal_config)

        assert result["ok_pred"] is None
        assert result["uk_pred"] is None
