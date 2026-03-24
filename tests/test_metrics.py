"""
tests/test_metrics.py
---------------------
Unit tests for src.utils.metrics.evaluate_model().
"""

import math
import pytest
import numpy as np

from src.utils.metrics import evaluate_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perfect(n: int = 50):
    """Ground truth and perfect predictions."""
    y = np.arange(n, dtype=float)
    return y, y.copy()


def _constant_pred(n: int = 50):
    """Ground truth varies; prediction is the constant mean → R² ≈ 0."""
    rng = np.random.default_rng(1)
    y_true = rng.normal(10.0, 3.0, n)
    y_pred = np.full(n, y_true.mean())
    return y_true, y_pred


def _bad_pred(n: int = 50):
    """Predictions worse than the mean → R² < 0."""
    rng = np.random.default_rng(2)
    y_true = rng.normal(10.0, 3.0, n)
    # Predict the mirror image around 0 — far from the truth
    y_pred = -y_true + 100.0
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Return-value structure
# ---------------------------------------------------------------------------

class TestReturnStructure:
    def test_returns_dict(self):
        y_true, y_pred = _perfect()
        result = evaluate_model(y_true, y_pred)
        assert isinstance(result, dict)

    def test_dict_has_expected_keys(self):
        y_true, y_pred = _perfect()
        result = evaluate_model(y_true, y_pred, model_name="TestModel")
        assert set(result.keys()) == {"model", "mse", "mae", "r2"}

    def test_model_name_stored(self):
        y_true, y_pred = _perfect()
        result = evaluate_model(y_true, y_pred, model_name="Ridge")
        assert result["model"] == "Ridge"

    def test_model_name_empty_by_default(self):
        y_true, y_pred = _perfect()
        result = evaluate_model(y_true, y_pred)
        assert result["model"] == ""

    def test_metric_values_are_floats(self):
        y_true, y_pred = _perfect()
        result = evaluate_model(y_true, y_pred)
        for key in ("mse", "mae", "r2"):
            assert isinstance(result[key], float), f"{key} should be float"


# ---------------------------------------------------------------------------
# Metric correctness
# ---------------------------------------------------------------------------

class TestMetricValues:
    def test_perfect_mse_is_zero(self):
        y_true, y_pred = _perfect()
        assert evaluate_model(y_true, y_pred)["mse"] == pytest.approx(0.0, abs=1e-9)

    def test_perfect_mae_is_zero(self):
        y_true, y_pred = _perfect()
        assert evaluate_model(y_true, y_pred)["mae"] == pytest.approx(0.0, abs=1e-9)

    def test_perfect_r2_is_one(self):
        y_true, y_pred = _perfect()
        assert evaluate_model(y_true, y_pred)["r2"] == pytest.approx(1.0, abs=1e-9)

    def test_mse_is_non_negative(self):
        y_true, y_pred = _constant_pred()
        assert evaluate_model(y_true, y_pred)["mse"] >= 0.0

    def test_mae_is_non_negative(self):
        y_true, y_pred = _constant_pred()
        assert evaluate_model(y_true, y_pred)["mae"] >= 0.0

    def test_constant_pred_r2_near_zero(self):
        """Predicting the mean → R² should be ≈ 0 (within float tolerance)."""
        y_true, y_pred = _constant_pred(200)
        r2 = evaluate_model(y_true, y_pred)["r2"]
        assert abs(r2) < 0.01

    def test_bad_pred_r2_is_negative(self):
        y_true, y_pred = _bad_pred()
        assert evaluate_model(y_true, y_pred)["r2"] < 0.0

    def test_known_mse(self):
        """MSE([0,1,2], [1,2,3]) == 1.0."""
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert evaluate_model(y_true, y_pred)["mse"] == pytest.approx(1.0)

    def test_known_mae(self):
        """MAE([0,1,2], [1,2,3]) == 1.0."""
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert evaluate_model(y_true, y_pred)["mae"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Print behaviour
# ---------------------------------------------------------------------------

class TestPrintOutput:
    def test_model_name_triggers_print(self, capsys):
        evaluate_model(np.array([1.0]), np.array([1.0]), model_name="Foo")
        captured = capsys.readouterr()
        assert "Foo" in captured.out

    def test_no_model_name_no_print(self, capsys):
        evaluate_model(np.array([1.0]), np.array([1.0]))
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_print_contains_metric_labels(self, capsys):
        evaluate_model(np.array([1.0, 2.0]), np.array([1.0, 2.0]), model_name="X")
        captured = capsys.readouterr()
        for label in ("MSE", "MAE", "R²"):
            assert label in captured.out
