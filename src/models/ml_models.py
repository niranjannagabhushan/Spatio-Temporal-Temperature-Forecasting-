"""
src/models/ml_models.py
-----------------------
Train and evaluate a suite of regression models on the preprocessed
geospatial weather dataset.

Public API
----------
run_all_models(station_gdf, config)  →  list[dict]
"""

from __future__ import annotations

from typing import Any, Dict, List

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Feature specification
# ---------------------------------------------------------------------------

_FEATURE_COLS = [
    "altitude",
    "latitude",
    "longitude",
    "distance_to_lake",
    "distance_to_river",
    "hour",
    "month",
]
_TARGET_COL = "temperature"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prepare_data(
    gdf: gpd.GeoDataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Return (X_train, X_test, y_train, y_test) from the GeoDataFrame."""
    available_features = [c for c in _FEATURE_COLS if c in gdf.columns]
    df = gdf[available_features + [_TARGET_COL]].dropna()

    X = df[available_features].values
    y = df[_TARGET_COL].values

    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def _evaluate(name: str, model: Any, X_test, y_test) -> Dict[str, Any]:
    """Return a results dict for one fitted model."""
    y_pred = model.predict(X_test)
    return {
        "model": name,
        "mse": float(mean_squared_error(y_test, y_pred)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }


def _build_models(config: dict) -> List[tuple]:
    """Instantiate all models.  Wraps linear models in a scaling pipeline."""
    models_cfg = config.get("models", {})

    # Optional hyperparameter overrides from config
    rf_n = models_cfg.get("random_forest", {}).get("n_estimators", 200)
    rf_depth = models_cfg.get("random_forest", {}).get("max_depth", None)
    ridge_alpha = models_cfg.get("ridge", {}).get("alpha", 1.0)
    lasso_alpha = models_cfg.get("lasso", {}).get("alpha", 0.01)

    estimators = [
        (
            "Ridge",
            Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=ridge_alpha))]),
        ),
        (
            "Lasso",
            Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=lasso_alpha, max_iter=5000))]),
        ),
        (
            "SVR",
            Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=10, epsilon=0.1))]),
        ),
        (
            "RandomForest",
            RandomForestRegressor(
                n_estimators=rf_n,
                max_depth=rf_depth,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]

    # Optional heavy-weight boosters — imported lazily so the module loads
    # even if the packages are absent.
    try:
        import lightgbm as lgb  # noqa: F401

        estimators.append(
            (
                "LightGBM",
                lgb.LGBMRegressor(
                    n_estimators=models_cfg.get("lightgbm", {}).get("n_estimators", 300),
                    learning_rate=models_cfg.get("lightgbm", {}).get("learning_rate", 0.05),
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                ),
            )
        )
    except ImportError:
        pass

    try:
        import xgboost as xgb  # noqa: F401

        estimators.append(
            (
                "XGBoost",
                xgb.XGBRegressor(
                    n_estimators=models_cfg.get("xgboost", {}).get("n_estimators", 300),
                    learning_rate=models_cfg.get("xgboost", {}).get("learning_rate", 0.05),
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                ),
            )
        )
    except ImportError:
        pass

    return estimators


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_all_models(
    station_gdf: gpd.GeoDataFrame,
    config: dict,
) -> List[Dict[str, Any]]:
    """Train and evaluate all configured regression models.

    Parameters
    ----------
    station_gdf:
        Output of :func:`src.data.preprocessor.build_pipeline`.
    config:
        Dictionary loaded from ``config.yaml``.

    Returns
    -------
    list of dict
        Each element has keys: ``model``, ``mse``, ``mae``, ``r2``.
    """
    X_train, X_test, y_train, y_test = _prepare_data(station_gdf)

    results: List[Dict[str, Any]] = []
    for name, estimator in _build_models(config):
        print(f"  Training {name} …", end=" ", flush=True)
        estimator.fit(X_train, y_train)
        metrics = _evaluate(name, estimator, X_test, y_test)
        results.append(metrics)
        print(f"R² = {metrics['r2']:.4f}")

    return results
