"""
tests/test_imports.py
---------------------
Compilation smoke-tests: verify that every public ``src`` module can be
imported cleanly without raising any exception.

These are the project's equivalent of a "compilation check" — if any
module has a top-level syntax error, circular import, or missing hard
dependency, these tests will catch it immediately.
"""

import importlib
import pytest


# ---------------------------------------------------------------------------
# Modules that must always be importable (no optional dependencies needed)
# ---------------------------------------------------------------------------

ALWAYS_AVAILABLE = [
    "src",
    "src.utils",
    "src.utils.metrics",
    "src.data",
    "src.data.loader",
    "src.data.preprocessor",
    "src.models",
    "src.models.ml_models",
    "src.models.kriging",
    "src.models.deep_learning",
    "src.models.gcn",
]


@pytest.mark.parametrize("module_path", ALWAYS_AVAILABLE)
def test_module_imports_cleanly(module_path: str) -> None:
    """Each core module must import without raising any exception."""
    mod = importlib.import_module(module_path)
    assert mod is not None, f"importlib returned None for {module_path}"


# ---------------------------------------------------------------------------
# Public API surface checks
# ---------------------------------------------------------------------------

def test_metrics_public_api() -> None:
    """src.utils.metrics exposes evaluate_model."""
    from src.utils.metrics import evaluate_model
    assert callable(evaluate_model)


def test_loader_public_api() -> None:
    """src.data.loader exposes the four loader functions."""
    from src.data.loader import (
        load_config,
        load_weather_data,
        load_station_metadata,
        load_shapefiles,
    )
    for fn in (load_config, load_weather_data, load_station_metadata, load_shapefiles):
        assert callable(fn)


def test_preprocessor_public_api() -> None:
    """src.data.preprocessor exposes build_pipeline."""
    from src.data.preprocessor import build_pipeline
    assert callable(build_pipeline)


def test_ml_models_public_api() -> None:
    """src.models.ml_models exposes all six trainers and the helpers."""
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
    for fn in (
        build_preprocessor, prepare_features, train_ridge, train_lasso,
        train_svr, train_random_forest, train_lightgbm, train_xgboost,
        run_all_models,
    ):
        assert callable(fn)


def test_kriging_public_api() -> None:
    """src.models.kriging exposes the three kriging functions."""
    from src.models.kriging import (
        run_ordinary_kriging,
        run_universal_kriging,
        run_kriging,
    )
    for fn in (run_ordinary_kriging, run_universal_kriging, run_kriging):
        assert callable(fn)


def test_deep_learning_public_api() -> None:
    """src.models.deep_learning exposes the three DL trainers."""
    from src.models.deep_learning import (
        train_keras_tuner_mlp,
        train_transformer,
        train_tcn,
        run_all_deep_models,
    )
    for fn in (train_keras_tuner_mlp, train_transformer, train_tcn, run_all_deep_models):
        assert callable(fn)


def test_gcn_public_api() -> None:
    """src.models.gcn exposes train_gcn."""
    from src.models.gcn import train_gcn
    assert callable(train_gcn)
