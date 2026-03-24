"""
ML model training module.

Extracts and consolidates all ML model logic from notebooks/krigmain.ipynb
into clean, reusable, config-driven functions.  Every trainer accepts the
same (X_train, y_train, X_test, y_test, config) signature and returns a
result dict produced by evaluate_model().
"""

import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR

from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.utils.metrics import evaluate_model


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Build a ColumnTransformer that:
    - Applies StandardScaler to all numeric columns
      (selected via ``select_dtypes(include=['number'])``).
    - Applies OneHotEncoder(handle_unknown='ignore') to all categorical
      columns (selected via ``select_dtypes(exclude=['number'])``).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix used to determine column types.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor ready to be embedded in a Pipeline.
    """
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=["number"]).columns.tolist()

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        )

    return ColumnTransformer(transformers=transformers)


def prepare_features(station_gdf, config: dict):
    """
    Split *station_gdf* into a feature matrix X and target vector y, then
    produce train / test splits.

    Steps
    -----
    1. ``y = station_gdf[config['features']['target']]``
    2. ``X = station_gdf.drop(columns=config['features']['drop_cols'])``
    3. Numeric NaNs are filled with the column median.
    4. Non-numeric NaNs are filled with the string ``'missing'``.
    5. Returns ``(X_train, X_test, y_train, y_test)`` via
       ``train_test_split(test_size=0.2, random_state=42)``.

    Parameters
    ----------
    station_gdf : GeoDataFrame or DataFrame
        Fully-joined and feature-engineered station dataset.
    config : dict
        Loaded config.yaml dict.  Must contain ``config['features']['target']``
        and ``config['features']['drop_cols']``.

    Returns
    -------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    target = config["features"]["target"]
    drop_cols = config["features"]["drop_cols"]

    y = station_gdf[target]
    X = station_gdf.drop(columns=drop_cols)

    # Fill missing values
    numeric_cols = X.select_dtypes(include=["number"]).columns
    non_numeric_cols = X.select_dtypes(exclude=["number"]).columns

    X = X.copy()
    X[numeric_cols] = X[numeric_cols].apply(lambda col: col.fillna(col.median()))
    X[non_numeric_cols] = X[non_numeric_cols].fillna("missing")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Individual trainers
# ---------------------------------------------------------------------------


def train_ridge(X_train, y_train, X_test, y_test, config: dict) -> dict:
    """
    Train a Ridge regression model inside a preprocessing Pipeline.

    Hyperparameter ``alpha`` is seeded from
    ``config['models']['ridge']['alpha']``, then tuned via GridSearchCV over
    ``[0.1, 1, 10, 100]`` (cv=5, scoring='neg_mean_squared_error',
    n_jobs=-1).  The best estimator is used for final evaluation.

    Returns
    -------
    dict
        evaluate_model() result with model_name='Ridge'.
    """
    alpha = config["models"]["ridge"]["alpha"]

    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=alpha)),
        ]
    )

    param_grid = {"model__alpha": [0.1, 1, 10, 100]}
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    return evaluate_model(y_test, y_pred, model_name="Ridge")


def train_lasso(X_train, y_train, X_test, y_test, config: dict) -> dict:
    """
    Train a Lasso regression model inside a preprocessing Pipeline.

    Hyperparameter ``alpha`` is seeded from
    ``config['models']['lasso']['alpha']``, then tuned via GridSearchCV over
    ``[0.1, 0.5, 1, 10, 100]`` (cv=5, scoring='neg_mean_squared_error',
    n_jobs=-1).  The best estimator is used for final evaluation.

    Returns
    -------
    dict
        evaluate_model() result with model_name='Lasso'.
    """
    alpha = config["models"]["lasso"]["alpha"]

    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Lasso(alpha=alpha)),
        ]
    )

    param_grid = {"model__alpha": [0.1, 0.5, 1, 10, 100]}
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    return evaluate_model(y_test, y_pred, model_name="Lasso")


def train_svr(X_train, y_train, X_test, y_test, config: dict) -> dict:
    """
    Train a Support Vector Regressor (SVR) inside a preprocessing Pipeline.

    Because SVR is computationally expensive, the training set is first
    sub-sampled to 10 % using
    ``train_test_split(train_size=0.1, random_state=42)``.  The preprocessor
    is rebuilt on the sub-sample so that scaler statistics reflect only the
    data SVR actually trains on.

    ``kernel`` and ``C`` are read from ``config['models']['svr']``.

    Returns
    -------
    dict
        evaluate_model() result with model_name='SVR'.
    """
    kernel = config["models"]["svr"]["kernel"]
    C = config["models"]["svr"]["C"]

    # Sub-sample to keep SVR tractable
    X_train_sub, _, y_train_sub, _ = train_test_split(
        X_train, y_train, train_size=0.1, random_state=42
    )

    # Rebuild preprocessor on the subsample
    preprocessor = build_preprocessor(X_train_sub)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", SVR(kernel=kernel, C=C)),
        ]
    )

    pipeline.fit(X_train_sub, y_train_sub)
    y_pred = pipeline.predict(X_test)
    return evaluate_model(y_test, y_pred, model_name="SVR")


def train_random_forest(X_train, y_train, X_test, y_test, config: dict) -> dict:
    """
    Train a RandomForestRegressor inside a preprocessing Pipeline.

    ``n_estimators`` and ``random_state`` are read from
    ``config['models']['random_forest']``.

    Returns
    -------
    dict
        evaluate_model() result with model_name='RandomForest'.
    """
    n_estimators = config["models"]["random_forest"]["n_estimators"]
    random_state = config["models"]["random_forest"]["random_state"]

    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators, random_state=random_state
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return evaluate_model(y_test, y_pred, model_name="RandomForest")


def train_lightgbm(X_train, y_train, X_test, y_test, config: dict) -> dict:
    """
    Train an LGBMRegressor inside a preprocessing Pipeline.

    ``n_estimators``, ``learning_rate``, and ``max_depth`` are read from
    ``config['models']['lightgbm']``.  ``min_child_samples=20`` and
    ``num_leaves=15`` are added to suppress the 'No further splits with
    positive gain' warnings that appeared in the original notebook when those
    constraints were absent.

    Returns
    -------
    dict
        evaluate_model() result with model_name='LightGBM'.
    """
    n_estimators = config["models"]["lightgbm"]["n_estimators"]
    learning_rate = config["models"]["lightgbm"]["learning_rate"]
    max_depth = config["models"]["lightgbm"]["max_depth"]

    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LGBMRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    min_child_samples=20,
                    num_leaves=15,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return evaluate_model(y_test, y_pred, model_name="LightGBM")


def train_xgboost(X_train, y_train, X_test, y_test, config: dict) -> dict:
    """
    Train an XGBRegressor inside a preprocessing Pipeline.

    ``n_estimators``, ``learning_rate``, ``max_depth``, and ``random_state``
    are read from ``config['models']['xgboost']``.

    Returns
    -------
    dict
        evaluate_model() result with model_name='XGBoost'.
    """
    n_estimators = config["models"]["xgboost"]["n_estimators"]
    learning_rate = config["models"]["xgboost"]["learning_rate"]
    max_depth = config["models"]["xgboost"]["max_depth"]
    random_state = config["models"]["xgboost"]["random_state"]

    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBRegressor(
                    n_estimators=n_estimators,
                    learning_rate=learning_rate,
                    max_depth=max_depth,
                    random_state=random_state,
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    return evaluate_model(y_test, y_pred, model_name="XGBoost")


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def run_all_models(station_gdf, config: dict) -> list:
    """
    Convenience function: prepare features then train all 6 models sequentially.

    Calls ``prepare_features()`` to produce the train/test split, then runs
    each of the six model trainers.  Prints a summary table via
    ``pandas.DataFrame`` at the end.

    Parameters
    ----------
    station_gdf : GeoDataFrame or DataFrame
        Fully-joined and feature-engineered station dataset.
    config : dict
        Loaded config.yaml dict.

    Returns
    -------
    list[dict]
        One result dict per model, each as returned by ``evaluate_model()``.
    """
    X_train, X_test, y_train, y_test = prepare_features(station_gdf, config)

    trainers = [
        train_ridge,
        train_lasso,
        train_svr,
        train_random_forest,
        train_lightgbm,
        train_xgboost,
    ]

    results = []
    for trainer in trainers:
        result = trainer(X_train, y_train, X_test, y_test, config)
        results.append(result)

    # Summary table
    summary_df = pd.DataFrame(results).set_index("model")
    print("\n=== Model Comparison Summary ===")
    print(summary_df.to_string())

    return results
