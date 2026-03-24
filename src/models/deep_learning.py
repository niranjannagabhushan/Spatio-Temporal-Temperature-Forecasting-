"""
src/models/deep_learning.py
----------------------------
Deep-learning regression models for the spatial temperature pipeline.

Three architectures are provided, each with a consistent
``(X_train, y_train, X_test, y_test, config)`` interface:

train_keras_tuner_mlp  — MLP whose hyper-parameters are tuned via
                          keras-tuner RandomSearch.
                          **Issue 3 fix**: after ``tuner.search`` the best
                          model is retrieved with
                          ``tuner.get_best_models(num_models=1)[0]`` so
                          the stale/un-fitted model is never used for
                          evaluation.

train_transformer      — Lightweight Transformer encoder for tabular data.
                          **Issues 2 & 4 fix**: receives pre-split
                          (X_train, X_test) so evaluation is strictly on
                          held-out data.

train_tcn              — Temporal Convolutional Network reshaped from
                          tabular features.
                          **Issues 2 & 4 fix**: same train/test contract.

All three return the standard result dict from
:func:`src.utils.metrics.evaluate_model`.
"""

from __future__ import annotations

from typing import Dict, Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from src.utils.metrics import evaluate_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit a StandardScaler on training data and transform both splits.

    The scaler is fitted **only** on *X_train* so that test-set statistics
    are never leaked into the normalisation step.

    Parameters
    ----------
    X_train, X_test:
        Raw feature arrays (2-D, numeric).

    Returns
    -------
    tuple
        ``(X_train_scaled, X_test_scaled)``
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


# ---------------------------------------------------------------------------
# 1. Keras Tuner MLP  (Issue 3 fix)
# ---------------------------------------------------------------------------


def train_keras_tuner_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
) -> Dict[str, Any]:
    """Train an MLP whose architecture is tuned by keras-tuner RandomSearch.

    **Issue 3 fix** — after ``tuner.search`` the best model is explicitly
    retrieved via::

        best_model = tuner.get_best_models(num_models=1)[0]

    This replaces the incorrect pattern of only extracting hyperparameter
    values and then using a stale (un-trained / wrong-config) model for
    prediction.

    Parameters
    ----------
    X_train, y_train:
        Training features and targets (already split from the full dataset).
    X_test, y_test:
        Held-out test features and targets.
    config:
        ``config.yaml`` dict.  Reads from
        ``config['models']['keras_tuner_mlp']`` — keys
        ``max_trials`` (int, default 5), ``epochs`` (int, default 30),
        and ``validation_split`` (float, default 0.2).

    Returns
    -------
    dict
        ``evaluate_model()`` result with ``model_name='KerasTunerMLP'``.
    """
    try:
        import tensorflow as tf
        import keras_tuner as kt
    except ImportError as exc:
        raise ImportError(
            "TensorFlow and keras-tuner are required.  "
            "Install with: pip install tensorflow keras-tuner"
        ) from exc

    kt_cfg = config.get("models", {}).get("keras_tuner_mlp", {})
    max_trials = kt_cfg.get("max_trials", 5)
    epochs = kt_cfg.get("epochs", 30)
    val_split = kt_cfg.get("validation_split", 0.2)

    # Scale features — scaler fitted on training data only
    X_train_scaled, X_test_scaled = _scale_features(X_train, X_test)

    n_features = X_train_scaled.shape[1]

    def build_model(hp):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.InputLayer(input_shape=(n_features,)))

        for i in range(hp.Int("num_layers", 1, 4)):
            units = hp.Choice(f"units_{i}", [32, 64, 128, 256])
            model.add(tf.keras.layers.Dense(units, activation="relu"))
            dropout_rate = hp.Float(f"dropout_{i}", 0.0, 0.4, step=0.1)
            model.add(tf.keras.layers.Dropout(dropout_rate))

        model.add(tf.keras.layers.Dense(1))

        lr = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mse",
        )
        return model

    tuner = kt.RandomSearch(
        build_model,
        objective="val_loss",
        max_trials=max_trials,
        executions_per_trial=1,
        directory="kt_dir",
        project_name="keras_tuner_mlp",
        overwrite=True,
    )

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    tuner.search(
        X_train_scaled,
        y_train,
        epochs=epochs,
        validation_split=val_split,
        callbacks=[early_stop],
        verbose=0,
    )

    # ------------------------------------------------------------------
    # Issue 3 fix: retrieve the best FITTED model rather than only the
    # best hyperparameter values.  Using only get_best_hyperparameters()
    # leaves the model un-trained and produces garbage predictions.
    # ------------------------------------------------------------------
    best_model = tuner.get_best_models(num_models=1)[0]

    # Build the model on the full input shape before predicting
    best_model.build(input_shape=(None, n_features))

    y_pred = best_model.predict(X_test_scaled, verbose=0).flatten()
    return evaluate_model(y_test, y_pred, model_name="KerasTunerMLP")


# ---------------------------------------------------------------------------
# 2. Transformer encoder  (Issues 2 & 4 fix)
# ---------------------------------------------------------------------------


def train_transformer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
) -> Dict[str, Any]:
    """Train a lightweight Transformer encoder for tabular regression.

    **Issues 2 & 4 fix** — the model is fitted exclusively on
    *(X_train, y_train)* and evaluated exclusively on *(X_test, y_test)*.
    The previous pattern in ``mlkricat.ipynb`` used *X_reshaped* /
    *y_reshaped* for both fitting and evaluation, making every metric
    an optimistic in-sample score.

    Architecture
    ------------
    Each tabular feature is treated as a single-step sequence token.
    ``n_features`` → Embedding → MultiHeadAttention → GlobalAvgPool → Dense(1).

    Parameters
    ----------
    X_train, y_train:
        Training split (not the full dataset).
    X_test, y_test:
        Held-out test split — **never seen during fitting**.
    config:
        Reads from ``config['models']['transformer']`` — keys
        ``d_model`` (int, default 32), ``num_heads`` (int, default 2),
        ``num_layers`` (int, default 2), ``dropout`` (float, default 0.1),
        ``epochs`` (int, default 30), ``batch_size`` (int, default 64).

    Returns
    -------
    dict
        ``evaluate_model()`` result with ``model_name='Transformer'``.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required.  Install with: pip install tensorflow"
        ) from exc

    tf_cfg = config.get("models", {}).get("transformer", {})
    d_model = tf_cfg.get("d_model", 32)
    num_heads = tf_cfg.get("num_heads", 2)
    num_layers = tf_cfg.get("num_layers", 2)
    dropout_rate = tf_cfg.get("dropout", 0.1)
    epochs = tf_cfg.get("epochs", 30)
    batch_size = tf_cfg.get("batch_size", 64)

    # ------------------------------------------------------------------
    # Issue 4 fix: scale features using ONLY the training split.
    # ------------------------------------------------------------------
    X_train_scaled, X_test_scaled = _scale_features(X_train, X_test)

    n_features = X_train_scaled.shape[1]

    # Reshape to (samples, seq_len=1, features) so each row is one "token"
    X_train_reshaped = X_train_scaled[:, :, np.newaxis]  # (N, features, 1)
    X_test_reshaped = X_test_scaled[:, :, np.newaxis]

    # ------------------------------------------------------------------
    # Model definition
    # ------------------------------------------------------------------
    inputs = tf.keras.Input(shape=(n_features, 1))
    x = tf.keras.layers.Dense(d_model)(inputs)

    for _ in range(num_layers):
        # Multi-head self-attention
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=d_model // max(num_heads, 1)
        )(x, x)
        attn_output = tf.keras.layers.Dropout(dropout_rate)(attn_output)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        # Feed-forward sub-layer
        ff = tf.keras.layers.Dense(d_model * 2, activation="relu")(x)
        ff = tf.keras.layers.Dense(d_model)(ff)
        ff = tf.keras.layers.Dropout(dropout_rate)(ff)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ff)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    # ------------------------------------------------------------------
    # Issue 2 fix: fit ONLY on the training split.
    # ------------------------------------------------------------------
    model.fit(
        X_train_reshaped,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0,
    )

    # ------------------------------------------------------------------
    # Issue 2 fix: evaluate on held-out test data.
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test_reshaped, verbose=0).flatten()
    return evaluate_model(y_test, y_pred, model_name="Transformer")


# ---------------------------------------------------------------------------
# 3. TCN  (Issues 2 & 4 fix)
# ---------------------------------------------------------------------------


def train_tcn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
) -> Dict[str, Any]:
    """Train a Temporal Convolutional Network (TCN) for tabular regression.

    The TCN interprets each feature dimension as a channel in a 1-D
    convolution, allowing the network to learn local patterns across feature
    groups.

    **Issues 2 & 4 fix** — identical to :func:`train_transformer`:
    the model is fitted exclusively on *(X_train, y_train)* and all metrics
    are computed on the held-out *(X_test, y_test)*.

    Parameters
    ----------
    X_train, y_train:
        Training split (not the full dataset).
    X_test, y_test:
        Held-out test split — **never seen during fitting**.
    config:
        Reads from ``config['models']['tcn']`` — keys
        ``filters`` (int, default 64), ``kernel_size`` (int, default 3),
        ``num_blocks`` (int, default 3), ``dropout`` (float, default 0.1),
        ``epochs`` (int, default 30), ``batch_size`` (int, default 64).

    Returns
    -------
    dict
        ``evaluate_model()`` result with ``model_name='TCN'``.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required.  Install with: pip install tensorflow"
        ) from exc

    tcn_cfg = config.get("models", {}).get("tcn", {})
    filters = tcn_cfg.get("filters", 64)
    kernel_size = tcn_cfg.get("kernel_size", 3)
    num_blocks = tcn_cfg.get("num_blocks", 3)
    dropout_rate = tcn_cfg.get("dropout", 0.1)
    epochs = tcn_cfg.get("epochs", 30)
    batch_size = tcn_cfg.get("batch_size", 64)

    # ------------------------------------------------------------------
    # Issue 4 fix: scale features using ONLY the training split.
    # ------------------------------------------------------------------
    X_train_scaled, X_test_scaled = _scale_features(X_train, X_test)

    n_features = X_train_scaled.shape[1]

    # Reshape: (samples, timesteps=1, channels=n_features)
    X_train_reshaped = X_train_scaled[:, np.newaxis, :]  # (N, 1, F)
    X_test_reshaped = X_test_scaled[:, np.newaxis, :]

    # ------------------------------------------------------------------
    # TCN model: dilated causal Conv1D blocks
    # ------------------------------------------------------------------
    inputs = tf.keras.Input(shape=(1, n_features))
    x = inputs
    for i in range(num_blocks):
        dilation = 2 ** i
        residual = x
        x = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="causal",
            dilation_rate=dilation,
            activation="relu",
        )(x)
        x = tf.keras.layers.SpatialDropout1D(dropout_rate)(x)
        x = tf.keras.layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            padding="causal",
            dilation_rate=dilation,
            activation="relu",
        )(x)
        x = tf.keras.layers.SpatialDropout1D(dropout_rate)(x)

        # Residual projection if channel sizes differ
        if residual.shape[-1] != filters:
            residual = tf.keras.layers.Conv1D(filters, kernel_size=1)(residual)
        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="mse")

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    # ------------------------------------------------------------------
    # Issue 2 fix: fit ONLY on the training split.
    # ------------------------------------------------------------------
    model.fit(
        X_train_reshaped,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0,
    )

    # ------------------------------------------------------------------
    # Issue 2 fix: evaluate on held-out test data.
    # ------------------------------------------------------------------
    y_pred = model.predict(X_test_reshaped, verbose=0).flatten()
    return evaluate_model(y_test, y_pred, model_name="TCN")


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------


def run_all_deep_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
) -> list:
    """Train all three deep-learning models and return a list of result dicts.

    Parameters
    ----------
    X_train, y_train:
        Training split.
    X_test, y_test:
        Held-out test split.
    config:
        Loaded ``config.yaml`` dict.

    Returns
    -------
    list[dict]
        One result dict per model (KerasTunerMLP, Transformer, TCN).
    """
    import pandas as pd

    trainers = [
        train_keras_tuner_mlp,
        train_transformer,
        train_tcn,
    ]

    results = []
    for trainer in trainers:
        result = trainer(X_train, y_train, X_test, y_test, config)
        results.append(result)

    summary_df = pd.DataFrame(results).set_index("model")
    print("\n=== Deep Learning Model Comparison ===")
    print(summary_df.to_string())

    return results
