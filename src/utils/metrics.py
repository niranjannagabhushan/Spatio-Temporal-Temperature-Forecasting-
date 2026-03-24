"""
Shared evaluation utility used by all models in this project.
"""

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_model(y_true, y_pred, model_name: str = "") -> dict:
    """
    Compute and return MSE, MAE, R² for a set of predictions.
    Also prints a formatted summary if model_name is provided.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values.
    y_pred : array-like
        Predicted target values.
    model_name : str, optional
        Human-readable label for the model; used in the printed summary.

    Returns
    -------
    dict
        {"model": model_name, "mse": float, "mae": float, "r2": float}
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2  = r2_score(y_true, y_pred)

    if model_name:
        print(f"{model_name} — MSE: {mse:.4f} | MAE: {mae:.4f} | R²: {r2:.4f}")

    return {"model": model_name, "mse": mse, "mae": mae, "r2": r2}
