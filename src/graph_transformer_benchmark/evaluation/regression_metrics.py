"""Regression metric computation."""

from typing import Dict

import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute standard regression metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth values
    y_pred : np.ndarray
        Model predictions

    Returns
    -------
    Dict[str, float]
        MSE, RMSE, MAE, and R² scores
    """
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "r2": float(r2_score(y_true, y_pred))
    }
