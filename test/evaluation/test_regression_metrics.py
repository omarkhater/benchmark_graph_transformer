"""
Unit tests for regression metric computations.

This module tests the compute_regression_metrics function in various
scenarios to ensure correct computation of mean squared error (MSE),
root mean squared error (RMSE), mean absolute error (MAE), and R² score.
"""

import numpy as np

from graph_transformer_benchmark.evaluation.regression_metrics import (
    compute_regression_metrics,
)


def test_regression_metrics_perfect():
    """
    Test compute_regression_metrics with perfect predictions.

    Expected behavior:
    - MSE, RMSE, and MAE should be 0.0.
    - R² score should be 1.0.
    """
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.0, 2.0, 3.0])
    metrics = compute_regression_metrics(y_true, y_pred)
    assert np.isclose(metrics["mse"], 0.0)
    assert np.isclose(metrics["rmse"], 0.0)
    assert np.isclose(metrics["mae"], 0.0)
    assert np.isclose(metrics["r2"], 1.0)


def test_regression_metrics_shifted():
    """
    Test compute_regression_metrics with uniformly shifted predictions.

    Expected behavior:
    - The computed MSE, RMSE, and MAE should match the expected error
      given the shift between y_true and y_pred.
    - The R² score is expected to be -0.5 for the given test case.
    """
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 3.0, 4.0])
    metrics = compute_regression_metrics(y_true, y_pred)
    expected_mse = np.mean((y_true - y_pred)**2)
    expected_rmse = np.sqrt(expected_mse)
    expected_mae = np.mean(np.abs(y_true - y_pred))
    # For y_true=[1,2,3] and y_pred=[2,3,4]: r2_score = 1 - 3/2 = -0.5
    assert np.isclose(metrics["mse"], expected_mse)
    assert np.isclose(metrics["rmse"], expected_rmse)
    assert np.isclose(metrics["mae"], expected_mae)
    assert np.isclose(metrics["r2"], -0.5)
