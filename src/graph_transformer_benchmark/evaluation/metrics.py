"""Task-level metric orchestration for graph-transformer benchmarks.

This module provides high-level metric computation for different graph learning
tasks. It automatically selects appropriate metric sets based on the task type
and dataset source.

Functions
---------
compute_graph_metrics : Calculate metrics for graph-level classification
compute_node_metrics : Calculate metrics for node-level classification
compute_regression_metrics : Calculate metrics for regression tasks

See Also
--------
.classification_metrics : Lower-level classification metric implementations
.predictors : Utilities for collecting model predictions
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch import nn
from torch_geometric.loader import DataLoader

from .classification_metrics import (
    compute_generic_classification,
    compute_ogb_graph_metrics,
    compute_ogb_node_metrics,
)
from .predictors import collect_predictions

MetricDict = Dict[str, float]
Array = np.ndarray


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[Array, Array]:
    """Run model inference and collect predictions.

    Parameters
    ----------
    model : nn.Module
        PyTorch model in evaluation mode
    loader : DataLoader
        PyTorch Geometric data loader
    device : torch.device
        Device to run inference on

    Returns
    -------
    Tuple[Array, Array]
        Ground truth labels and model predictions as NumPy arrays
    """
    y_true_t, y_pred_t = collect_predictions(model, loader, device)
    return y_true_t, y_pred_t


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
def compute_regression_metrics(
    y_true: Array,
    y_pred: Array,
) -> MetricDict:
    """Calculate standard regression metrics.

    Parameters
    ----------
    y_true : Array
        Ground truth values, shape (n_samples,)
    y_pred : Array
        Predicted values, shape (n_samples,)

    Returns
    -------
    MetricDict
        Dictionary containing:
            - mse: Mean squared error
            - rmse: Root mean squared error
            - mae: Mean absolute error
            - r2: R-squared score
    """
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


# --------------------------------------------------------------------------- #
# Graph-level classification
# --------------------------------------------------------------------------- #
def compute_graph_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dataset_name: str,
) -> MetricDict:
    """Compute metrics for graph-level classification tasks.

    Automatically handles OGB vs custom datasets by checking the dataset name
    prefix. For OGB datasets, includes official evaluation metrics.

    Parameters
    ----------
    model : nn.Module
        PyTorch model in evaluation mode
    loader : DataLoader
        PyTorch Geometric data loader with graph samples
    device : torch.device
        Device to run inference on
    dataset_name : str
        Name of the dataset, e.g. "ogbg-molhiv" or "custom_graphs"

    Returns
    -------
    MetricDict
        Dictionary of metric names and values. For OGB datasets,
        includes official metrics alongside standard ones.

    See Also
    --------
    .classification_metrics.compute_ogb_graph_metrics
    .classification_metrics.compute_generic_classification
    """
    y_true, y_pred = _collect_predictions(model, loader, device)

    if dataset_name.startswith("ogbg"):
        return compute_ogb_graph_metrics(y_true, y_pred, dataset_name)

    return compute_generic_classification(
        y_true, y_pred, is_multiclass=True
    )


# --------------------------------------------------------------------------- #
# Node-level classification
# --------------------------------------------------------------------------- #
def compute_node_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dataset_name: str,
) -> MetricDict:
    """Compute metrics for node-level classification tasks.

    Similar to compute_graph_metrics but for node classification tasks.
    Automatically selects appropriate metrics based on dataset type.

    Parameters
    ----------
    model : nn.Module
        PyTorch model in evaluation mode
    loader : DataLoader
        PyTorch Geometric data loader with node features and labels
    device : torch.device
        Device to run inference on
    dataset_name : str
        Name of the dataset, e.g. "ogbn-arxiv" or "custom_nodes"

    Returns
    -------
    MetricDict
        Dictionary of metric names and values. For OGB datasets,
        includes official metrics alongside standard ones.

    See Also
    --------
    .classification_metrics.compute_ogb_node_metrics
    .classification_metrics.compute_generic_classification
    """
    y_true, y_pred = _collect_predictions(model, loader, device)

    if dataset_name.startswith("ogbn"):
        return compute_ogb_node_metrics(y_true, y_pred, dataset_name)

    return compute_generic_classification(
        y_true, y_pred, is_multiclass=True
    )
