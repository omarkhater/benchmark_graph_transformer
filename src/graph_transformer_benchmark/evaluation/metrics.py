"""Core metric computation functions for different dataset types.

This module implements performance metrics for graph and node level tasks,
including both classification and regression scenarios.
"""

from typing import Dict

import numpy as np
import torch
from ogb.graphproppred import Evaluator as GraphEvaluator
from ogb.nodeproppred import Evaluator as NodeEvaluator
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from torch import nn
from torch_geometric.loader import DataLoader

from .classification_metrics import compute_generic_classification
from .predictors import collect_predictions


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    is_multiclass: bool = True
) -> Dict[str, float]:
    """Compute standard classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels
    y_pred : np.ndarray
        Model predictions
    is_multiclass : bool, optional
        Whether to treat as multiclass task, by default True

    Returns
    -------
    Dict[str, float]
        Dictionary containing accuracy and macro-F1 scores
    """
    if is_multiclass and y_pred.ndim > 1:
        preds = y_pred.argmax(axis=-1)
    else:
        preds = (y_pred > 0).astype(int)

    return {
        "accuracy": float((preds == y_true).mean()),
        "macro_f1": float(
            f1_score(y_true, preds, average="macro", zero_division=0)
        )
    }


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
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
        Dictionary containing MSE, RMSE, MAE and R² scores
    """
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "r2": float(r2_score(y_true, y_pred))
    }


def compute_graph_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dataset_name: str
) -> Dict[str, float]:
    """Compute metrics for OGB graph-level tasks.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate
    loader : DataLoader
        DataLoader containing validation/test data
    device : torch.device
        Device to run inference on
    dataset_name : str
        Name of the OGB dataset

    Returns
    -------
    Dict[str, float]
        Dictionary containing accuracy, ROC-AUC and macro-F1 scores
    """
    evaluator = GraphEvaluator(name=dataset_name)
    y_true, y_pred = collect_predictions(model, loader, device)
    result = evaluator.eval({"y_true": y_true, "y_pred": y_pred})

    return {
        "accuracy": result.get("acc", 0.0),
        "rocauc": result["rocauc"],
        "macro_f1": result.get("macro_f1", 0.0)
    }


def compute_node_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    dataset_name: str
) -> Dict[str, float]:
    """Compute metrics for OGB node-level tasks.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate
    loader : DataLoader
        DataLoader containing validation/test data
    device : torch.device
        Device to run inference on
    dataset_name : str
        Name of the OGB dataset

    Returns
    -------
    Dict[str, float]
        Dictionary containing accuracy and macro-F1 scores
    """
    y_true, y_pred = collect_predictions(model, loader, device)
    if not dataset_name.startswith("ogbn"):
        return compute_generic_classification(
            y_true, y_pred, is_multiclass=True)

    evaluator = NodeEvaluator(name=dataset_name)
    preds = y_pred.argmax(axis=-1) if y_pred.ndim > 1 else y_pred
    result = evaluator.eval({"y_true": y_true, "y_pred": preds})
    return {
        "accuracy": result["acc"],
        "macro_f1": result.get("macro_f1", 0.0)
    }
