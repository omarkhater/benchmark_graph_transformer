"""Public evaluation API for GraphTransformer models."""

import torch
from omegaconf import DictConfig
from torch import nn
from torch_geometric.loader import DataLoader

from .metrics import (
    collect_predictions,
    compute_graph_metrics,
    compute_node_metrics,
    compute_regression_metrics,
)
from .task_detection import detect_task_type
from .types import TaskType


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: DictConfig,
) -> dict[str, float]:
    """Evaluate model performance with appropriate metrics.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate
    loader : DataLoader
        DataLoader containing validation/test data
    device : torch.device
        Device to run inference on
    cfg : DictConfig
        Configuration containing dataset information

    Returns
    -------
    dict[str, float]
        Dictionary of computed metrics:
        - Graph classification: accuracy, ROC-AUC, macro-F1
        - Node classification: accuracy, macro-F1
        - Graph regression: MSE, RMSE, MAE, R²
        - Node regression: MSE, RMSE, MAE, R²

    Raises
    ------
    ValueError
        If loader type is unsupported
    """
    if not isinstance(loader, DataLoader):
        raise ValueError(f"Unsupported loader type: {type(loader)}")

    task = detect_task_type(loader)
    predictions = collect_predictions(model, loader, device)

    if task == TaskType.GRAPH_CLASSIFICATION:
        return compute_graph_metrics(model, loader, device, cfg.data.dataset)
    if task == TaskType.NODE_CLASSIFICATION:
        return compute_node_metrics(model, loader, device, cfg.data.dataset)
    if task in (TaskType.GRAPH_REGRESSION, TaskType.NODE_REGRESSION):
        return compute_regression_metrics(predictions)

    raise ValueError(f"Unsupported task type: {task}")
