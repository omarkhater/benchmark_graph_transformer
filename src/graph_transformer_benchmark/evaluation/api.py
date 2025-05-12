"""Public evaluation API for GraphTransformer models."""

import torch
from omegaconf import DictConfig
from torch import nn
from torch_geometric.loader import DataLoader

from .metrics import (
    collect_predictions,
    compute_generic_classification,
    compute_graph_metrics,
    compute_node_metrics,
    compute_regression_metrics,
)
from .task_detection import detect_task_type, is_multiclass_task
from .types import TaskType


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    cfg: DictConfig,
) -> dict[str, float]:
    """Evaluate model performance with appropriate metrics."""
    if not isinstance(loader, DataLoader):
        raise ValueError(f"Unsupported loader type: {type(loader)}")

    task = detect_task_type(loader)
    predictions = collect_predictions(model, loader, device)
    y_true, y_pred = predictions
    dataset_name = cfg.data.dataset

    if task in (TaskType.GRAPH_REGRESSION, TaskType.NODE_REGRESSION):
        return compute_regression_metrics(y_true, y_pred)

    if dataset_name.startswith('ogb'):
        if task == TaskType.GRAPH_CLASSIFICATION:
            return compute_graph_metrics(model, loader, device, dataset_name)
        if task == TaskType.NODE_CLASSIFICATION:
            return compute_node_metrics(model, loader, device, dataset_name)

    if task == TaskType.GRAPH_CLASSIFICATION:
        is_multiclass = is_multiclass_task(loader)
        return compute_generic_classification(y_true, y_pred, is_multiclass)
