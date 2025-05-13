"""Task type detection utilities."""

import logging
from typing import Union

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from .types import TaskType

logger = logging.getLogger(__name__)


def detect_task_type(loader: DataLoader) -> TaskType:
    """Determine task type by examining data structure.

    Parameters
    ----------
    loader : DataLoader
        PyTorch Geometric DataLoader to inspect

    Returns
    -------
    TaskType
        Detected task type based on data characteristics
    """
    batch = next(iter(loader))
    if not isinstance(batch, (Data, Batch)):
        raise ValueError(f"Unsupported batch type: {type(batch)}")

    is_regression = batch.y.dtype in (torch.float32, torch.float64)
    is_graph_level = _is_graph_level_task(batch)

    logger.debug(
        "Task detection: regression=%s, graph_level=%s, "
        "y_shape=%s, x_shape=%s",
        is_regression, is_graph_level, batch.y.shape, batch.x.shape
    )

    if is_regression:
        return (
            TaskType.GRAPH_REGRESSION if is_graph_level
            else TaskType.NODE_REGRESSION
        )
    return (
        TaskType.GRAPH_CLASSIFICATION if is_graph_level
        else TaskType.NODE_CLASSIFICATION
    )


def _is_graph_level_task(batch: Union[Data, Batch]) -> bool:
    """Determine if task is graph-level based on data structure.

    More reliable detection using target/feature dimensions and ptr/batch.
    """
    if not hasattr(batch, 'batch'):
        # Single graph case
        return batch.y.size(0) == 1

    # Get number of graphs in batch
    num_graphs = int(batch.batch.max()) + 1
    # True if number of targets matches number of graphs
    return batch.y.size(0) == num_graphs


def is_multiclass_task(loader: DataLoader) -> bool:
    """Determine whether a DataLoader’s task is multiclass classification.

    Returns False immediately for regression (float targets),
    otherwise returns True if there are more than two classes.

    Strategy:
    1. Peek at the first example’s `y.dtype` to detect regression (float).
    2. If `dataset.num_classes` exists, use it.
    3. Otherwise scan all integer labels to count unique values.

    Parameters
    ----------
    loader : DataLoader
        A PyG DataLoader wrapping a dataset.

    Returns
    -------
    bool
        True if this is multiclass classification (>2 classes), else False.
    """
    dataset = loader.dataset

    # 1) Regression check: float targets → not multiclass classification
    first = dataset[0]
    y0 = first.y
    # If y0 is a tensor of shape (N, C), squeeze to 1D
    if y0.ndim > 1 and y0.shape[1] == 1:
        y0 = y0.squeeze(1)
    if y0.dtype in (torch.float32, torch.float64):
        return False

    # 2) Use dataset metadata if present
    num_classes = getattr(dataset, "num_classes", None)
    if num_classes is not None:
        return num_classes > 2

    # 3) Fallback: scan entire dataset to count unique integer labels
    all_labels = []
    for data in dataset:
        y = data.y
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.squeeze(1)
        all_labels.append(y.view(-1))
    all_labels = torch.cat(all_labels)
    unique_count = int(torch.unique(all_labels).numel())
    return unique_count > 2
