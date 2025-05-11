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
