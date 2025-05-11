"""Task type detection utilities."""

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from .types import TaskType


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

    Notes
    -----
    Detection logic:
    1. Check if regression (float targets) or classification
    2. For both cases, check if graph-level or node-level:
       - Graph-level: has batch attribute or y.size(0) < x.size(0)
       - Node-level: y.size(0) matches number of nodes
    """
    batch = next(iter(loader))
    if not isinstance(batch, (Data, Batch)):
        raise ValueError(f"Unsupported batch type: {type(batch)}")

    is_regression = batch.y.dtype in (torch.float32, torch.float64)
    is_graph_level = (
        hasattr(batch, 'batch') or
        batch.y.size(0) < batch.x.size(0)
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
