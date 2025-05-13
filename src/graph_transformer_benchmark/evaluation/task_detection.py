"""Task type detection utilities."""

import logging
from typing import Any, Iterable, Union

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from .types import TaskType

try:
    from torch_geometric.loader import ClusterLoader, NeighborLoader
except Exception:
    NeighborLoader = ClusterLoader = tuple()


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

    Parameters
    ----------
    batch : Union[Data, Batch]
        Data or Batch object to inspect
    Returns
    -------
    bool
        True if task is graph-level, False otherwise
    """
    if not hasattr(batch, 'batch'):
        # Single graph case
        return batch.y.size(0) == 1

    num_graphs = int(batch.batch.max()) + 1
    # True if number of targets matches number of graphs
    return batch.y.size(0) == num_graphs


def _grab_first(source: Any) -> Data:
    """Return the first ``Data`` object from source.

    Parameters
    ----------
    source : Any
        Source object to extract the first Data from.
    Returns
    -------
    Data
        The first Data object from the source.
    Raises
    ------
    TypeError
        If the source is not a supported type.
    """
    if isinstance(source, Data):
        return source

    if isinstance(source, (DataLoader, NeighborLoader, ClusterLoader)):
        return next(iter(source))

    if hasattr(source, "__getitem__"):  # Dataset-like
        return source[0]  # type: ignore[index]

    raise TypeError(f"Unsupported type for task detection: {type(source)}")


def _iter_datas(source: Any) -> Iterable[Data]:
    """Yield ``Data`` objects from *dataset* or *loader*.

    Parameters
    ----------
    source : Any
        Source object to iterate over.
    Yields
    ------
    Data
        The Data objects from the source.
    Raises
    ------
    TypeError
        If the source is not a supported type.

    """
    if isinstance(source, (DataLoader, NeighborLoader, ClusterLoader)):
        yield from iter(source)
    elif hasattr(source, "__iter__"):
        yield from source
    else:
        yield source


def is_multiclass_task(source: Any) -> bool:
    """Return **True** if the classification task has **> 2 classes**.

    Accepts a single ``Data`` object, a *Dataset*, or any PyG
    ``DataLoader`` (incl. ``NeighborLoader`` / ``ClusterLoader``).

    Heuristics
    ----------
    1. *Regression* – if the target dtype is floating-point → **False**.
    2. *Metadata*   – if the underlying dataset carries ``num_classes``.
    3. *Fallback*   – scan all integer targets and count unique labels.
    """
    # ── 1) peek at first target ------------------------------------------ #
    first = _grab_first(source)
    y0 = first.y
    if y0.ndim > 1 and y0.shape[1] == 1:  # shape (N,1) → (N,)
        y0 = y0.squeeze(1)
    if y0.is_floating_point():
        return False  # regression

    # ── 2) use dataset metadata if available ----------------------------- #
    dataset = getattr(source, "dataset", source)  # DataLoader → Dataset
    num_classes = getattr(dataset, "num_classes", None)
    if isinstance(num_classes, int):
        return num_classes > 2

    # ── 3) brute-force unique label count -------------------------------- #
    uniq: set[int] = set()
    for data in _iter_datas(source):
        y = data.y
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.squeeze(1)
        uniq.update(int(v) for v in y.view(-1))
        if len(uniq) > 2:  # early-exit
            return True
    return len(uniq) > 2
