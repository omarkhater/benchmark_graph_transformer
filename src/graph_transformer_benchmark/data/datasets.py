from __future__ import annotations

from pathlib import Path
from typing import Tuple

from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data, Dataset
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader

DataLoaders = Tuple[DataLoader, DataLoader, DataLoader]


def flag_single_graph(data: Data) -> Data:
    """Ensure sub-graph batches expose `.num_graphs == 1`.

    This is a workaround for the fact that PyG's `NeighborLoader` and
    `ClusterLoader` do not set the `.num_graphs` attribute on the
    resulting batches.  This is important for the `GraphTransformer`
    model, which expects the batch to have a single graph (and
    therefore a single target).  The model will raise an error if
    `.num_graphs` is not set to 1.
    This function is called by the `transform` argument of the
    `NeighborLoader` and `ClusterLoader` classes.
    It is a no-op for other loaders.

    Parameters
    ----------
    data : Data
        The data object to be transformed.
    Returns
    -------
    Data
        The transformed data object with `.num_graphs` set to 1.
    """
    data.num_graphs = 1
    return data


def get_dataset(name: str, root: Path) -> Dataset:
    """Return a PyG dataset object chosen by *name*.
    The dataset is downloaded to *root* if it does not exist yet.
    The dataset is expected to be in the format used by PyTorch Geometric.

    Parameters
    ----------
    name : str
        The name of the dataset to load.
    root : Path
        The root directory where the dataset will be downloaded.
    Returns
    -------
    Dataset
        The loaded dataset object.
    Raises
    ------
    ValueError
        If the dataset name is not supported.
    """
    key = name.lower()
    if key.startswith("ogbg-"):
        return PygGraphPropPredDataset(name=key, root=str(root / "OGB"))
    if key.startswith("ogbn-"):
        return PygNodePropPredDataset(name=key, root=str(root / "OGB"))
    if key in {"mutag", "proteins"}:
        return TUDataset(root=str(root / "TUD"), name=key.upper())
    if key in {"cora", "citeseer", "pubmed"}:
        return Planetoid(root=str(root / "Planetoid"), name=key.capitalize())
    raise ValueError(f"Unsupported dataset '{name}'.")
