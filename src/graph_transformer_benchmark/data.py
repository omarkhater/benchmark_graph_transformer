from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from omegaconf import DictConfig
from torch_geometric.data import Batch, Dataset
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree


def _get_dataset(name: str, root: Path) -> Dataset:
    """
    Instantiate and return a PyG Dataset based on the provided name.

    Supports:
      - OGB graph-level   (prefix "ogbg-")
      - OGB node-level    (prefix "ogbn-")
      - TUDatasets        ("MUTAG", "PROTEINS")
      - Planetoid basics  ("Cora", "CiteSeer", "PubMed")

    Args:
        name: Dataset identifier string from config.
        root: Filesystem path to store or load dataset.

    Returns:
        A torch_geometric.data.Dataset instance.

    Raises:
        ValueError: If `name` does not match any supported dataset.
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
    raise ValueError(
        f"Unsupported dataset '{name}'. "
        "Supported:\n"
        "  - ogbg-...    (OGB graph-level)\n"
        "  - ogbn-...    (OGB node-level)\n"
        "  - MUTAG, PROTEINS\n"
        "  - Cora, CiteSeer, PubMed"
    )


LoaderFn = Callable[[Dataset, Dict[str, Any]], Tuple[DataLoader, DataLoader]]
"""
Type alias for a function that builds (train_loader, val_loader)
from a Dataset and DataLoader args.
"""


def _load_graph_level(
    dataset: PygGraphPropPredDataset, args: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
    """
    Build DataLoaders for OGB graph-level tasks.

    Args:
        dataset: PygGraphPropPredDataset instance.
        args: Keyword args for DataLoader instantiation.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    splits = dataset.get_idx_split()
    train_ds = dataset[splits["train"]]
    val_ds = dataset[splits["valid"]]
    train_loader = DataLoader(train_ds, shuffle=True, **args)
    val_loader = DataLoader(val_ds, shuffle=False, **args)
    return train_loader, val_loader


def _load_node_level(
    dataset: PygNodePropPredDataset, args: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
    """
    Build DataLoaders for OGB node-level tasks.

    Entire graph is loaded each batch; training and validation masks
    are attached to the Data object.

    Args:
        dataset: PygNodePropPredDataset instance.
        args: Keyword args for DataLoader instantiation.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    data = dataset[0]
    splits = dataset.get_idx_split()
    mask_train = splits["train"]
    mask_val = splits["valid"]
    data.train_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    data.train_mask[mask_train] = True
    data.val_mask = torch.zeros(data.y.size(0), dtype=torch.bool)
    data.val_mask[mask_val] = True

    train_loader = DataLoader([data], shuffle=True, **args)
    val_loader = DataLoader([data], shuffle=False, **args)
    return train_loader, val_loader


def _load_generic(
    dataset: Dataset, args: Dict[str, Any]
) -> Tuple[DataLoader, DataLoader]:
    """
    Build DataLoaders for generic graph datasets (TUDataset or Planetoid).

    Args:
        dataset: Any torch_geometric.data.Dataset instance.
        args: Keyword args for DataLoader instantiation.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    train_loader = DataLoader(dataset, shuffle=True, **args)
    val_loader = DataLoader(dataset, shuffle=False, **args)
    return train_loader, val_loader


def build_dataloaders(
    cfg: DictConfig,
    generator: Optional[torch.Generator] = None,
    worker_init_fn: Optional[Callable[[int], None]] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders per dataset type in config.

    Dispatch table is used to select the appropriate loader function.

    Args:
        cfg: Hydra config with data fields:
            - dataset: str, dataset name
            - root: Optional[str], data root directory
            - batch_size: int
            - num_workers: int
        generator: RNG for DataLoader shuffling.
        worker_init_fn: Worker init function for reproducibility.

    Returns:
        A tuple of (train_loader, val_loader).
    """
    root = Path(getattr(cfg.data, "root", "data"))
    name = cfg.data.dataset
    dataset = _get_dataset(name, root)

    loader_args: Dict[str, Any] = {
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
        "generator": generator,
        "worker_init_fn": worker_init_fn,
    }

    key = name.lower()
    # Ordered dispatch: graph-level, node-level
    dispatch: List[Tuple[Callable[[str], bool], LoaderFn]] = [
        (lambda k: k.startswith("ogbg-"), _load_graph_level),
        (lambda k: k.startswith("ogbn-"), _load_node_level),
    ]

    for predicate, loader_fn in dispatch:
        if predicate(key):
            return loader_fn(dataset, loader_args)

    # Fallback for TUDataset or Planetoid
    return _load_generic(dataset, loader_args)


def enrich_batch(
    batch: Batch,
    cfg_data: DictConfig,
) -> Batch:
    """
    Enrich a PyG Batch with positional encodings and attention biases.

    Checks config flags and adds attributes:
      - out_degree, in_degree
      - eig_pos_emb
      - svd_pos_emb
      - spatial_pos, edge_dist, hop_dist

    Args:
        batch: A Batch object from torch_geometric.loader.
        cfg_data: Config with boolean flags and dimension params.

    Returns:
        The modified Batch with new fields as needed.
    """
    num_nodes = batch.x.size(0)

    if getattr(cfg_data, "with_degree_enc", False):
        row, col = batch.edge_index
        batch.out_degree = degree(row, num_nodes, dtype=torch.long)
        batch.in_degree = degree(col, num_nodes, dtype=torch.long)

    if getattr(cfg_data, "with_eig_enc", False):
        dim = getattr(cfg_data, "num_eigenc", 0)
        batch.eig_pos_emb = batch.x.new_empty((num_nodes, dim)).normal_()

    if getattr(cfg_data, "with_svd_enc", False):
        r = getattr(cfg_data, "num_svdenc", 0)
        batch.svd_pos_emb = batch.x.new_empty((num_nodes, 2 * r)).normal_()

    bias_shape = (num_nodes, num_nodes)
    if getattr(cfg_data, "with_spatial_bias", False):
        batch.spatial_pos = torch.zeros(bias_shape, dtype=torch.long)
    if getattr(cfg_data, "with_edge_bias", False):
        batch.edge_dist = torch.zeros(bias_shape, dtype=torch.long)
    if getattr(cfg_data, "with_hop_bias", False):
        batch.hop_dist = torch.zeros(bias_shape, dtype=torch.long)

    return batch
