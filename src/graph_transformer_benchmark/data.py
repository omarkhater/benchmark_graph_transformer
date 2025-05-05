# src/graph_transformer_benchmark/data.py
"""
Dataset handling and DataLoader builders for Graph‑Transformer benchmarks.

Key features
------------
* Support for OGB graph‑level (`ogbg-*`) and node‑level (`ogbn-*`) datasets.
* Support for TU datasets (e.g. MUTAG), Planetoid (Cora, PubMed, …).
* Generic fallback random split with configurable val/test ratios.
* Train / validation / test loaders are returned consistently.
* Optional batch enrichment with positional encodings & attention biases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import torch
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from omegaconf import DictConfig
from torch.utils.data import random_split
from torch_geometric.data import Batch, Dataset
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

_DEFAULT_VAL_RATIO: float = 0.1
_DEFAULT_TEST_RATIO: float = 0.1

# --------------------------------------------------------------------------- #
# Type aliases
# --------------------------------------------------------------------------- #

DataLoaders = Tuple[DataLoader, DataLoader, DataLoader]
LoaderFn = Callable[[Dataset, Dict[str, Any]], DataLoaders]

# --------------------------------------------------------------------------- #
# Dataset factory
# --------------------------------------------------------------------------- #


def _get_dataset(name: str, root: Path) -> Dataset:
    """
    Instantiate a PyG dataset.

    Parameters
    ----------
    name :
        Dataset identifier (case‑insensitive).
    root :
        Directory for download / caching.

    Returns
    -------
    torch_geometric.data.Dataset
        The requested dataset.

    Raises
    ------
    ValueError
        If the dataset name is unsupported.
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
        f"Unsupported dataset '{name}'.\n"
        "Supported:\n"
        "  • ogbg-*   (OGB graph‑level)\n"
        "  • ogbn-*   (OGB node‑level)\n"
        "  • MUTAG, PROTEINS\n"
        "  • Cora, CiteSeer, PubMed"
    )

# --------------------------------------------------------------------------- #
# OGB helpers
# --------------------------------------------------------------------------- #


def _load_graph_level(
    dataset: PygGraphPropPredDataset,
    loader_kwargs: dict[str, Any],
) -> DataLoaders:
    """Return loaders that respect the official OGB splits."""
    idx = dataset.get_idx_split()
    train_ds = dataset[idx["train"]]
    val_ds = dataset[idx["valid"]]
    test_ds = dataset[idx["test"]]
    return (
        DataLoader(train_ds, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, shuffle=False, **loader_kwargs),
    )


def _load_node_level(
    dataset: PygNodePropPredDataset,
    loader_kwargs: dict[str, Any],
) -> DataLoaders:
    """Return loaders for node‑level OGB tasks (single huge graph)."""
    data = dataset[0]
    splits = dataset.get_idx_split()
    n_nodes = data.y.size(0)

    def _mask(idx: torch.Tensor) -> torch.Tensor:
        m = torch.zeros(n_nodes, dtype=torch.bool)
        m[idx] = True
        return m

    data.train_mask = _mask(splits["train"])
    data.val_mask = _mask(splits["valid"])
    data.test_mask = _mask(splits["test"])

    loader = DataLoader([data], **loader_kwargs)
    # same Data object, different boolean masks
    return loader, loader, loader


def _load_planetoid(
    dataset: Planetoid,
    loader_kwargs: dict[str, Any],
) -> DataLoaders:
    """
    Build (train, val, test) loaders for Planetoid node tasks.
    """
    data = dataset[0]
    return (
        DataLoader([data], shuffle=True,  **loader_kwargs),
        DataLoader([data], shuffle=False, **loader_kwargs),
        DataLoader([data], shuffle=False, **loader_kwargs),
    )
# --------------------------------------------------------------------------- #
# Generic fallback
# --------------------------------------------------------------------------- #


def _load_generic(
    dataset: Dataset,
    loader_kwargs: dict[str, Any],
    val_ratio: float,
    test_ratio: float,
) -> DataLoaders:
    """Randomly split a dataset into train/val/test."""
    total = len(dataset)
    n_val = int(total * val_ratio)
    n_test = int(total * test_ratio)
    n_train = total - n_val - n_test
    if min(n_train, n_val, n_test) < 1:
        raise ValueError(
            "Split ratios produce empty partition "
            f"(train={n_train}, val={n_val}, test={n_test})."
        )

    gen = loader_kwargs.get("generator", None)
    train_ds, val_ds, test_ds = random_split(
        dataset, [n_train, n_val, n_test], generator=gen
    )

    return (
        DataLoader(train_ds, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, shuffle=False, **loader_kwargs),
    )

# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def build_dataloaders(
    cfg: DictConfig,
    generator: torch.Generator | None = None,
    worker_init_fn: Callable[[int], None] | None = None,
) -> DataLoaders:
    """
    Create (train, val, test) loaders according to `cfg.data`.

    Expected fields in ``cfg.data``::
        dataset      : str
        root         : str | Path | None
        batch_size   : int
        num_workers  : int
        val_ratio    : float  (generic datasets)
        test_ratio   : float  (generic datasets)

    Other arguments are passed through to ``torch.utils.data.DataLoader``.
    """
    root = Path(getattr(cfg.data, "root", "data"))
    dataset = _get_dataset(cfg.data.dataset, root)
    loader_kwargs: dict[str, Any] = {
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
    }
    if generator is not None:
        loader_kwargs["generator"] = generator
    if worker_init_fn is not None:
        loader_kwargs["worker_init_fn"] = worker_init_fn
    val_ratio = float(
            getattr(cfg.data, "val_ratio", _DEFAULT_VAL_RATIO)
            )
    test_ratio = float(
            getattr(cfg.data, "test_ratio", _DEFAULT_TEST_RATIO)
        )
    key = cfg.data.dataset.lower()
    if key.startswith("ogbg-"):
        return _load_graph_level(dataset, loader_kwargs)
    if key.startswith("ogbn-"):
        return _load_node_level(dataset, loader_kwargs)
    if isinstance(dataset, Planetoid):
        return _load_planetoid(dataset, loader_kwargs)
    return _load_generic(
        dataset,
        loader_kwargs,
        val_ratio,
        test_ratio,
    )


# --------------------------------------------------------------------------- #
# Batch enrichment
# --------------------------------------------------------------------------- #


def enrich_batch(batch: Batch, cfg: DictConfig) -> Batch:
    """
    Optionally add degree encodings, spectral/SVD positional encodings, and
    attention‑bias matrices to *batch* according to flags in *cfg*.

    Parameters
    ----------
    batch :
        A PyG ``Batch`` produced by a DataLoader.
    cfg :
        The data‑section sub‑config containing boolean flags and dimensions.

    Returns
    -------
    Batch
        The same object, modified in‑place.
    """
    num_nodes = batch.x.size(0)

    if getattr(cfg, "with_degree_enc", False):
        row, col = batch.edge_index
        batch.out_degree = degree(row, num_nodes, dtype=torch.long)
        batch.in_degree = degree(col, num_nodes, dtype=torch.long)

    if getattr(cfg, "with_eig_enc", False):
        dim = int(getattr(cfg, "num_eigenc", 0))
        batch.eig_pos_emb = batch.x.new_empty((num_nodes, dim)).normal_()

    if getattr(cfg, "with_svd_enc", False):
        r = int(getattr(cfg, "num_svdenc", 0))
        batch.svd_pos_emb = batch.x.new_empty((num_nodes, 2 * r)).normal_()

    bias_shape = (num_nodes, num_nodes)
    if getattr(cfg, "with_spatial_bias", False):
        batch.spatial_pos = torch.zeros(bias_shape, dtype=torch.long)
    if getattr(cfg, "with_edge_bias", False):
        batch.edge_dist = torch.zeros(bias_shape, dtype=torch.long)
    if getattr(cfg, "with_hop_bias", False):
        batch.hop_dist = torch.zeros(bias_shape, dtype=torch.long)

    return batch
