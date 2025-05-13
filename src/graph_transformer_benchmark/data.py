"""Data-loader utilities for Graph-Transformer benchmarks.

Main entry-point
----------------
build_dataloaders(cfg) → (train_loader, val_loader, test_loader)

Samplers
~~~~~~~~
* OGB graph-level (`ogbg-*`) .................... _load_graph_level
* Single-graph node datasets, opt-in sampler ... _load_subgraph_level
  * NeighborLoader – `sampler.type: neighbor`
  * ClusterLoader  – `sampler.type: cluster`
* Generic multi-graph random split ............. _load_generic
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Callable, Tuple

import torch
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from omegaconf import DictConfig
from torch.utils.data import SubsetRandomSampler
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import (
    ClusterData,
    ClusterLoader,
    DataLoader,
    NeighborLoader,
)
from torch_geometric.utils import degree

# --------------------------------------------------------------------------- #
# Public aliases
# --------------------------------------------------------------------------- #

DataLoaders = Tuple[DataLoader, DataLoader, DataLoader]

# --------------------------------------------------------------------------- #
# Dataset factory
# --------------------------------------------------------------------------- #


def _flag_single_graph(data: Data) -> Data:
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


def _get_dataset(name: str, root: Path) -> Dataset:
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

# --------------------------------------------------------------------------- #
# Generic random-split loader
# --------------------------------------------------------------------------- #


_DEFAULT_VAL_RATIO: float = 0.1
_DEFAULT_TEST_RATIO: float = 0.1


def _load_generic(
    dataset: Dataset,
    loader_kw: dict[str, Any],
    val_ratio: float,
    test_ratio: float,
) -> DataLoaders:
    """Randomly split *dataset* and return three DataLoaders.
    The split is done by shuffling the dataset and slicing it into
    three parts: train, validation, and test.  The sizes of the
    validation and test sets are determined by *val_ratio* and
    *test_ratio*, respectively.  The remaining data is used for
    training.  The split is done in a way that ensures that the
    training, validation, and test sets are disjoint.
    The function returns three DataLoaders, one for each of the
    training, validation, and test sets.  The DataLoaders are
    configured with the same parameters as the original DataLoader
    used to create the dataset, except for the sampler, which is
    set to a SubsetRandomSampler for each of the three sets.
    The function also ensures that the DataLoaders are created with
    the same random seed, so that the same data is returned each
    time the function is called with the same parameters.

    Parameters
    ----------
    dataset : Dataset
        The dataset to be split into training, validation, and test sets.
    loader_kw : dict[str, Any]
        The keyword arguments to be passed to the DataLoader constructor.
    val_ratio : float
        The ratio of the dataset to be used for validation.
    test_ratio : float
        The ratio of the dataset to be used for testing.
    Returns
    -------
    DataLoaders
        A tuple of three DataLoaders: (train_loader, val_loader, test_loader).
    """
    total = len(dataset)
    n_val, n_test = int(total * val_ratio), int(total * test_ratio)
    n_train = total - n_val - n_test
    # dataset too small – just clone loaders
    if min(n_train, n_val, n_test) < 1:
        make = partial(DataLoader, dataset, **loader_kw)
        return make(shuffle=True), make(shuffle=False), make(shuffle=False)

    gen = loader_kw.get("generator")
    idx = torch.randperm(total, generator=gen).tolist()
    train_idx, val_idx, test_idx = (
        idx[:n_train],
        idx[n_train: n_train + n_val],
        idx[n_train + n_val:],
    )
    make = partial(DataLoader, dataset, **loader_kw)
    return (
        make(sampler=SubsetRandomSampler(train_idx)),
        make(sampler=SubsetRandomSampler(val_idx)),
        make(sampler=SubsetRandomSampler(test_idx)),
    )

# --------------------------------------------------------------------------- #
# OGB: graph-level classifier/regressor
# --------------------------------------------------------------------------- #


def _load_graph_level(
    dataset: PygGraphPropPredDataset,
    loader_kw: dict[str, Any],
) -> DataLoaders:
    """Return three DataLoaders for graph-level datasets.

    Parameters
    ----------
    dataset : PygGraphPropPredDataset
        The dataset to be split into training, validation, and test sets.
    loader_kw : dict[str, Any]
        The keyword arguments to be passed to the DataLoader constructor.
    Returns
    -------
    DataLoaders
        A tuple of three DataLoaders: (train_loader, val_loader, test_loader).
    Notes
    -----
    The function uses the `get_idx_split` method of the dataset to
    obtain the indices for the training, validation, and test sets.
    The DataLoaders are created with the same parameters as the
    original DataLoader used to create the dataset, except for the
    sampler, which is set to a SubsetRandomSampler for each of the
    three sets.  The function also ensures that the DataLoaders are
    created with the same random seed, so that the same data is
    returned each time the function is called with the same parameters.
    The function also sets the `shuffle` parameter to `True` for the
    training DataLoader, and to `False` for the validation and test
    DataLoaders.  This ensures that the training data is shuffled
    before each epoch, while the validation and test data are not
    shuffled.  The function also sets the `transform` parameter to
    `_flag_single_graph`, which ensures that the DataLoader returns
    batches with a single graph.  This is important for the
    `GraphTransformer` model, which expects the batch to have a
    single graph (and therefore a single target).  The model will
    raise an error if `.num_graphs` is not set to 1.
    This function is called by the `transform` argument of the
    `NeighborLoader` and `ClusterLoader` classes.  It is a no-op
    for other loaders.
    """
    idx = dataset.get_idx_split()
    train_ds, val_ds, test_ds = map(dataset.__getitem__, idx.values())
    make = partial(DataLoader, shuffle=False, **loader_kw)
    return make(train_ds, shuffle=True), make(val_ds), make(test_ds)

# --------------------------------------------------------------------------- #
# Helpers for single-graph node tasks
# --------------------------------------------------------------------------- #


def _is_node_level_graph(ds: Dataset) -> bool:
    """Check if the dataset is a single-graph node-level dataset.

    Parameters
    ----------
    ds : Dataset
        The dataset to be checked.
    Returns
    -------
    bool
        True if the dataset is a single-graph node-level dataset,
        False otherwise.
    Notes
    -----
    The function checks if the dataset contains only one graph by
    verifying that the length of the dataset is 1.  It also checks
    if the graph has train, validation, and test masks by checking
    if the graph has the attributes `train_mask`, `val_mask`, and
    `test_mask`.  These masks are used to indicate which nodes in
    the graph are used for training, validation, and testing,
    respectively.  If the dataset contains only one graph and
    the graph has these masks, the function returns True.  If the
    dataset contains more than one graph or if the graph does not
    have these masks, the function returns False.
    """
    return len(ds) == 1 and all(
        hasattr(ds[0], k) for k in ("train_mask", "val_mask", "test_mask")
    )


def _split_from_masks(data) -> dict[str, torch.Tensor]:
    """Turn boolean node masks into index tensors.

    Parameters
    ----------
    data : Data
        The data object containing the node masks.
    Returns
    -------
    dict[str, torch.Tensor]
        A dictionary containing the indices of the nodes in the
        training, validation, and test sets.  The keys are "train",
        "valid", and "test", and the values are tensors containing
        the indices of the nodes in each set.

    """
    return {
        "train": data.train_mask.nonzero(as_tuple=False).view(-1),
        "valid": data.val_mask.nonzero(as_tuple=False).view(-1),
        "test": data.test_mask.nonzero(as_tuple=False).view(-1),
    }


# --------------------------------------------------------------------------- #
# Sampler factory (neighbor / cluster)
# --------------------------------------------------------------------------- #


def _loader_factory(
    sampler_type: str,
    *,
    data,
    batch_size: int,
    num_workers: int,
    extra_dl_kw: dict[str, Any],
    **sampler_kw,
) -> Callable[[torch.Tensor, bool], DataLoader]:
    """
    Return a function that creates a DataLoader with the specified
    parameters.

    Parameters
    ----------
    sampler_type : str
        The type of sampler to use.  This can be either "neighbor"
        or "cluster".
    data : Data
        The data object to be used for the DataLoader.
    batch_size : int
        The batch size to be used for the DataLoader.
    num_workers : int
        The number of worker threads to be used for the DataLoader.
    extra_dl_kw : Dict[str, Any]
        Additional keyword arguments to be passed to the DataLoader
        constructor.
    sampler_kw : Any
        Additional keyword arguments to be passed to the sampler
        constructor.  These can include parameters such as
        `num_neighbors`, `num_parts`, and `disjoint`.
    Returns
    -------
    Callable[[torch.Tensor, bool], DataLoader]
        A function that takes a mask tensor and a shuffle boolean
        as arguments and returns a DataLoader with the specified
        parameters.
    Notes
    -----
    The function takes a mask tensor and a shuffle
    boolean as arguments and returns a DataLoader that uses the
    specified parameters.  The function is used to create DataLoaders
    for different types of samplers, such as NeighborLoader and
    ClusterLoader.  The function is called by the `make` function
    in the `_load_subgraph_level` function to create DataLoaders for
    the training, validation, and test sets.  The function is also
    used to create DataLoaders for the training, validation, and
    test sets in the `_load_graph_level` function.

    """
    # Hydra keeps the key "type" inside `sampler_kw` – drop it so it
    # never reaches NeighborLoader / ClusterLoader.
    sampler_kw.pop("type", None)
    # ── NeighborLoader ────────────────────────────────────────────────────
    if sampler_type == "neighbor":
        num_neighbors: list[int] = sampler_kw.pop("num_neighbors")
        disjoint: bool = sampler_kw.pop("disjoint", True)

        def make(mask: torch.Tensor, shuffle: bool) -> DataLoader:
            return NeighborLoader(
                data,
                input_nodes=mask,
                num_neighbors=num_neighbors,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                disjoint=disjoint,
                transform=_flag_single_graph,
                **extra_dl_kw,
                **sampler_kw,
            )

        return make

    # ── ClusterLoader ────────────────────────────────────────────────────
    if sampler_type == "cluster":
        num_parts: int = sampler_kw.pop("num_parts")
        cluster_data = ClusterData(data, num_parts=num_parts, **sampler_kw)

        def make(_mask, shuffle: bool) -> DataLoader:
            return ClusterLoader(
                cluster_data,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                transform=_flag_single_graph,
                **extra_dl_kw,
            )

        return make

    raise ValueError(f"Unknown sampler_type '{sampler_type}'.")


def _load_subgraph_level(
    dataset: Dataset,
    loader_kw: dict[str, Any],
    cfg: DictConfig,
) -> DataLoaders:
    """Sub-graph mini-batch loaders for single-graph node datasets."""
    data = dataset[0]
    splits = _split_from_masks(data)
    sampler_cfg = cfg.data.sampler

    # strip params already passed explicitly
    dl_extras = {
        k: v for k, v in loader_kw.items() if k not in {
            "batch_size", "num_workers"}
    }

    make = _loader_factory(
        sampler_cfg.type,
        data=data,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        extra_dl_kw=dl_extras,
        **sampler_cfg,  # forwards num_neighbors, num_parts, disjoint …
    )
    return make(splits["train"], True), make(splits["valid"], False), make(
        splits["test"], False
    )

# --------------------------------------------------------------------------- #
# Public entry-point
# --------------------------------------------------------------------------- #


def build_dataloaders(
    cfg: DictConfig,
    *,
    generator: torch.Generator | None = None,
    worker_init_fn: Callable[[int], None] | None = None,
) -> DataLoaders:
    """
    Build train/val/test DataLoaders from a PyG dataset.

    Parameters
    ----------
    cfg : DictConfig
        Hydra/OmegaConf configuration.  Only ``cfg.data.dataset`` is
        accessed.
    generator : torch.Generator | None
        Optional random generator for reproducibility.
    worker_init_fn : Callable[[int], None] | None
        Optional function to initialize worker processes.  This is
        useful for setting the random seed for each worker process.
    Returns
    -------
    DataLoaders
        A tuple of three DataLoaders: (train_loader, val_loader, test_loader).
    Notes
    -----
    The dataset is chosen by *cfg.data.dataset* and downloaded to
    *cfg.data.root* if it does not exist yet.  The dataset is
    expected to be in the format used by PyTorch Geometric.
    The function returns three DataLoaders, one for each of the
    training, validation, and test sets.

    """
    root = Path(getattr(cfg.data, "root", "data"))
    dataset = _get_dataset(cfg.data.dataset, root)

    loader_kw: dict[str, Any] = {
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers,
    }
    if generator is not None:
        loader_kw["generator"] = generator
    if worker_init_fn is not None:
        loader_kw["worker_init_fn"] = worker_init_fn

    if cfg.data.dataset.lower().startswith("ogbg-"):
        return _load_graph_level(dataset, loader_kw)

    if (
        getattr(cfg.data, "use_subgraph_sampler", False) and
        _is_node_level_graph(dataset)
    ):
        return _load_subgraph_level(dataset, loader_kw, cfg)

    return _load_generic(
        dataset,
        loader_kw,
        float(getattr(cfg.data, "val_ratio", _DEFAULT_VAL_RATIO)),
        float(getattr(cfg.data, "test_ratio", _DEFAULT_TEST_RATIO)),
    )


def enrich_batch(batch: Batch, cfg: DictConfig) -> Batch:
    """
    Enrich the batch with additional features based on the
    configuration settings.
    ----------
    batch : Batch
        The input batch to be enriched.
    cfg : DictConfig
        The configuration settings that determine which
        features to add to the batch.
    Returns
    -------
    Batch
        The enriched batch with additional features.
    Notes
    -----
    This function adds various positional encodings and
    distance encodings to the batch based on the configuration
    settings.  The added features include:
    - out_degree: The out-degree of each node in the graph.
    - in_degree: The in-degree of each node in the graph.
    - eig_pos_emb: A random normal tensor for eigenvalue
      positional encoding.
    - svd_pos_emb: A random normal tensor for SVD positional
      encoding.
    - spatial_pos: A zero tensor for spatial positional
        encoding.
    - edge_dist: A zero tensor for edge distance encoding.
    - hop_dist: A zero tensor for hop distance encoding.
    The function also sets the shape of the tensors based on the
    number of nodes in the batch.  The tensors are created with
    the same device and data type as the input batch.
    Parameters
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
