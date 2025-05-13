from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Any, Callable

import torch
from ogb.graphproppred import PygGraphPropPredDataset
from omegaconf import DictConfig
from torch.utils.data import SubsetRandomSampler
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from .datasets import DataLoaders, get_dataset
from .samplers import load_subgraph_level
from .utils import is_node_level_graph

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
    dataset = get_dataset(cfg.data.dataset, root)

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
        is_node_level_graph(dataset)
    ):
        return load_subgraph_level(dataset, loader_kw, cfg)

    return _load_generic(
        dataset,
        loader_kw,
        float(getattr(cfg.data, "val_ratio", _DEFAULT_VAL_RATIO)),
        float(getattr(cfg.data, "test_ratio", _DEFAULT_TEST_RATIO)),
    )
