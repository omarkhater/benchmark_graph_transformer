from __future__ import annotations

from typing import Any, Callable

import torch
from omegaconf import DictConfig
from torch_geometric.data import Dataset
from torch_geometric.loader import (
    ClusterData,
    ClusterLoader,
    DataLoader,
    NeighborLoader,
)

from .datasets import DataLoaders, flag_single_graph
from .utils import split_from_masks


def _make_neighbor_loader(
    data,
    batch_size: int,
    num_workers: int,
    extra_dl_kw: dict[str, Any],
    **sampler_kw,
) -> Callable[[torch.Tensor, bool], DataLoader]:
    """Create a NeighborLoader with the specified parameters.
    
    Parameters
    ----------
    data : Data
        The data object to be used for sampling
    batch_size : int
        The batch size to be used for the DataLoader
    num_workers : int
        The number of worker threads to be used for the DataLoader
    extra_dl_kw : Dict[str, Any]
        Additional keyword arguments to be passed to the DataLoader
    sampler_kw : Any
        Additional keyword arguments for NeighborLoader configuration
        
    Returns
    -------
    Callable[[torch.Tensor, bool], DataLoader]
        A function that creates a NeighborLoader with the specified parameters
        
    Notes
    -----
    The function filters the provided kwargs to only pass supported
    parameters to the NeighborLoader. Required parameters like num_neighbors
    are extracted separately from the kwargs dictionary.
    """
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
            transform=flag_single_graph,
            **extra_dl_kw,
            **sampler_kw,
        )
    return make

_CLUSTER_LOADER_KWARGS = {
    "batch_size", "shuffle", "num_workers", 
    "pin_memory", "drop_last"
}

_CLUSTER_DATA_KWARGS = {
    "num_parts", "recursive", "save_on_disk", 
    "log", "keep_inter_cluster_edges"
}

def _make_cluster_loader(
    data,
    batch_size: int,
    num_workers: int,
    extra_dl_kw: dict[str, Any],
    **sampler_kw,
) -> Callable[[torch.Tensor, bool], DataLoader]:
    """Create a ClusterLoader with the specified parameters.
    
    Parameters
    ----------
    data : Data
        The data object to be used for clustering
    batch_size : int
        The batch size to be used for the DataLoader
    num_workers : int
        The number of worker threads to be used for the DataLoader
    extra_dl_kw : Dict[str, Any]
        Additional keyword arguments to be passed to the DataLoader
    sampler_kw : Any
        Additional keyword arguments for ClusterData configuration
        
    Returns
    -------
    Callable[[torch.Tensor, bool], DataLoader]
        A function that creates a ClusterLoader with the specified parameters
        
    Notes
    -----
    The function filters the provided kwargs to only pass supported
    parameters to both ClusterData and ClusterLoader. Required parameters 
    like num_parts are extracted separately from the kwargs dictionary.
    """
    num_parts: int = sampler_kw.pop("num_parts")
    cluster_data_kwargs = {
        k: v for k, v in sampler_kw.items() 
        if k in (_CLUSTER_DATA_KWARGS - {"num_parts"}) 
    }
    cluster_data = ClusterData(
        data, 
        num_parts=num_parts,
        **cluster_data_kwargs
    )
    
    def make(_mask, shuffle: bool) -> DataLoader:
        loader_kwargs = {
            k: v for k, v in extra_dl_kw.items()
            if k in _CLUSTER_LOADER_KWARGS
        }
        return ClusterLoader(
            cluster_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            **loader_kwargs,
        )
    return make


_LOADER_FACTORIES = {
    "neighbor": _make_neighbor_loader,
    "cluster": _make_cluster_loader,
}


def loader_factory(
    sampler_type: str,
    *,
    data,
    batch_size: int,
    num_workers: int,
    extra_dl_kw: dict[str, Any],
    **sampler_kw,
) -> Callable[[torch.Tensor, bool], DataLoader]:
    """Return a function that creates a DataLoader with specified parameters.
    
    Parameters
    ----------
    sampler_type : str
        The type of sampler to use. Must be one of: "neighbor", "cluster"
    data : Data
        The data object to be used for the DataLoader
    batch_size : int
        The batch size to be used for the DataLoader
    num_workers : int
        The number of worker threads to be used for the DataLoader
    extra_dl_kw : Dict[str, Any]
        Additional keyword arguments to be passed to the DataLoader
    sampler_kw : Any
        Additional keyword arguments to be passed to the sampler
    
    Returns
    -------
    Callable[[torch.Tensor, bool], DataLoader]
        A function that creates a DataLoader with the specified parameters
    """
    sampler_kw.pop("type", None)  # Remove Hydra type key
    
    maker = _LOADER_FACTORIES.get(sampler_type)
    if maker is None:
        raise ValueError(f"Unknown sampler_type '{sampler_type}'")
        
    return maker(
        data=data,
        batch_size=batch_size,
        num_workers=num_workers,
        extra_dl_kw=extra_dl_kw,
        **sampler_kw,
    )


def load_subgraph_level(
    dataset: Dataset,
    loader_kw: dict[str, Any],
    cfg: DictConfig,
) -> DataLoaders:
    """Sub-graph mini-batch loaders for single-graph node datasets.

    Parameters
    ----------
    dataset : Dataset
        The dataset to be used for loading sub-graphs
    loader_kw : dict[str, Any]
        Additional keyword arguments for the DataLoader configuration
    cfg : DictConfig
        Object containing the sampler type and other parameters
    Returns
    -------
    DataLoaders
        A tuple of three DataLoaders: (train_loader, val_loader, test_loader)
    Notes
    -----
    The function uses the provided configuration to create a DataLoader
    for sub-graph sampling. It uses the NeighborLoader or ClusterLoader
    based on the specified sampler type. The function also handles
    additional parameters like batch size, number of workers, and
    disjoint sampling. The function returns a tuple of three DataLoaders
    for training, validation, and test sets. The function also ensures
    that the DataLoaders are created with the same random seed, so that
    the same data is returned each time the function is called with the
    same parameters. The function also sets the `transform` parameter
    to `_flag_single_graph`, which ensures that the DataLoader returns
    batches with a single graph. This is important for the
    `GraphTransformer` model, which expects the batch to have a
    single graph (and therefore a single target). The model will
    raise an error if `.num_graphs` is not set to 1.
    
    """
    data = dataset[0]
    splits = split_from_masks(data)
    sampler_cfg = cfg.data.sampler

    dl_extras = {
        k: v for k, v in loader_kw.items() if k not in {
            "batch_size", "num_workers"}
    }

    make = loader_factory(
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
