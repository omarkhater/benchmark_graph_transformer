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


def loader_factory(
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
    in the `load_subgraph_level` function to create DataLoaders for
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
                transform=flag_single_graph,
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
                transform=flag_single_graph,
                **extra_dl_kw,
            )

        return make

    raise ValueError(f"Unknown sampler_type '{sampler_type}'.")


def load_subgraph_level(
    dataset: Dataset,
    loader_kw: dict[str, Any],
    cfg: DictConfig,
) -> DataLoaders:
    """Sub-graph mini-batch loaders for single-graph node datasets."""
    data = dataset[0]
    splits = split_from_masks(data)
    sampler_cfg = cfg.data.sampler

    # strip params already passed explicitly
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
