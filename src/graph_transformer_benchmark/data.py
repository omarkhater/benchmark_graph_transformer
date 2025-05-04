#!/usr/bin/env python
"""
Data module: dataset loading and batch enrichment for GraphTransformer.
"""
from typing import Callable, Optional, Tuple

import torch
from omegaconf import DictConfig
from torch_geometric.data import Batch
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree


def build_dataloaders(
    cfg: DictConfig,
    generator: Optional[torch.Generator] = None,
    worker_init_fn: Optional[Callable] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and test DataLoaders, seeding their RNGs for reproducibility.

    Uses getattr to guard access to model config flags so that
    non-GraphTransformer configs won't error.

    Args:
        cfg (DictConfig): Contains both data and model settings.
        generator (Optional[torch.Generator]): RNG for shuffling.
        worker_init_fn (Optional[Callable]): Worker fn to seed each worker.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and test loaders.
    """
    name = cfg.data.dataset
    if name in ("MUTAG", "PROTEINS"):
        ds = TUDataset("data/TUD", name=name)
    else:
        ds = Planetoid("data/PubMed", name=name)

    # Auto-compute max_degree only if GraphTransformer degree encoding is used
    if getattr(cfg.model, "with_degree_enc", False):
        max_deg = 0
        for data in ds:
            row, _ = data.edge_index
            if row.numel() > 0:
                degs = degree(row, data.num_nodes, dtype=torch.long)
                max_deg = max(max_deg, int(degs.max().item()))
        # store back into config for encoder
        cfg.model.max_degree = max_deg + 1

    train_loader = DataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    test_loader = DataLoader(
        ds,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        generator=generator,
        worker_init_fn=worker_init_fn,
    )
    return train_loader, test_loader


def enrich_batch(
    batch: Batch,
    cfg_data: DictConfig,
) -> Batch:
    """
    Enrich a Batch with positional encodings and biases.

    Guards each feature/bias with getattr so configs without those
    keys (e.g., GCN/SAGE/GAT) won't error.

    Args:
        batch (Batch): PyG Batch object.
        cfg_data (DictConfig): Config with enrichment flags under data.

    Returns:
        Batch: Enriched batch.
    """
    num_nodes = batch.x.size(0)

    if getattr(cfg_data, "with_degree_enc", False):
        row, col = batch.edge_index
        batch.out_degree = degree(row, num_nodes, dtype=torch.long)
        batch.in_degree = degree(col, num_nodes, dtype=torch.long)

    if getattr(cfg_data, "with_eig_enc", False):
        emb_dim = getattr(cfg_data, "num_eigenc", 0)
        batch.eig_pos_emb = batch.x.new_empty(
            (num_nodes, emb_dim)
        ).normal_()

    if getattr(cfg_data, "with_svd_enc", False):
        r = getattr(cfg_data, "num_svdenc", 0)
        batch.svd_pos_emb = batch.x.new_empty(
            (num_nodes, 2 * r)
        ).normal_()

    # prepare square bias tensors only if requested
    bias_shape = (num_nodes, num_nodes)
    if getattr(cfg_data, "with_spatial_bias", False):
        batch.spatial_pos = torch.zeros(bias_shape, dtype=torch.long)

    if getattr(cfg_data, "with_edge_bias", False):
        batch.edge_dist = torch.zeros(bias_shape, dtype=torch.long)

    if getattr(cfg_data, "with_hop_bias", False):
        batch.hop_dist = torch.zeros(bias_shape, dtype=torch.long)

    return batch
