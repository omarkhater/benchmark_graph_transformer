#!/usr/bin/env python
"""
Data module: dataset loading and batch enrichment for GraphTransformer.
"""
from typing import Tuple

import torch
from omegaconf import DictConfig
from torch_geometric.data import Batch
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree


def build_dataloaders(cfg: DictConfig, ) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders based on the config.

    Args:
        cfg (DictConfig): Merged configuration with `data` and `model`.

    Returns:
        Tuple[DataLoader, DataLoader]: Training and test loaders.
    """
    if cfg.data.dataset in ("MUTAG", "PROTEINS"):
        dataset = TUDataset(root="data/TUD", name=cfg.data.dataset)
    else:
        dataset = Planetoid(root="data/Planetoid", name=cfg.data.dataset)

    if cfg.model.with_degree_enc:
        max_deg = 0
        for graph in dataset:
            row, _ = graph.edge_index
            degs = degree(row, graph.num_nodes, dtype=torch.long)
            max_deg = max(int(degs.max().item()), max_deg)
        cfg.model.max_degree = max_deg + 1

    train_loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
    )
    test_loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
    )
    return train_loader, test_loader


def enrich_batch(
    batch: Batch,
    cfg_data: DictConfig,
) -> Batch:
    """
    Enrich a Batch with positional encodings and biases.

    Args:
        batch (Batch): PyG Batch object.
        cfg_data (DictConfig): Config with enrichment flags.

    Returns:
        Batch: Enriched batch.
    """
    num_nodes = batch.x.size(0)

    if cfg_data.with_degree_enc:
        row, col = batch.edge_index
        batch.out_degree = degree(row, num_nodes, dtype=torch.long)
        batch.in_degree = degree(col, num_nodes, dtype=torch.long)

    if cfg_data.with_eig_enc:
        batch.eig_pos_emb = batch.x.new_empty(
            (num_nodes, cfg_data.num_eigenc)
        ).normal_()

    if cfg_data.with_svd_enc:
        batch.svd_pos_emb = batch.x.new_empty(
            (num_nodes, 2 * cfg_data.num_svdenc)
        ).normal_()

    bias_shape = (num_nodes, num_nodes)
    if cfg_data.with_spatial_bias:
        batch.spatial_pos = torch.zeros(bias_shape, dtype=torch.long)

    if cfg_data.with_edge_bias:
        batch.edge_dist = torch.zeros(bias_shape, dtype=torch.long)

    if cfg_data.with_hop_bias:
        batch.hop_dist = torch.zeros(bias_shape, dtype=torch.long)

    return batch
