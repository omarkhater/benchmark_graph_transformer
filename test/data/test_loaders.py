"""Unit tests for loaders module."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader, NeighborLoader

from graph_transformer_benchmark.data import build_dataloaders, datasets


@pytest.mark.parametrize("use_sampler", [False, True])
def test_build_dataloaders_single(
        tmp_path, monkeypatch, one_graph_dataset, use_sampler):
    """Neighbour loaders only when flag + single graph."""
    monkeypatch.setattr(
        datasets, "get_dataset", lambda n, r: one_graph_dataset)
    cfg = OmegaConf.create(
        {
            "data": {
                "dataset": "pubmed",
                "root": str(tmp_path),
                "batch_size": 4,
                "num_workers": 0,
                "use_subgraph_sampler": use_sampler,
                "sampler": {"type": "neighbor", "num_neighbors": [3, 3]},
            }
        }
    )
    train_ld, val_ld, test_ld = build_dataloaders(cfg)
    if use_sampler:
        assert isinstance(train_ld, NeighborLoader)
    else:
        assert isinstance(train_ld, DataLoader)
