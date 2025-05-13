"""Unit tests for neighbor loader functionality."""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf
from torch_geometric.loader import NeighborLoader

from graph_transformer_benchmark.data import build_dataloaders, datasets
from graph_transformer_benchmark.data.samplers import loader_factory


def test_neighbor_loader(tmp_path, monkeypatch, one_graph_dataset):
    """Test NeighborLoader with basic configuration."""
    monkeypatch.setattr(
        datasets, "get_dataset", lambda n, r: one_graph_dataset)
    cfg = OmegaConf.create({
        "data": {
            "dataset": "pubmed",
            "root": str(tmp_path),
            "batch_size": 2,
            "num_workers": 0,
            "use_subgraph_sampler": True,
            "sampler": {
                "type": "neighbor",
                "num_neighbors": [2, 2],
                "disjoint": True,
            },
        }
    })
    train_ld, val_ld, test_ld = build_dataloaders(cfg)
    for loader in (train_ld, val_ld, test_ld):
        assert isinstance(loader, NeighborLoader)
        batch = next(iter(loader))
        assert hasattr(batch, 'x')
        assert hasattr(batch, 'edge_index')
        assert batch.x.size(0) > 0


def test_neighbor_loader_factory(one_graph_dataset):
    """Test neighbor loader factory with various parameters."""
    data = one_graph_dataset[0]
    make_loader = loader_factory(
        "neighbor",
        data=data,
        batch_size=2,
        num_workers=0,
        extra_dl_kw={},
        num_neighbors=[2, 1],
        disjoint=True
    )

    # Test with training mask and shuffle=True
    train_loader = make_loader(data.train_mask, True)
    assert isinstance(train_loader, NeighborLoader)

    # Test with validation mask and shuffle=False
    val_loader = make_loader(data.val_mask, False)
    assert isinstance(val_loader, NeighborLoader)


def test_neighbor_loader_invalid_config(one_graph_dataset):
    """Test that invalid sampler type raises ValueError."""
    data = one_graph_dataset[0]
    with pytest.raises(ValueError, match="Unknown sampler_type"):
        loader_factory(
            "invalid_type",
            data=data,
            batch_size=2,
            num_workers=0,
            extra_dl_kw={},
        )
