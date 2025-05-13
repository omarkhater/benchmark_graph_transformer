"""Unit tests for loaders module."""
from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader, NeighborLoader

from graph_transformer_benchmark.data import build_dataloaders, datasets
from graph_transformer_benchmark.data.loaders import (
    _load_generic,
    _load_graph_level,
)


@pytest.mark.parametrize("use_sampler", [False, True])
def test_build_dataloaders_single(
    tmp_path,
    monkeypatch,
    one_graph_dataset,
    use_sampler
):
    """Neighbour loaders only when flag + single graph."""
    monkeypatch.setattr(
        datasets,
        "get_dataset",
        lambda n, r: one_graph_dataset
    )
    cfg = OmegaConf.create({
        "data": {
            "dataset": "pubmed",
            "root": str(tmp_path),
            "batch_size": 4,
            "num_workers": 0,
            "use_subgraph_sampler": use_sampler,
            "sampler": {
                "type": "neighbor",
                "num_neighbors": [3, 3]
            },
        }
    })
    train_ld, val_ld, test_ld = build_dataloaders(cfg)
    if use_sampler:
        assert isinstance(train_ld, NeighborLoader)
    else:
        assert isinstance(train_ld, DataLoader)


def test_load_generic_normal_split(generic_cfg_and_cls, generic_loader):
    """Test _load_generic with normal dataset size."""
    dataset = generic_loader.dataset
    cfg, _ = generic_cfg_and_cls
    loader_kwargs = {
        "batch_size": cfg.data.batch_size,
        "num_workers": cfg.data.num_workers
    }
    loaders = _load_generic(
        dataset,
        loader_kwargs,
        val_ratio=0.33,
        test_ratio=0.33,
    )
    assert all(isinstance(loader, DataLoader) for loader in loaders)
    train_ld, val_ld, test_ld = loaders
    # With 2 samples split 34/33/33, expect 1 batch per loader
    total_samples = len(list(generic_loader))
    expected_batches = max(1, total_samples // 3)
    assert len(list(train_ld)) == expected_batches
    assert len(list(val_ld)) == expected_batches
    assert len(list(test_ld)) == expected_batches


def test_load_generic_small_split_fallback(generic_loader):
    """Test _load_generic fallback with dataset too small to split."""
    dataset = generic_loader.dataset
    loader_kwargs = {"batch_size": 1, "num_workers": 0}
    loaders = _load_generic(
        dataset,
        loader_kwargs,
        val_ratio=0.5,
        test_ratio=0.5,
    )
    assert all(isinstance(loader, DataLoader) for loader in loaders)
    train_ld, val_ld, test_ld = loaders
    # Should create identical loaders
    assert len(list(train_ld)) == len(list(val_ld)) == len(list(test_ld))


def test_load_graph_level(ogb_graph_dataset):
    """Test _load_graph_level with OGB dataset."""
    mock_ds, _ = ogb_graph_dataset
    loader_kwargs = {"batch_size": 1, "num_workers": 0}
    loaders = _load_graph_level(mock_ds, loader_kwargs)

    assert all(isinstance(loader, DataLoader) for loader in loaders)
    train_ld, val_ld, test_ld = loaders
    assert len(list(train_ld)) == 1
    assert len(list(val_ld)) == 1
    assert len(list(test_ld)) == 1


def test_build_dataloaders_with_generator(
        tmp_path, monkeypatch):
    """Test build_dataloaders with custom generator."""
    # Create a small deterministic dataset
    g0 = Data(
        x=torch.ones(2, 2),  # Use fixed features
        y=torch.tensor(0),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    g1 = Data(
        x=torch.ones(2, 2) * 2,  # Different fixed features
        y=torch.tensor(1),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
    )
    fixed_dataset = [g0, g1]

    # Mock get_dataset to return our fixed dataset
    class FixedDataset(Dataset):
        def __len__(self): return len(fixed_dataset)
        def get(self, idx): return fixed_dataset[idx]

    monkeypatch.setattr(
        datasets,
        "get_dataset",
        lambda n, r: FixedDataset()
    )

    cfg = OmegaConf.create({
        "data": {
            "dataset": "mutag",
            "root": str(tmp_path),
            "batch_size": 1,
            "num_workers": 0,
            "val_ratio": 0.5,  # Equal split for simplicity
            "test_ratio": 0.0
        }
    })

    # Use same generator twice
    generator = torch.Generator().manual_seed(42)
    loaders1 = build_dataloaders(cfg, generator=generator)

    generator = torch.Generator().manual_seed(42)
    loaders2 = build_dataloaders(cfg, generator=generator)

    # Should get exactly same order in training sets
    train_batch1 = next(iter(loaders1[0]))
    train_batch2 = next(iter(loaders2[0]))
    assert torch.equal(train_batch1.x, train_batch2.x)
    assert torch.equal(train_batch1.edge_index, train_batch2.edge_index)
