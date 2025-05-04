# tests/test_data.py

import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import Planetoid, TUDataset

from graph_transformer_benchmark.data import (
    _get_dataset,
    _load_graph_level,
    _load_node_level,
    build_dataloaders,
    enrich_batch,
)


@pytest.mark.parametrize(
    "name, expected_cls, subdir",
    [
        ("MUTAG", TUDataset, "TUD"),
        ("proteins", TUDataset, "TUD"),
        ("cora", Planetoid, "Planetoid"),
        ("PubMed", Planetoid, "Planetoid"),
    ],
)
def test_get_dataset_basic(
        name: str,
        expected_cls: type,
        subdir: str,
        tmp_path: Path
        ):
    """returns correct class and root subdirectory for TU/Planetoid."""
    ds = _get_dataset(name, tmp_path)
    assert isinstance(ds, expected_cls)
    # root attribute may be either .root or .raw_dir depending on class
    root_str = getattr(ds, "root", None) or getattr(ds, "raw_dir", "")
    assert subdir in root_str


def test_get_dataset_unsupported_name(tmp_path: Path):
    """_get_dataset should raise ValueError on unknown dataset names."""
    with pytest.raises(ValueError) as exc:
        _get_dataset("not_a_real_dataset", tmp_path)
    msg = str(exc.value)
    assert "Unsupported dataset 'not_a_real_dataset'" in msg


@pytest.mark.parametrize(
    "name, dataset_cls",
    [
        ("MUTAG", TUDataset),
        ("PubMed", Planetoid),
    ],
)
def test_build_dataloaders_generic(
    name: str, dataset_cls: type, tmp_path: Path
):
    """
    build_dataloaders should return train/val loaders
    for generic graph datasets (no OGB).
    """
    cfg = OmegaConf.create({
        "data": {
            "dataset": name,
            "root": str(tmp_path),
            "batch_size": 3,
            "num_workers": 0,
        }
    })
    train_loader, val_loader = build_dataloaders(cfg)
    # Under the hood both loaders use the same dataset
    assert isinstance(train_loader.dataset, dataset_cls)
    assert isinstance(val_loader.dataset, dataset_cls)
    assert train_loader.batch_size == 3
    assert val_loader.batch_size == 3


def test_load_graph_level_split():
    """_load_graph_level honors train/valid splits from OGB mock."""
    mock_ds = MagicMock()
    # two sample graphs
    g1 = Data(x=torch.randn(2, 5), edge_index=torch.tensor([[0], [1]]))
    g2 = Data(x=torch.randn(3, 5), edge_index=torch.tensor([[0, 1], [1, 2]]))
    graphs = [g1, g2]

    mock_ds.get_idx_split.return_value = {
        "train": torch.tensor([0]),
        "valid": torch.tensor([1]),
    }
    mock_ds.__getitem__.side_effect = lambda idx: graphs[idx]
    mock_ds.__len__.return_value = len(graphs)

    args = {"batch_size": 1, "num_workers": 0}
    train_loader, val_loader = _load_graph_level(mock_ds, args)

    # The loader.dataset should be a single Data object (one graph)
    assert isinstance(train_loader.dataset, Data)
    assert isinstance(val_loader.dataset, Data)

    # And its features match the original mocked graphs
    assert torch.equal(train_loader.dataset.x, g1.x)
    assert torch.equal(val_loader.dataset.x,   g2.x)


def test_load_node_level_masks():
    """load_node_level attaches correct train/val masks on the single graph."""
    mock_ds = MagicMock()
    # single graph with 4 nodes
    num_nodes = 4
    data = Data(
        x=torch.randn(num_nodes, 8),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        y=torch.randint(0, 3, (num_nodes,)),
    )

    train_idx = torch.tensor([0, 2])
    valid_idx = torch.tensor([1, 3])
    mock_ds.get_idx_split.return_value = {
        "train": train_idx,
        "valid": valid_idx,
    }
    mock_ds.__getitem__.return_value = data
    mock_ds.__len__.return_value = 1

    args = {"batch_size": 1, "num_workers": 0}
    train_loader, val_loader = _load_node_level(mock_ds, args)

    # Single Data object batched
    tb = next(iter(train_loader))
    vb = next(iter(val_loader))
    # Check masks
    assert tb.train_mask.sum().item() == len(train_idx)
    assert vb.val_mask.sum().item() == len(valid_idx)
    for i in train_idx.tolist():
        assert tb.train_mask[i]
    for i in valid_idx.tolist():
        assert vb.val_mask[i]


def test_enrich_batch_various_flags():
    """enrich_batch adds exactly the requested attributes."""
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 2, 3], [1, 3, 4]])
    data = Data(x=x, edge_index=edge_index, y=torch.zeros(5, dtype=torch.long))
    batch = Batch.from_data_list([data])

    cfg = OmegaConf.create({
        "with_degree_enc": True,
        "with_eig_enc": True,
        "num_eigenc": 2,
        "with_svd_enc": True,
        "num_svdenc": 1,
        "with_spatial_bias": True,
        "with_edge_bias": True,
        "with_hop_bias": True,
    })

    enriched = enrich_batch(batch, cfg)
    n = x.size(0)
    # degree
    assert enriched.out_degree.shape == (n,)
    assert enriched.in_degree.shape == (n,)
    # positional
    assert enriched.eig_pos_emb.shape == (n, 2)
    assert enriched.svd_pos_emb.shape == (n, 2)
    # biases
    assert enriched.spatial_pos.shape == (n, n)
    assert enriched.edge_dist.shape == (n, n)
    assert enriched.hop_dist.shape == (n, n)


@pytest.fixture(autouse=True)
def cleanup_all(tmp_path: Path):
    """
    Remove any downloaded dataset subfolders under tmp_path
    to keep the filesystem clean.
    """
    yield
    for sub in ("TUD", "Planetoid", "OGB"):
        shutil.rmtree(tmp_path / sub, ignore_errors=True)
