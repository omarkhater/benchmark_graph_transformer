"""Unit tests for data-loading helpers."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data, Dataset
from torch_geometric.loader import ClusterLoader, DataLoader, NeighborLoader

import graph_transformer_benchmark.data as data_mod

# --------------------------------------------------------------------------- #
# Fixtures ------------------------------------------------------------------ #


@pytest.fixture(scope="module")
def one_graph_dataset() -> Dataset:
    """Return a single-graph dataset with boolean masks."""
    class DS(Dataset):  # noqa: D401
        def __len__(self):  # type: ignore[override]
            return 1

        def __getitem__(self, idx):  # type: ignore[override]
            x = torch.randn(8, 3)
            y = torch.randint(0, 2, (8,))
            # Add some edges to make clustering possible
            edge_index = torch.tensor(
                [
                    [0, 1, 1, 2, 2, 3, 4, 5, 6, 7],
                    [1, 0, 2, 1, 3, 2, 5, 4, 7, 6]
                ], dtype=torch.long)
            data = Data(x=x, y=y, edge_index=edge_index)
            data.train_mask = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0]).bool()
            data.val_mask = torch.tensor([0, 0, 0, 1, 1, 0, 0, 0]).bool()
            data.test_mask = ~(data.train_mask | data.val_mask)
            return data

    return DS()


# --------------------------------------------------------------------------- #
# _is_node_level_graph ------------------------------------------------------ #


def test_is_node_level_graph(one_graph_dataset):
    """Test node-level graph detection functionality.

    Tests if the _is_node_level_graph function correctly identifies
    single-graph and multi-graph datasets.

    Args:
        one_graph_dataset: A fixture providing a single-graph dataset.
    """
    assert data_mod._is_node_level_graph(one_graph_dataset)
    many = [one_graph_dataset[0] for _ in range(3)]
    many_ds = type(
        "Many",
        (Dataset,),
        {"__len__": lambda self: 3, "__getitem__": lambda s, i: many[i]}
    )()
    assert not data_mod._is_node_level_graph(many_ds)


# --------------------------------------------------------------------------- #
# Loader behaviour ---------------------------------------------------------- #


@pytest.mark.parametrize("use_sampler", [False, True])
def test_build_dataloaders_single(
        tmp_path, monkeypatch, one_graph_dataset, use_sampler):
    """Neighbour loaders only when flag + single graph."""
    monkeypatch.setattr(
        data_mod, "_get_dataset", lambda n, r: one_graph_dataset)
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
    train_ld, val_ld, test_ld = data_mod.build_dataloaders(cfg)
    if use_sampler:
        assert isinstance(train_ld, NeighborLoader)
    else:
        assert isinstance(train_ld, DataLoader)


def test_cluster_sampler(tmp_path, monkeypatch, one_graph_dataset):
    """ClusterLoader path returns per-cluster batches."""
    monkeypatch.setattr(
        data_mod, "_get_dataset", lambda n, r: one_graph_dataset)
    cfg = OmegaConf.create(
        {
            "data": {
                "dataset": "pubmed",
                "root": str(tmp_path),
                "batch_size": 2,
                "num_workers": 0,
                "use_subgraph_sampler": True,
                "sampler": {"type": "cluster", "num_parts": 2},
            }
        }
    )
    train_ld, val_ld, test_ld = data_mod.build_dataloaders(cfg)
    for loader in (train_ld, val_ld, test_ld):
        assert isinstance(loader, ClusterLoader)
        batch = next(iter(loader))
        # Verify we get a valid batch with expected attributes
        assert hasattr(batch, 'x')
        assert hasattr(batch, 'edge_index')
        assert batch.x.size(0) > 0  # Should have some nodes


# --------------------------------------------------------------------------- #
# Generic enrich_batch smoke test ------------------------------------------ #


def test_enrich_batch_smoke():
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 2, 3], [1, 3, 4]])
    data = Data(x=x, edge_index=edge_index, y=torch.zeros(5, dtype=torch.long))
    batch = Batch.from_data_list([data])
    cfg = OmegaConf.create({"with_degree_enc": True})
    enriched = data_mod.enrich_batch(batch, cfg)
    assert hasattr(enriched, "in_degree")
    assert enriched.x.shape == x.shape
