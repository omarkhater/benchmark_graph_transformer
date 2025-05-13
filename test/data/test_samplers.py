"""Unit tests for samplers module."""

from __future__ import annotations

from omegaconf import OmegaConf
from torch_geometric.loader import ClusterLoader

from graph_transformer_benchmark.data import build_dataloaders, datasets


def test_cluster_sampler(tmp_path, monkeypatch, one_graph_dataset):
    """ClusterLoader path returns per-cluster batches."""
    monkeypatch.setattr(
        datasets, "get_dataset", lambda n, r: one_graph_dataset)
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
    train_ld, val_ld, test_ld = build_dataloaders(cfg)
    for loader in (train_ld, val_ld, test_ld):
        assert isinstance(loader, ClusterLoader)
        batch = next(iter(loader))
        # Verify we get a valid batch with expected attributes
        assert hasattr(batch, 'x')
        assert hasattr(batch, 'edge_index')
        assert batch.x.size(0) > 0  # Should have some nodes
