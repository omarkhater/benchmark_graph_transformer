"""Tests for dataset loading functionality."""

from pathlib import Path

import pytest
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, TUDataset

from graph_transformer_benchmark.data.datasets import (
    flag_single_graph,
    get_dataset,
)


def test_flag_single_graph():
    """Test that flag_single_graph correctly sets num_graphs attribute."""
    data = Data()
    assert not hasattr(data, 'num_graphs')
    flagged = flag_single_graph(data)
    assert flagged.num_graphs == 1
    assert flagged is data  # Should modify in-place


class MockDataset:
    def __init__(self, *args, **kwargs):
        self.root = kwargs.get('root') or args[0]
        self.name = kwargs.get('name') or args[1] if len(args) > 1 else None


@pytest.mark.parametrize("name,expected_class,subdir", [
    ("ogbg-molhiv", PygGraphPropPredDataset, "OGB"),
    ("ogbn-arxiv", PygNodePropPredDataset, "OGB"),
    ("MUTAG", TUDataset, "TUD"),
    ("proteins", TUDataset, "TUD"),
    ("cora", Planetoid, "Planetoid"),
    ("PubMed", Planetoid, "Planetoid"),
])
def test_get_dataset_valid(
        tmp_path: Path, name: str, expected_class, subdir: str, monkeypatch):
    """Test get_dataset with various valid dataset names."""
    monkeypatch.setattr(
        PygGraphPropPredDataset, "__init__", MockDataset.__init__)
    monkeypatch.setattr(
        PygNodePropPredDataset, "__init__", MockDataset.__init__)
    monkeypatch.setattr(TUDataset, "__init__", MockDataset.__init__)
    monkeypatch.setattr(Planetoid, "__init__", MockDataset.__init__)

    dataset = get_dataset(name, tmp_path)
    assert isinstance(dataset, expected_class)
    expected_path = tmp_path / subdir
    assert str(expected_path) in str(dataset.root)


def test_get_dataset_invalid(tmp_path):
    """Test get_dataset with invalid dataset name."""
    with pytest.raises(ValueError, match="Unsupported dataset"):
        get_dataset("invalid-dataset", tmp_path)
