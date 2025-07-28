"""Handle real-world dataset interactions, configurations, and management"""
import shutil
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.datasets import ZINC, Planetoid, TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree

CACHE_ROOT = Path(__file__).parent.parent.parent / "data"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)


@pytest.fixture(autouse=True)
def cleanup_all(tmp_path: Path):
    """Remove downloaded subfolders under tmp_path to keep filesystem clean."""
    yield
    for sub in ("TUD", "Planetoid", "OGB"):
        shutil.rmtree(tmp_path / sub, ignore_errors=True)


class DummyDataset:
    """Wraps a list of Data objects and exposes length and metadata."""
    def __init__(self, data_list: List[Data]) -> None:
        self._list = data_list
        first = data_list[0]
        self.num_node_features = first.x.size(1)
        # assume y is a 1D or 2D tensor of labels
        y = first.y.view(-1)
        self.num_classes = int(y.max().item()) + 1

    def __len__(self) -> int:
        return len(self._list)

    def __getitem__(self, idx: int) -> Data:
        return self._list[idx]


@pytest.fixture(params=[
    ("MUTAG", TUDataset, "TUD"),
    ("proteins", TUDataset, "TUD"),
    ("cora", Planetoid, "Planetoid"),
    ("PubMed", Planetoid, "Planetoid"),
])
def dataset_info(request, tmp_path: Path):
    """Provides (name, expected_class, expected_subdir, tmp_path) tuples."""
    name, expected_cls, subdir = request.param
    return name, expected_cls, subdir, tmp_path


@pytest.fixture(params=[
    ("MUTAG", TUDataset),
    ("PubMed", Planetoid),
])
def generic_cfg_and_cls(request, tmp_path: Path):
    """Provides (cfg, expected_class) for generic dataset splits."""
    name, expected_cls = request.param
    cfg = OmegaConf.create({
        "data": {
            "dataset": name,
            "root": str(tmp_path),
            "batch_size": 3,
            "num_workers": 0,
        }
    })
    return cfg, expected_cls


@pytest.fixture
def ogb_graph_dataset():
    """
    Fixtures a MagicMock OGB graph-level dataset with three sample graphs
    and explicit train/valid/test splits.
    """
    mock_ds = MagicMock(spec=PygGraphPropPredDataset)

    # Create sample graphs with all required attributes
    g1 = Data(
        x=torch.randn(2, 5),
        edge_index=torch.tensor([[0], [1]]),
        edge_attr=torch.randn(1, 4),
        y=torch.tensor([[0]])
    )
    g2 = Data(
        x=torch.randn(3, 5),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_attr=torch.randn(2, 4),
        y=torch.tensor([[1]])
    )
    g3 = Data(
        x=torch.randn(4, 5),
        edge_index=torch.tensor([[0, 2], [2, 3]]),
        edge_attr=torch.randn(2, 4),
        y=torch.tensor([[1]])
    )
    graphs = [g1, g2, g3]

    # Mock get_idx_split to return proper indices
    mock_ds.get_idx_split.return_value = {
        "train": torch.tensor([0]),
        "valid": torch.tensor([1]),
        "test": torch.tensor([2])
    }

    # Mock data loading behavior
    mock_ds.data = graphs[0]  # First graph as data
    mock_ds.slices = None  # Not needed for our test

    # Handle __getitem__ directly without file loading
    def getitem(idx):
        if isinstance(idx, (list, torch.Tensor)):
            if isinstance(idx, torch.Tensor):
                idx = idx.tolist()
            return [graphs[i] for i in idx]
        if isinstance(idx, slice):
            return graphs[idx]
        return graphs[idx]

    mock_ds.__getitem__.side_effect = getitem
    mock_ds.__len__.return_value = len(graphs)

    # Mock OGB-specific attributes
    mock_ds.num_tasks = 1
    mock_ds.task_type = "binary classification"
    mock_ds.eval_metric = "rocauc"

    return mock_ds, graphs


@pytest.fixture
def ogb_node_dataset():
    """Fixtures a MagicMock OGB node-level dataset (single graph)
    with train/valid/test masks across nodes."""
    mock_ds = MagicMock()
    num_nodes = 4
    data = Data(
        x=torch.randn(num_nodes, 8),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        y=torch.randint(0, 3, (num_nodes,)),
    )
    train_idx = torch.tensor([0, 2])
    valid_idx = torch.tensor([1])
    test_idx = torch.tensor([3])
    mock_ds.get_idx_split.return_value = {
        "train": train_idx,
        "valid": valid_idx,
        "test":  test_idx,
    }
    mock_ds.__getitem__.return_value = data
    mock_ds.__len__.return_value = 1
    return mock_ds, data, train_idx, valid_idx, test_idx


@pytest.fixture
def mutag_graph_dataloaders():
    """Fixtures dataloaders for the MUTAG dataset.
    It is a graph level dataset for binary classification."""

    ds = TUDataset(root=str(CACHE_ROOT / "TUD"), name="MUTAG")
    n = len(ds)
    train = ds[: n // 2]
    val = ds[n // 2: 3 * n // 4]
    test = ds[3 * n // 4:]
    return (
        DataLoader(train, batch_size=32),
        DataLoader(val, batch_size=32),
        DataLoader(test, batch_size=32),
    )


@pytest.fixture
def cora_node_dataloaders():
    """Fixtures dataloaders for the Cora dataset.
    It is a node-level dataset for multi-class classification.
    """
    ds = Planetoid(root=str(CACHE_ROOT / "Planetoid"), name="Cora")
    data = ds[0]
    return (
        DataLoader([data], batch_size=1),
        DataLoader([data], batch_size=1),
        DataLoader([data], batch_size=1),
    )


@pytest.fixture
def zinc_graph_regression_dataloaders():
    """Fixtures dataloaders for the ZINC dataset.
    It is a graph-level regression dataset.
    """
    ds = ZINC(root=str(CACHE_ROOT / "ZINC"), subset=True)
    sub = ds[:100]
    n = len(sub)
    idx = torch.randperm(n)
    train_idx, val_idx, test_idx = (
        idx[: n // 2],
        idx[n // 2: 3 * n // 4],
        idx[3 * n // 4:],
    )
    train = [sub[i] for i in train_idx]
    val = [sub[i] for i in val_idx]
    test = [sub[i] for i in test_idx]
    return (
        DataLoader(train, batch_size=32),
        DataLoader(val, batch_size=32),
        DataLoader(test, batch_size=32),
    )


@pytest.fixture
def cora_node_degree_regression_dataloaders():
    """
    Fixtures dataloaders for the Cora dataset with node degree as target.
    It is a node-level regression dataset.
    """
    ds = Planetoid(root=str(CACHE_ROOT / "Planetoid"), name="Cora")
    data = ds[0]
    row, _ = data.edge_index
    data.y = degree(row, data.num_nodes).float().unsqueeze(1)
    return (
        DataLoader([data], batch_size=1),
        DataLoader([data], batch_size=1),
        DataLoader([data], batch_size=1),
    )
