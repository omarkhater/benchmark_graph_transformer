import pytest
import torch
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph_transformer_benchmark.utils.data_utils import (
    infer_num_classes,
    infer_num_node_features,
    log_dataset_stats,
)


class DummyDataset(list):
    """Simple dataset wrapping a list with metadata attributes"""


def test_infer_num_node_features_from_dataset_attr():
    data = Data(x=torch.randn(2, 7), y=torch.tensor([0, 1]))
    ds = DummyDataset([data])
    ds.num_node_features = 7
    loader = DataLoader(ds, batch_size=1)
    assert infer_num_node_features(loader) == 7


def test_infer_num_node_features_from_subset_attr():
    data = Data(x=torch.randn(3, 5), y=torch.tensor([0, 1, 2]))
    ds = DummyDataset([data])
    ds.num_node_features = 5
    subset = Subset(ds, [0])
    loader = DataLoader(subset, batch_size=1)
    assert infer_num_node_features(loader) == 5


def test_infer_num_node_features_from_batch():
    data = Data(x=torch.randn(4, 9), y=torch.tensor([0, 1, 2, 3]))
    loader = DataLoader([data], batch_size=1)
    assert infer_num_node_features(loader) == 9


# ----------------------------------------------------------------------
# Tests for _infer_num_classes
# ----------------------------------------------------------------------

def test_infer_num_classes_from_dataset_attr():
    data = Data(x=torch.randn(2, 4), y=torch.tensor([0, 1]))
    ds = DummyDataset([data])
    ds.num_classes = 4
    loader = DataLoader(ds, batch_size=1)
    assert infer_num_classes(loader) == 4


def test_infer_num_classes_from_subset_attr():
    data = Data(x=torch.randn(3, 3), y=torch.tensor([0, 2, 1]))
    ds = DummyDataset([data])
    ds.num_classes = 3
    subset = Subset(ds, [0])
    loader = DataLoader(subset, batch_size=1)
    assert infer_num_classes(loader) == 3


def test_infer_num_classes_from_batch_scalar_labels():
    data = Data(x=torch.randn(2, 6), y=torch.tensor([0, 2]))
    loader = DataLoader([data], batch_size=1)
    assert infer_num_classes(loader) == 3


def test_infer_num_classes_from_batch_one_hot_labels():
    one_hot = torch.eye(5)
    data = Data(x=torch.randn(5, 5), y=one_hot)
    loader = DataLoader([data], batch_size=1)
    assert infer_num_classes(loader) == 5


# ----------------------------------------------------------------------
# Tests for log_dataset_stats
# ----------------------------------------------------------------------


def make_test_graphs():
    """Helper to create test graphs with known stats"""
    g1 = Data(x=torch.randn(2, 4),
              edge_index=torch.tensor([[0, 1], [1, 0]]))
    g2 = Data(x=torch.randn(3, 4),
              edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]))
    g3 = Data(x=torch.randn(2, 4),
              edge_index=torch.tensor([[0], [1]]))
    return [g1, g2, g3]


def test_log_dataset_stats_computes_correct_stats(caplog, monkeypatch):
    graphs = make_test_graphs()
    loader = DataLoader(graphs, batch_size=1)
    logged_params = {}
    monkeypatch.setattr(
        "mlflow.log_params", lambda params: logged_params.update(params))

    with caplog.at_level("INFO"):
        log_dataset_stats(loader, "test")

    assert "[TEST ]" in caplog.text
    assert "#Graphs=3" in caplog.text
    assert "Nodes (avg/min/max)=2.3/2/3" in caplog.text
    assert "Edges (avg/min/max)=2.0/1/3" in caplog.text
    assert "Mode Nodes=2" in caplog.text
    assert logged_params["data/test/num_graphs"] == 3
    assert abs(logged_params["data/test/avg_nodes"] - 2.33) < 1e-2
    assert logged_params["data/test/min_nodes"] == 2
    assert logged_params["data/test/max_nodes"] == 3
    assert abs(logged_params["data/test/avg_edges"] - 2) < 1e-2
    assert logged_params["data/test/min_edges"] == 1
    assert logged_params["data/test/max_edges"] == 3


def test_log_dataset_stats_mlflow_params(monkeypatch):
    logged_params = {}
    monkeypatch.setattr("mlflow.log_params",
                        lambda params: logged_params.update(params))

    graphs = make_test_graphs()
    loader = DataLoader(graphs, batch_size=1)

    log_dataset_stats(loader, "train", log_to_mlflow=True)

    assert logged_params["data/train/num_graphs"] == 3
    assert abs(logged_params["data/train/avg_nodes"] - 2.33) < 1e-2
    assert logged_params["data/train/min_nodes"] == 2
    assert logged_params["data/train/max_nodes"] == 3
    assert abs(logged_params["data/train/avg_edges"] - 2) < 1e-2
    assert logged_params["data/train/min_edges"] == 1
    assert logged_params["data/train/max_edges"] == 3


def test_log_dataset_stats_empty_dataset(caplog):
    loader = DataLoader([], batch_size=1)

    with caplog.at_level("INFO"), pytest.raises(ValueError):
        log_dataset_stats(loader)
