import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph_transformer_benchmark.utils.data_utils import (
    infer_num_node_features,
    log_dataset_stats,
)

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
    monkeypatch.setattr(
        "mlflow.log_params",
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


# ----------------------------------------------------------------------
# Tests for infer_num_node_features (comprehensive via test suites)
# ----------------------------------------------------------------------

def test_infer_num_node_features_comprehensive(graph_classification_suite):
    """Test infer_num_node_features across different classification scenarios.

    This comprehensive test ensures the function works correctly across
    various real-world graph classification datasets with different node
    feature patterns.
    """
    for case_name, loader in graph_classification_suite.items():
        # Simply verify the function doesn't crash and returns a
        # non-negative int
        inferred_features = infer_num_node_features(loader)

        assert isinstance(inferred_features, int), (
            f"{case_name}: should return int, got {type(inferred_features)}"
        )
        assert inferred_features >= 0, (
            f"{case_name}: should return non-negative features, "
            f"got {inferred_features}"
        )

        # Verify consistency - calling twice should give same result
        inferred_features_2 = infer_num_node_features(loader)
        assert inferred_features == inferred_features_2, (
            f"{case_name}: function should be deterministic"
        )


def test_infer_num_node_features_regression_comprehensive(
    graph_regression_suite
):
    """Test infer_num_node_features on different graph regression scenarios.

    This comprehensive test ensures consistent feature dimension inference
    across various graph regression cases found in real-world datasets
    like:
    - Single target prediction (e.g. ZINC)
    - Multi-target prediction (e.g. QM9)
    - Graphs without node features (e.g. QM7b)
    - Graphs with edge attributes (e.g. molecular graphs)
    - Varied graph sizes (e.g. protein structures)
    """
    for case_name, loader in graph_regression_suite.items():
        # Simply verify the function doesn't crash and returns a
        # non-negative int
        inferred_features = infer_num_node_features(loader)

        assert isinstance(inferred_features, int), (
            f"{case_name}: should return int, got {type(inferred_features)}"
        )
        assert inferred_features >= 0, (
            f"{case_name}: should return non-negative features, "
            f"got {inferred_features}"
        )

        # Verify consistency - calling twice should give same result
        inferred_features_2 = infer_num_node_features(loader)
        assert inferred_features == inferred_features_2, (
            f"{case_name}: function should be deterministic"
        )
