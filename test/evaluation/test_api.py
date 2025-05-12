import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from graph_transformer_benchmark.evaluation import evaluate


class DummyModel(nn.Module):
    """Model that returns predictions simulating a real model."""
    def __init__(self, pred_dim: int = 1):
        super().__init__()
        self.pred_dim = pred_dim

    def forward(self, batch):
        num_items = batch.num_nodes
        return torch.randn(num_items, self.pred_dim)


def test_evaluate_returns_classification_metrics(
    node_loader, device, cfg_generic
):
    """Test classification metrics for discrete targets."""
    model = DummyModel(pred_dim=2)
    metrics = evaluate(model, node_loader, device, cfg_generic)

    assert isinstance(metrics, dict)
    assert "accuracy" in metrics
    assert isinstance(metrics["accuracy"], float)
    assert "macro_f1" in metrics
    assert isinstance(metrics["macro_f1"], float)
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1


def test_evaluate_returns_regression_metrics(
    regression_loader, device, cfg_generic
):
    """Test regression metrics for continuous targets."""
    model = DummyModel(pred_dim=1)
    metrics = evaluate(model, regression_loader, device, cfg_generic)

    assert isinstance(metrics, dict)
    assert all(k in metrics for k in ["mse", "rmse", "mae", "r2"])
    assert all(isinstance(v, float) for v in metrics.values())


def test_evaluate_graph_regression(graph_loader, device, cfg_generic):
    """Should compute regression metrics for graph-level targets."""
    # Verify we're working with float targets
    assert all(data.y.dtype == torch.float32 for data in graph_loader.dataset)
    assert all(len(data.y) == 1 for data in graph_loader.dataset)

    model = DummyModel(pred_dim=1)
    metrics = evaluate(model, graph_loader, device, cfg_generic)

    assert isinstance(metrics, dict)
    assert all(k in metrics for k in ["mse", "rmse", "mae", "r2"])
    assert all(isinstance(v, float) for v in metrics.values())


def test_evaluate_node_regression(regression_loader, device, cfg_generic):
    """Should compute regression metrics for node-level targets."""
    model = DummyModel(pred_dim=1)
    metrics = evaluate(model, regression_loader, device, cfg_generic)

    assert isinstance(metrics, dict)
    assert all(k in metrics for k in ["mse", "rmse", "mae", "r2"])
    assert all(isinstance(v, float) for v in metrics.values())


def test_unsupported_loader_type():
    """Should raise error for unsupported loader type."""
    with pytest.raises(ValueError, match="Unsupported loader type"):
        evaluate(
            model=DummyModel(),
            loader=["not", "a", "dataloader"],
            device=torch.device("cpu"),
            cfg={"data": {"dataset": "test"}}
        )


def test_predictions_match_targets(node_loader, device, cfg_generic):
    """Verify predictions and targets have compatible shapes."""
    model = DummyModel(pred_dim=2)
    metrics = evaluate(model, node_loader, device, cfg_generic)

    batch = next(iter(node_loader))
    assert model(batch).shape[-1] == 2
    assert isinstance(metrics["accuracy"], float)


def test_evaluate_ogb_graph_classification(
    graph_loader, device, monkeypatch
):
    """Test OGB graph classification metrics computation."""
    cfg = OmegaConf.create({"data": {"dataset": "ogbg-molhiv"}})
    model = DummyModel(pred_dim=1)

    # Mock the task detection to return GRAPH_CLASSIFICATION
    def mock_detect_task(loader):
        from graph_transformer_benchmark.evaluation.types import TaskType
        return TaskType.GRAPH_CLASSIFICATION

    monkeypatch.setattr(
        "graph_transformer_benchmark.evaluation.api.detect_task_type",
        mock_detect_task
    )

    def mock_compute_graph_metrics(*args, **kwargs):
        return {
            "accuracy": 0.75,
            "rocauc": 0.85,
            "macro_f1": 0.80
        }

    monkeypatch.setattr(
        "graph_transformer_benchmark.evaluation.api.compute_graph_metrics",
        mock_compute_graph_metrics
    )

    metrics = evaluate(model, graph_loader, device, cfg)
    assert "accuracy" in metrics
    assert "rocauc" in metrics
    assert "macro_f1" in metrics
    assert metrics["rocauc"] == 0.85


def test_evaluate_ogb_node_classification(
    node_loader, device, monkeypatch
):
    """Test OGB node classification metrics computation."""
    cfg = OmegaConf.create({"data": {"dataset": "ogbn-arxiv"}})
    model = DummyModel(pred_dim=3)  # 3 classes

    def mock_compute_node_metrics(*args, **kwargs):
        return {
            "accuracy": 0.90,
            "macro_f1": 0.85
        }

    monkeypatch.setattr(
        "graph_transformer_benchmark.evaluation.api.compute_node_metrics",
        mock_compute_node_metrics
    )

    metrics = evaluate(model, node_loader, device, cfg)
    assert metrics["accuracy"] == 0.90
    assert metrics["macro_f1"] == 0.85
