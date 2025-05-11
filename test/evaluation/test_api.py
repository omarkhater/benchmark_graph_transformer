import pytest
import torch
import torch.nn as nn

from graph_transformer_benchmark.evaluation import evaluate


class DummyModel(nn.Module):
    """Model that returns zeros of appropriate shape."""
    def __init__(self, is_regression: bool = False):
        super().__init__()
        self.is_regression = is_regression

    def forward(self, batch):
        if self.is_regression:
            return torch.zeros(batch.num_nodes)
        return torch.zeros((batch.num_nodes, 2))


def test_evaluate_returns_classification_metrics(
    node_loader, device, cfg_generic
):
    model = DummyModel(is_regression=False)
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
    model = DummyModel(is_regression=True)
    metrics = evaluate(model, regression_loader, device, cfg_generic)

    assert isinstance(metrics, dict)
    assert all(k in metrics for k in ["mse", "rmse", "mae", "r2"])
    assert all(isinstance(v, float) for v in metrics.values())


def test_evaluate_graph_regression(graph_loader, device, cfg_generic):
    """Should compute regression metrics for graph-level targets."""
    model = DummyModel(is_regression=True)
    metrics = evaluate(model, graph_loader, device, cfg_generic)

    assert isinstance(metrics, dict)
    assert all(k in metrics for k in ["mse", "rmse", "mae", "r2"])
    assert all(isinstance(v, float) for v in metrics.values())


def test_evaluate_node_regression(regression_loader, device, cfg_generic):
    """Should compute regression metrics for node-level targets."""
    model = DummyModel(is_regression=True)
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
