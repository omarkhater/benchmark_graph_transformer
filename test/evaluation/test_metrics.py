"""Tests for metric computation functions."""

import numpy as np
import pytest
import torch
from torch import nn

from graph_transformer_benchmark.evaluation.metrics import (
    collect_predictions,
    compute_classification_metrics,
    compute_regression_metrics,
)


class IdentityModel(nn.Module):
    """Model that returns identity mapping for testing."""
    def forward(self, batch):
        return batch.y


@pytest.fixture
def perfect_predictions():
    """Return perfect predictions for testing metrics."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1])
    return y_true, y_pred


@pytest.fixture
def random_regression():
    """Return regression data for testing metrics."""
    y_true = np.array([0.1, 0.2, 0.3, 0.4])
    y_pred = np.array([0.15, 0.25, 0.35, 0.45])
    return y_true, y_pred


def test_classification_metrics_perfect(perfect_predictions):
    """Should compute perfect scores for identical predictions."""
    y_true, y_pred = perfect_predictions
    metrics = compute_classification_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_regression_metrics(random_regression):
    """Should compute reasonable scores for close predictions."""
    y_true, y_pred = random_regression
    metrics = compute_regression_metrics(y_true, y_pred)

    assert 0 <= metrics["mse"] <= 1
    assert 0 <= metrics["rmse"] <= 1
    assert 0 <= metrics["mae"] <= 1
    assert 0 <= metrics["r2"] <= 1


def test_multiclass_predictions(node_loader, device):
    """Should handle multiclass logits correctly."""
    class MulticlassModel(nn.Module):
        def forward(self, batch):
            num_classes = 3
            return torch.randn(batch.num_nodes, num_classes)

    model = MulticlassModel()
    y_true, y_pred = collect_predictions(model, node_loader, device)

    assert y_pred.ndim == 2  # (num_nodes, num_classes)
    assert y_true.ndim == 1  # (num_nodes,)
