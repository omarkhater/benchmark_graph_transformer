"""
Unit tests for classification metric computations.

Tests generic classification metrics as well as OGB-specific graph and node
level metrics with both binary and multiclass scenarios.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from graph_transformer_benchmark.evaluation.classification_metrics import (
    compute_generic_classification,
    compute_ogb_graph_metrics,
    compute_ogb_node_metrics,
)


def test_generic_binary_classification():
    """Test binary classification with threshold-based predictions."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8])
    metrics = compute_generic_classification(
        y_true, y_pred, is_multiclass=False
    )
    # Threshold is 0.5, so predictions are [0, 1, 0, 1]
    assert np.isclose(metrics["accuracy"], 1.0)
    assert np.isclose(metrics["macro_f1"], 1.0)


def test_generic_multiclass_classification():
    """Test multiclass classification with logit predictions."""
    y_true = np.array([0, 1, 2])
    # Perfect logits for each class
    y_pred = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    metrics = compute_generic_classification(y_true, y_pred)
    assert np.isclose(metrics["accuracy"], 1.0)
    assert np.isclose(metrics["macro_f1"], 1.0)


def test_generic_imperfect_predictions():
    """Test with some incorrect predictions."""
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([
        [1.0, 0.0, 0.0],  # correct - class 0
        [1.0, 0.0, 0.0],  # wrong - should be class 1
        [0.0, 0.0, 1.0],  # correct - class 2
        [0.0, 1.0, 0.0],  # correct - class 1
    ])
    metrics = compute_generic_classification(y_true, y_pred)
    assert np.isclose(metrics["accuracy"], 0.75)
    # Actual macro F1 calculation:
    # Class 0: precision=1.0, recall=1.0, f1=1.0
    # Class 1: precision=1.0, recall=0.5, f1=0.667
    # Class 2: precision=1.0, recall=1.0, f1=1.0
    # Macro F1 = (1.0 + 0.667 + 1.0) / 3 ≈ 0.889
    assert np.isclose(metrics["macro_f1"], 0.778, atol=1e-3)


@pytest.fixture
def mock_graph_evaluator():
    """Create a mock OGB graph evaluator."""
    with patch(
        "graph_transformer_benchmark.evaluation"
        ".classification_metrics.GraphEvaluator"
    ) as mock:
        evaluator = MagicMock()
        evaluator.eval.return_value = {
            "rocauc": 0.85,
            "acc": 0.75,
            "macro_f1": 0.80
        }
        mock.return_value = evaluator
        yield evaluator


def test_ogb_graph_metrics(mock_graph_evaluator):
    """Test OGB graph classification metrics."""
    y_true = np.array([[0], [1]])
    y_pred = np.array([[0.1], [0.9]])
    metrics = compute_ogb_graph_metrics(
        y_true, y_pred, dataset_name="ogbg-molhiv"
    )
    assert metrics["rocauc"] == 0.85
    assert metrics["accuracy"] == 0.75
    assert metrics["macro_f1"] == 0.80
    mock_graph_evaluator.eval.assert_called_once()


@pytest.fixture
def mock_node_evaluator():
    """Create a mock OGB node evaluator."""
    with patch(
        "graph_transformer_benchmark.evaluation"
        ".classification_metrics.NodeEvaluator"
    ) as mock:
        evaluator = MagicMock()
        evaluator.eval.return_value = {"acc": 0.90, "macro_f1": 0.85}
        mock.return_value = evaluator
        yield evaluator


def test_ogb_node_metrics(mock_node_evaluator):
    """Test OGB node classification metrics."""
    y_true = np.array([0, 1, 2])
    y_pred = np.array([
        [0.9, 0.1, 0.0],
        [0.1, 0.8, 0.1],
        [0.0, 0.1, 0.9]
    ])
    metrics = compute_ogb_node_metrics(
        y_true, y_pred, dataset_name="ogbn-arxiv"
    )
    assert metrics["accuracy"] == 0.90
    assert metrics["macro_f1"] == 0.85
    mock_node_evaluator.eval.assert_called_once()

def test_binary_with_2d_scores():
    # y_true is binary, but y_pred is shape (n,2)
    y_true = np.array([0, 1, 0, 1])
    # second column is "positive" score
    y_pred = np.vstack([1-y_true, y_true]).T  
    metrics = compute_generic_classification(
        y_true, y_pred, is_multiclass=False
    )
    # Should compute auroc on the positive column
    assert "auroc" in metrics
    assert 0.0 <= metrics["auroc"] <= 1.0