"""Tests for task type detection."""

import pytest
import torch

from graph_transformer_benchmark.evaluation.task_detection import (
    detect_task_type,
)
from graph_transformer_benchmark.evaluation.types import TaskType


def test_detect_node_classification(node_loader):
    """Should detect node-level classification task."""
    task = detect_task_type(node_loader)
    assert task == TaskType.NODE_CLASSIFICATION


def test_detect_graph_classification(graph_loader):
    """Should detect graph-level classification task."""
    # Convert targets to int64 to indicate classification
    for data in graph_loader.dataset:
        data.y = data.y.to(torch.int64)

    task = detect_task_type(graph_loader)
    assert task == TaskType.GRAPH_CLASSIFICATION


def test_detect_node_regression(regression_loader):
    """Should detect node-level regression task."""
    task = detect_task_type(regression_loader)
    assert task == TaskType.NODE_REGRESSION


def test_detect_graph_regression(generic_loader):
    """Should detect graph-level regression when y is float."""
    # Modify generic_loader to have float targets
    for data in generic_loader.dataset:
        data.y = torch.randn(1)  # Single float target per graph
    task = detect_task_type(generic_loader)
    assert task == TaskType.GRAPH_REGRESSION


def test_invalid_batch_type():
    """Should raise error for invalid batch type."""
    class InvalidLoader:
        def __iter__(self):
            yield {"x": torch.randn(4, 4)}

    with pytest.raises(ValueError, match="Unsupported batch type"):
        detect_task_type(InvalidLoader())
