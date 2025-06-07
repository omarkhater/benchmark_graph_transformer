"""Tests for task type detection."""

import pytest
import torch
from torch_geometric.datasets import (
    ZINC,
    Planetoid,
    TUDataset,
)
from torch_geometric.loader import DataLoader

from graph_transformer_benchmark.evaluation.task_detection import (
    detect_task_type,
    is_multiclass_task,
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


@pytest.mark.slow
@pytest.mark.parametrize(
    "dataset_name, dataset_cls, cls_kwargs, expected_task",
    [
        (
            "MUTAG",
            TUDataset,
            {"name": "MUTAG"},
            TaskType.GRAPH_CLASSIFICATION
        ),
        (
            "ENZYMES",
            TUDataset,
            {"name": "ENZYMES"},
            TaskType.GRAPH_CLASSIFICATION
        ),
        (
            "Cora",
            Planetoid,
            {"name": "Cora"},
            TaskType.NODE_CLASSIFICATION
        ),
        (
            "PubMed",
            Planetoid,
            {"name": "PubMed"},
            TaskType.NODE_CLASSIFICATION
        ),
        (
            "ZINC",
            ZINC,
            {"subset": True},
            TaskType.GRAPH_REGRESSION
        ),
    ],
)
def test_detect_task_type_standard_datasets(
    dataset_name,
    dataset_cls,
    cls_kwargs,
    expected_task,
    tmp_path,
):
    """Parameterized test covering standard PyG datasets.

    Tests:
      - MUTAG: graph classification (binary)
      - ENZYMES: graph classification (multiclass)
      - Cora: node classification (multiclass)
      - PubMed: node classification (multiclass)
      - ZINC: graph regression
    """
    root = tmp_path / dataset_name
    dataset = dataset_cls(root=str(root), **cls_kwargs)
    loader = DataLoader(dataset, batch_size=32)

    task = detect_task_type(loader)
    assert task == expected_task


@pytest.mark.slow
@pytest.mark.parametrize(
    "dataset_cls, cls_kwargs, expected_multiclass",
    [
        # Graph-level binary classification → False
        (
            TUDataset,
            {"root": None, "name": "MUTAG"},
            False,
        ),
        # Graph-level multi-class classification → True
        (
            TUDataset,
            {"root": None, "name": "ENZYMES"},
            True,
        ),
        # Node-level multi-class classification → True
        (
            Planetoid,
            {"root": None, "name": "Cora"},
            True,
        ),
        (
            Planetoid,
            {"root": None, "name": "PubMed"},
            True,
        ),
        # Graph-level regression → False
        (
            ZINC,
            {"root": None, "subset": True},
            False,
        ),
    ],
)
def test_is_multiclass_task_real_datasets(
    tmp_path,
    dataset_cls,
    cls_kwargs,
    expected_multiclass,
):
    """Verify is_multiclass_task on real PyG datasets.

    - MUTAG:    graph-binary → False
    - ENZYMES:  graph-multi-class → True
    - Cora:     node-multi-class → True
    - PubMed:   node-multi-class → True
    - ZINC:     graph-regression   → False
    """
    kwargs = cls_kwargs.copy()
    root = tmp_path / dataset_cls.__name__
    kwargs["root"] = str(root)
    dataset = dataset_cls(**kwargs)
    bs = 8 if dataset_cls is not Planetoid else 1
    loader = DataLoader(dataset, batch_size=bs)
    result = is_multiclass_task(loader)
    assert result is expected_multiclass


def test_detect_task_type_with_none_node_features(regression_none_x_loader):
    """
    Test that detect_task_type handles datasets with None node features.

    This test reproduces the AttributeError that occurs when batch.x is None
    in datasets like QM7b. The function should not crash when logging debug
    information about node feature shapes.
    """
    # This should not raise AttributeError when batch.x is None
    task_type = detect_task_type(regression_none_x_loader)

    # Should detect as graph regression since QM7b-like data has float targets
    assert task_type == TaskType.GRAPH_REGRESSION
