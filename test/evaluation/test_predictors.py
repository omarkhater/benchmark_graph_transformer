# tests/evaluation/test_predictors.py
"""
Unit-tests for graph_transformer_benchmark.evaluation.predictors

Covered scenarios
-----------------
✔ Node-level classification – binary & multiclass
✔ Graph-level classification – binary & multiclass
✔ Node-level regression      – single & multi-target
✔ Graph-level regression     – single & multi-target
✔ Device / NumPy round-trip integrity
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from graph_transformer_benchmark.evaluation.predictors import (
    BatchPrediction,
    collect_batch_prediction,
    collect_predictions,
)


# --------------------------------------------------------------------------- #
# 🧩 Dummy models used purely for testing
# --------------------------------------------------------------------------- #
class DummyNodeClassifier(nn.Module):
    """Perfect node-classifier (identity logits)."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, data: Data):
        y = data.y.view(-1)
        if self.num_classes == 1:                         # binary → (N, 1)
            return y.float().unsqueeze(1)
        logits = torch.zeros(y.size(0), self.num_classes, device=y.device)
        logits[torch.arange(y.size(0), device=y.device), y] = 1.0
        return logits


class DummyGraphClassifier(nn.Module):
    """Perfect graph-classifier."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, batch: Batch):
        y = batch.y
        if self.num_classes == 1:
            return y.float().unsqueeze(1) if y.ndim == 1 else y.float()
        logits = torch.zeros(y.size(0), self.num_classes, device=y.device)
        logits[torch.arange(y.size(0), device=y.device), y] = 1.0
        return logits


class DummyRegressor(nn.Module):
    """Returns labels untouched – perfect regressor."""

    def forward(self, batch: Batch):
        return batch.y.clone()


# --------------------------------------------------------------------------- #
# 🔬   1.  Classification – node level                                        #
# --------------------------------------------------------------------------- #
def test_node_classification_binary(binary_node_data):
    """Test binary node-level classification logits and labels shape/values."""
    batch = Batch.from_data_list([binary_node_data])
    model = DummyNodeClassifier(num_classes=1)

    result = collect_batch_prediction(model, batch, torch.device("cpu"))

    assert isinstance(result, BatchPrediction)
    assert result.labels.shape == (4,)
    assert result.logits.shape == (4, 1)
    # argmax over a single column → zeros; compare probabilities directly
    np.testing.assert_allclose(result.logits.squeeze(1), result.labels.float())


def test_node_classification_multiclass(multiclass_node_data):
    """Test multiclass node-level classification logits and argmax labels."""
    model = DummyNodeClassifier(num_classes=3)
    loader = DataLoader([multiclass_node_data], batch_size=1)

    labels, logits = collect_predictions(model, loader, torch.device("cpu"))

    assert labels.shape == (4,)
    assert logits.shape == (4, 3)
    assert np.array_equal(logits.argmax(axis=-1), labels)


# --------------------------------------------------------------------------- #
# 🔬   2.  Classification – graph level                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "dataset,model_cls,num_classes,expected_shape",
    [
        ("binary_graph_dataset", DummyGraphClassifier, 1, (4,)),
        ("multiclass_graph_dataset", DummyGraphClassifier, 3, (4, 3)),
    ],
)
def test_graph_classification(
    request,
    dataset,
    model_cls: type[nn.Module],
    num_classes: int,
    expected_shape
):
    """Test graph-level classification logits and argmax labels
    for binary/multiclass."""
    data_list = request.getfixturevalue(dataset)
    loader = DataLoader(data_list, batch_size=2)

    model = model_cls(num_classes)
    labels, logits = collect_predictions(model, loader, torch.device("cpu"))

    assert labels.shape == (4,)
    assert logits.shape == expected_shape
    assert np.array_equal(
        logits.argmax(-1) if num_classes > 1 else (logits > 0.5).astype(int),
        labels,
    )


# --------------------------------------------------------------------------- #
# 🔬   3.  Regression – node & graph                                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "data_fixture", ["node_reg_single_target", "node_reg_multi_target"]
)
def test_node_regression(request, data_fixture):
    """Test node-level regression for single and multi-target outputs."""
    data = request.getfixturevalue(data_fixture)
    loader = DataLoader([data], batch_size=1)

    labels, preds = collect_predictions(
        DummyRegressor(), loader, torch.device("cpu"))

    assert labels.shape == preds.shape
    np.testing.assert_allclose(labels, preds)


@pytest.mark.parametrize(
    "dataset_fixture", [
        "graph_reg_single_target",
        "graph_reg_multi_target",
        "regression_none_x_loader"  # Add QM7b-like dataset test
    ]
)
def test_graph_regression(request, dataset_fixture):
    """
    Test graph-level regression for various outputs:
    - Single target graphs
    - Multi-target graphs
    - QM7b-like graphs (x=None)
    """
    data = request.getfixturevalue(dataset_fixture)

    # Convert fixture data into a DataLoader
    if isinstance(data, DataLoader):
        loader = data  # Already a DataLoader (regression_none_x_loader)
    elif isinstance(data, list):
        loader = DataLoader(data, batch_size=2)  # List of graphs
    else:
        loader = DataLoader([data], batch_size=2)  # Single graph

    labels, preds = collect_predictions(
        DummyRegressor(), loader, torch.device("cpu"))

    assert labels.shape == preds.shape
    np.testing.assert_allclose(labels, preds)


# --------------------------------------------------------------------------- #
# 🔬   4.  Device & NumPy round-trip check                                    #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "data_fixture,model_cls",
    [
        ("binary_node_data", DummyNodeClassifier),
        ("multiclass_node_data", DummyNodeClassifier),
        ("binary_graph_dataset", DummyGraphClassifier),
        ("multiclass_graph_dataset", DummyGraphClassifier),
    ],
)
def test_batch_prediction_roundtrip(request, data_fixture, model_cls):
    """Test device and NumPy round-trip for BatchPrediction outputs."""
    data_or_list = request.getfixturevalue(data_fixture)
    batch = (
        Batch.from_data_list(data_or_list)
        if isinstance(data_or_list, list)
        else Batch.from_data_list([data_or_list])
    )
    model = model_cls(num_classes=1 if "binary" in data_fixture else 3)

    result = collect_batch_prediction(model, batch, torch.device("cpu"))
    labels_np, logits_np = result.to_numpy()

    assert isinstance(labels_np, np.ndarray) and \
        isinstance(logits_np, np.ndarray)
    assert labels_np.shape[0] == logits_np.shape[0]
