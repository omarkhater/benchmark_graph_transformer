# tests/evaluation/test_predictors.py
"""
Unit-tests for graph_transformer_benchmark.evaluation.predictors
"""

from __future__ import annotations

from test.fixtures.model_fixtures import (
    DummyGraphClassifier,
    DummyNodeClassifier,
    DummyRegressor,
)

import numpy as np
import torch

from graph_transformer_benchmark.evaluation.predictors import (
    collect_predictions,
)
from graph_transformer_benchmark.evaluation.task_detection import (
    _is_graph_level_task,
)
from graph_transformer_benchmark.utils.data_utils import infer_num_classes


def test_classification_comprehensive(graph_classification_suite):
    """Test classification prediction collection across different scenarios.

    This test ensures consistent model behavior across various classification
    scenarios found in real-world datasets:
    - Node-level binary/multiclass classification
    - Graph-level binary/multiclass classification
    - Sparse node features (e.g. citation networks)
    - Edge attributes (e.g. molecular graphs)
    - Masked evaluation (train/val/test splits)
    """
    device = torch.device("cpu")

    for case_name, loader in graph_classification_suite.items():
        sample_batch = next(iter(loader))
        is_graph_level = _is_graph_level_task(sample_batch)
        num_classes = infer_num_classes(loader)
        if is_graph_level:
            model_cls = DummyGraphClassifier
        else:
            model_cls = DummyNodeClassifier

        model = model_cls(num_classes)
        labels, logits = collect_predictions(model, loader, device)
        if is_graph_level:
            expected_samples = len(loader.dataset)
        else:
            # For node-level tasks, use the number of nodes in the single graph
            expected_samples = loader.dataset[0].num_nodes
        assert labels.shape[0] == expected_samples, \
            f"{case_name}: wrong number of predictions"

        if num_classes > 1:  # Multiclass
            assert logits.shape == (expected_samples, num_classes), \
                f"{case_name}: wrong multiclass shape"
            assert np.array_equal(
                logits.argmax(-1), labels
            ), f"{case_name}: predictions don't match"
        else:  # Binary
            assert labels.shape == logits.shape, \
                f"{case_name}: binary shape mismatch"
            # For binary classification, DummyClassifier returns probabilities
            # that exactly match the labels
            np.testing.assert_allclose(
                logits.squeeze(),
                labels.astype(float),
                err_msg=f"Binary prediction mismatch for {case_name}"
            )


def test_graph_regression_comprehensive(graph_regression_suite):
    """Test model behavior across different graph regression scenarios.

    This test ensures consistent model behavior across various graph regression
    cases found in real-world datasets like:
    - Single target prediction (e.g. ZINC)
    - Multi-target prediction (e.g. QM9)
    - Graphs without node features (e.g. QM7b)
    - Graphs with edge attributes (e.g. molecular graphs)
    - Varied graph sizes (e.g. protein structures)
    """
    device = torch.device("cpu")

    for case_name, loader in graph_regression_suite.items():
        # Configure model to match target dimensionality
        sample_batch = next(iter(loader))
        out_dim = (
            sample_batch.y.size(-1)
            if sample_batch.y.ndim > 1
            else 1
        )
        model = DummyRegressor(out_dim)
        labels, preds = collect_predictions(model, loader, device)
        assert labels.shape == preds.shape, f"{case_name}: shape mismatch"
        if case_name == 'multi_target':
            # For multi-target, PyG concatenates targets so we get 1D output
            # Each graph has 5 targets, 4 graphs = 20 total values
            assert labels.ndim == 1, (
                f"{case_name}: expected 1D concatenated targets"
            )
            assert labels.shape[0] % 5 == 0, (
                f"{case_name}: target count should be multiple of 5"
            )
            # Reshape to validate multi-target structure
            num_graphs = labels.shape[0] // 5
            labels_2d = labels.reshape(num_graphs, 5)
            assert labels_2d.shape[1] > 1, (
                f"{case_name}: expected multiple targets per graph"
            )

        # Value checks (DummyRegressor returns input as prediction)
        np.testing.assert_allclose(
            labels, preds,
            err_msg=f"Prediction mismatch for {case_name}"
        )
