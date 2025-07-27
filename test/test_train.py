"""Unit tests for the training pipeline using different dataset scenarios."""

from __future__ import annotations

import pytest
import torch

from graph_transformer_benchmark.train import run_training


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("case_name, task_type", [
    ("node_binary", "node"),
    ("node_multiclass", "node"),
    ("graph_binary", "graph"),
    ("graph_multiclass", "graph"),
    ("edge_attr", "graph"),
    ("sparse_features", "node"),
    ("masked_nodes", "node"),
    ("pyg_style", "node"),
    ("subset_with_parent", "node")
])
def test_classification_pipeline(
    monkeypatch,
    base_training_config,
    graph_classification_suite,
    case_name: str,
    task_type: str
):
    """Test training pipeline with different classification scenarios.

    Args:
        case_name: Name of the classification test case from suite
        monkeypatch: Pytest fixture for mocking
        base_training_config: Base configuration fixture
        graph_classification_suite: Suite of classification test cases
    """
    loader = graph_classification_suite[case_name]
    dataloaders = (loader, loader, loader)
    train_cfg = base_training_config.copy()
    model_cfg = train_cfg.get("model", {})
    model_cfg.update(
        {
            "task": task_type
        }
    )
    with monkeypatch.context() as m:
        # Mock the build_dataloaders function to return our test loaders
        m.setattr(
            "graph_transformer_benchmark.train.build_dataloaders",
            lambda *args, **kwargs: dataloaders
        )

        loss = run_training(train_cfg)
        print(f"Obtained loss: {loss:.4f}")

        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("case_name", [
    "single_target",     # ZINC-like single property
    "multi_target",      # QM9-like multiple properties
    "edge_attr",         # With edge features (e.g. distances)
    "no_features",       # QM7b-like no node features
    "varied_graphs",     # Different size graphs
    "pyg_style",         # PyG dataset style
    "subset_with_parent"  # Dataset subset testing
])
def test_regression_pipeline(
    monkeypatch,
    base_training_config,
    graph_regression_suite,
    case_name: str,
):
    """Test training pipeline with different regression scenarios.

    Args:
        case_name: Name of the regression test case from suite
        monkeypatch: Pytest fixture for mocking
        base_training_config: Base configuration fixture
        graph_regression_suite: Suite of regression test cases
    """
    loader = graph_regression_suite[case_name]
    dataloaders = (loader, loader, loader)

    with monkeypatch.context() as m:
        # Mock the build_dataloaders function to return our test loaders
        m.setattr(
            "graph_transformer_benchmark.train.build_dataloaders",
            lambda *args, **kwargs: dataloaders
        )

        loss = run_training(base_training_config)

        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))
