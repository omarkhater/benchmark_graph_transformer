"""Unit tests for the training pipeline using different dataset scenarios."""

from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from graph_transformer_benchmark.train import run_training


@pytest.fixture
def base_training_config(tmp_path):
    """Create base training configuration for tests."""
    return OmegaConf.create({
        "model": {
            "type": "graphtransformer",
            "task": "graph",
            "hidden_dim": 64,
            "num_layers": 4,
            "num_heads": 8,
            "dropout": 0.1,
            "ffn_hidden_dim": 128,
            "activation": "relu",
            "with_spatial_bias": True,
            "with_edge_bias": True,
            "gnn_conv_type": "gcn",
            "gnn_position": "post",
            "training": {
                "mlflow": {
                    "tracking_uri": str(tmp_path / "mlruns"),
                    "experiment": "test-experiment",
                    "run_name": "pytest-run",
                    "description": "CI unit-test",
                }
            }
        },
        "training": {
            "seed": 42,
            "epochs": 3,
            "batch_size": 32,
            "lr": 0.001,
            "val_frequency": 1,
            "patience": 2,
            "device": "cpu",
            "log_artifacts": False
        },
        "data": {
            "name": "test_dataset",
            "task": "graph",
            "split_seed": 42
        }
    })


@pytest.mark.integration
@pytest.mark.parametrize("case_name", [
    "node_binary",
    "node_multiclass",
    "graph_binary",
    "graph_multiclass",
    "edge_attr",
    "sparse_features",
    "masked_nodes",
    "pyg_style",
    "subset_with_parent"
])
def test_classification_pipeline(
    monkeypatch,
    base_training_config,
    graph_classification_suite,
    case_name: str
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

    with monkeypatch.context() as m:
        # Mock the build_dataloaders function to return our test loaders
        m.setattr(
            "graph_transformer_benchmark.train.build_dataloaders",
            lambda *args, **kwargs: dataloaders
        )

        loss = run_training(base_training_config)
        print(f"Obtained loss: {loss:.4f}")

        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))
