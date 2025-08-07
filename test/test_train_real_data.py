"""Integration tests for training pipeline on real datasets.

Each dataset is a separate class with its own test methods.

For example, `TestMutagGraph` tests the training pipeline on the MUTAG dataset
with different model configurations. Currently, there is a test case for each
conventional GNN type besides a separate case for each graph transformer
variant.

"""

from __future__ import annotations

import pytest
import torch

from graph_transformer_benchmark.train import run_training

MODEL_GNNS = ["GCN", "GAT", "GIN", "SAGE"]
EPOCHS = 2
VAL_FREQUENCY = 1
SEED = 42


def _fake_build_dataloaders(train, val, test):
    def _build(*args, **kwargs):
        return train, val, test
    return _build


class TestMutagGraph:
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "device",
        ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"],
    )
    def test_transformer_variants(
        self,
        monkeypatch,
        base_training_config,
        mutag_graph_dataloaders,
        cfg_transformer,
        device,
    ):
        train_loader, val_loader, test_loader = mutag_graph_dataloaders
        monkeypatch.setattr(
            "graph_transformer_benchmark.train.build_dataloaders",
            _fake_build_dataloaders(train_loader, val_loader, test_loader),
        )

        # cfg_transformer is parametrized over minimal, bias_only, …
        model_cfg = cfg_transformer.copy()
        model_cfg["type"] = "GraphTransformer"
        cfg = base_training_config.copy()
        cfg["model"] = model_cfg
        cfg["training"]["epochs"] = EPOCHS
        cfg["training"]["val_frequency"] = VAL_FREQUENCY
        cfg["training"]["seed"] = SEED
        cfg["training"]["device"] = device

        loss = run_training(cfg)
        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    )
    @pytest.mark.parametrize("model_type", MODEL_GNNS)
    def test_gnn_variants(
        self,
        monkeypatch,
        base_training_config,
        mutag_graph_dataloaders,
        cfg_graph,
        model_type,
        device,
    ):
        train_loader, val_loader, test_loader = mutag_graph_dataloaders
        monkeypatch.setattr(
            "graph_transformer_benchmark.train.build_dataloaders",
            _fake_build_dataloaders(train_loader, val_loader, test_loader),
        )

        # cfg_graph is a simple base dict; we override the type
        model_cfg = cfg_graph.copy()
        model_cfg["type"] = model_type
        cfg = base_training_config.copy()
        cfg["model"] = model_cfg
        cfg["training"]["epochs"] = EPOCHS
        cfg["training"]["val_frequency"] = VAL_FREQUENCY
        cfg["training"]["seed"] = SEED
        cfg["training"]["device"] = device

        loss = run_training(cfg)
        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))


class TestCoraNode:
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "device", ["cuda"] if torch.cuda.is_available() else ["cpu"]
    )
    def test_transformer_variants(
        self,
        monkeypatch,
        base_training_config,
        cora_node_dataloaders,
        cfg_transformer,
        device,
    ):
        train_loader, val_loader, test_loader = cora_node_dataloaders
        monkeypatch.setattr(
            "graph_transformer_benchmark.train.build_dataloaders",
            _fake_build_dataloaders(train_loader, val_loader, test_loader),
        )
        model_cfg = cfg_transformer.copy()
        model_cfg["type"] = "GraphTransformer"
        cfg = base_training_config.copy()
        cfg["model"] = model_cfg
        cfg["training"]["epochs"] = EPOCHS
        cfg["training"]["seed"] = SEED
        cfg["training"]["val_frequency"] = VAL_FREQUENCY
        cfg["training"]["device"] = device
        loss = run_training(cfg)
        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.parametrize("model_type", MODEL_GNNS)
    @pytest.mark.parametrize(
        "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    )
    def test_gnn_variants(
        self,
        monkeypatch,
        base_training_config,
        cora_node_dataloaders,
        cfg_graph,
        model_type,
        device,
    ):
        train_loader, val_loader, test_loader = cora_node_dataloaders
        monkeypatch.setattr(
            "graph_transformer_benchmark.train.build_dataloaders",
            _fake_build_dataloaders(train_loader, val_loader, test_loader),
        )
        model_cfg = cfg_graph.copy()
        model_cfg["type"] = model_type
        cfg = base_training_config.copy()
        cfg["model"] = model_cfg
        cfg["training"]["epochs"] = EPOCHS
        cfg["training"]["seed"] = SEED
        cfg["training"]["val_frequency"] = VAL_FREQUENCY
        cfg["training"]["device"] = device
        loss = run_training(cfg)
        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))


class TestZincGraphRegression:
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    )
    def test_transformer_variants(
        self,
        monkeypatch,
        base_training_config,
        zinc_graph_regression_dataloaders,
        cfg_transformer,
        device,
    ):
        train_loader, val_loader, test_loader = \
            zinc_graph_regression_dataloaders
        monkeypatch.setattr(
            "graph_transformer_benchmark.train.build_dataloaders",
            _fake_build_dataloaders(train_loader, val_loader, test_loader),
        )
        model_cfg = cfg_transformer.copy()
        model_cfg["type"] = "GraphTransformer"
        cfg = base_training_config.copy()
        cfg["model"] = model_cfg
        cfg["training"]["epochs"] = EPOCHS
        cfg["training"]["seed"] = SEED
        cfg["training"]["val_frequency"] = VAL_FREQUENCY
        cfg["training"]["device"] = device
        loss = run_training(cfg)
        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    )
    @pytest.mark.parametrize("model_type", MODEL_GNNS)
    def test_gnn_variants(
        self,
        monkeypatch,
        base_training_config,
        zinc_graph_regression_dataloaders,
        cfg_graph,
        model_type,
        device,
    ):
        train_loader, val_loader, test_loader = \
            zinc_graph_regression_dataloaders
        monkeypatch.setattr(
            "graph_transformer_benchmark.train.build_dataloaders",
            _fake_build_dataloaders(train_loader, val_loader, test_loader),
        )
        model_cfg = cfg_graph.copy()
        model_cfg["type"] = model_type
        cfg = base_training_config.copy()
        cfg["model"] = model_cfg
        cfg["training"]["epochs"] = EPOCHS
        cfg["training"]["seed"] = SEED
        cfg["training"]["val_frequency"] = VAL_FREQUENCY
        cfg["training"]["device"] = device
        loss = run_training(cfg)
        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))


class TestCoraNodeDegreeRegression:
    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.parametrize(
        "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    )
    def test_transformer_variants(
        self,
        monkeypatch,
        base_training_config,
        cora_node_degree_regression_dataloaders,
        cfg_transformer,
        device,
    ):
        train_loader, val_loader, test_loader = (
            cora_node_degree_regression_dataloaders
        )
        monkeypatch.setattr(
            "graph_transformer_benchmark.train.build_dataloaders",
            _fake_build_dataloaders(train_loader, val_loader, test_loader),
        )
        model_cfg = cfg_transformer.copy()
        model_cfg["type"] = "GraphTransformer"
        cfg = base_training_config.copy()
        cfg["model"] = model_cfg
        cfg["training"]["epochs"] = EPOCHS
        cfg["training"]["seed"] = SEED
        cfg["training"]["val_frequency"] = VAL_FREQUENCY
        cfg["training"]["device"] = device
        loss = run_training(cfg)
        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))

    @pytest.mark.slow
    @pytest.mark.integration
    @pytest.mark.parametrize("model_type", MODEL_GNNS)
    @pytest.mark.parametrize(
        "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
    )
    def test_gnn_variants(
        self,
        monkeypatch,
        base_training_config,
        cora_node_degree_regression_dataloaders,
        cfg_graph,
        model_type,
        device,
    ):
        train_loader, val_loader, test_loader = (
            cora_node_degree_regression_dataloaders
        )
        monkeypatch.setattr(
            "graph_transformer_benchmark.train.build_dataloaders",
            _fake_build_dataloaders(train_loader, val_loader, test_loader),
        )
        model_cfg = cfg_graph.copy()
        model_cfg["type"] = model_type
        cfg = base_training_config.copy()
        cfg["model"] = model_cfg
        cfg["training"]["epochs"] = EPOCHS
        cfg["training"]["seed"] = SEED
        cfg["training"]["val_frequency"] = VAL_FREQUENCY
        cfg["training"]["device"] = device
        loss = run_training(cfg)
        assert isinstance(loss, float)
        assert torch.isfinite(torch.tensor(loss))
