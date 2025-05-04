# tests/conftest.py

from typing import Any

import pytest
import torch
from omegaconf import OmegaConf
from torch import Tensor
from torch.nn import Module
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import graph_transformer_benchmark.evaluate as eval_mod


class DummyGraphEvaluator:
    """Dummy evaluator for graph-level OGB tasks."""

    def __init__(self, name: str) -> None:
        self.name = name

    def eval(self, inputs: dict) -> dict:
        return {"rocauc": 0.75}


class DummyNodeEvaluator:
    """Dummy evaluator for node-level OGB tasks."""

    def __init__(self, name: str) -> None:
        self.name = name

    def eval(self, inputs: dict) -> dict:
        return {"acc": 0.5}


class DummyModel(Module):
    """Model that returns perfect one-hot logits based on batch.y."""

    def forward(self, batch: Data) -> Tensor:
        labels = batch.y.view(-1)
        num_classes = int(labels.max().item()) + 1
        logits = torch.zeros(
            (labels.size(0), num_classes), device=labels.device)
        logits[torch.arange(labels.size(0), device=labels.device),
               labels] = 1.0
        return logits


@pytest.fixture(autouse=True)
def patch_ogb_evaluators(monkeypatch: Any) -> None:
    """Patch OGB evaluators with dummy implementations."""
    monkeypatch.setattr(eval_mod, "GraphEvaluator", DummyGraphEvaluator)
    monkeypatch.setattr(eval_mod, "NodeEvaluator", DummyNodeEvaluator)
    yield


@pytest.fixture
def dummy_model() -> DummyModel:
    """Provide a dummy model fixture."""
    return DummyModel()


@pytest.fixture
def graph_loader() -> DataLoader:
    """Provide DataLoader for a graph-level batch of size 2."""
    d0 = Data(x=torch.randn(1, 4), y=torch.tensor([[0]]))
    d1 = Data(x=torch.randn(1, 4), y=torch.tensor([[1]]))
    return DataLoader([d0, d1], batch_size=2)


@pytest.fixture
def node_loader() -> DataLoader:
    """Provide DataLoader for a node-level batch of one graph."""
    labels = torch.tensor([0, 1, 0, 1]).unsqueeze(1)
    data = Data(x=torch.randn(4, 4), y=labels)
    return DataLoader([data], batch_size=1)


@pytest.fixture
def generic_loader() -> DataLoader:
    """Provide DataLoader for generic graph classification."""
    g0 = Data(x=torch.randn(1, 4), y=torch.tensor([[0]]))
    g1 = Data(x=torch.randn(1, 4), y=torch.tensor([[1]]))
    return DataLoader([g0, g1], batch_size=2)


@pytest.fixture
def cfg_graph() -> Any:
    """Provide OmegaConf config for graph-level task."""
    return OmegaConf.create({"data": {"dataset": "ogbg-molhiv"}})


@pytest.fixture
def cfg_node() -> Any:
    """Provide OmegaConf config for node-level task."""
    return OmegaConf.create({"data": {"dataset": "ogbn-arxiv"}})


@pytest.fixture
def cfg_generic() -> Any:
    """Provide OmegaConf config for generic task."""
    return OmegaConf.create({"data": {"dataset": "MUTAG"}})


@pytest.fixture
def cfg_unsupported() -> Any:
    """Provide OmegaConf config for unsupported dataset."""
    return OmegaConf.create({"data": {"dataset": "unknown"}})
