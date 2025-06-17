"""Training-related fixtures."""
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import MagicMock

import mlflow
import pytest
import torch
from torch.nn import Module
from torch.optim import SGD
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import graph_transformer_benchmark.train as train_mod


class DummyDataset:
    """Wraps a list of Data objects and exposes length and metadata."""
    def __init__(self, data_list: List[Data]) -> None:
        self._list = data_list
        first = data_list[0]
        self.num_node_features = first.x.size(1)
        # assume y is a 1D or 2D tensor of labels
        y = first.y.view(-1)
        self.num_classes = int(y.max().item()) + 1

    def __len__(self) -> int:
        return len(self._list)

    def __getitem__(self, idx: int) -> Data:
        return self._list[idx]


class DummyRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def device() -> torch.device:
    """Provide a CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def optimizer(dummy_model) -> SGD:
    """Provide an SGD optimizer for the dummy_model."""
    return SGD(
        dummy_model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
    )


@pytest.fixture()
def patch_training_dependencies(
    monkeypatch: Any,
    dummy_model: Module,
    generic_loader: DataLoader,
    device: torch.device,
) -> SimpleNamespace:
    """Stub out data/model/MLflow for run_training tests."""
    data_list = list(generic_loader.dataset)
    dummy_dataset = DummyDataset(data_list)
    dummy_loader = DataLoader(
        dummy_dataset,
        batch_size=generic_loader.batch_size
    )

    monkeypatch.setattr(
        train_mod,
        "build_dataloaders",
        lambda cfg, **kw: (dummy_loader, dummy_loader, dummy_loader)
    )
    monkeypatch.setattr(
        train_mod,
        "build_model",
        lambda cfg_model, nf, nc: dummy_model
    )
    monkeypatch.setattr(train_mod, "get_device", lambda dev: device)
    monkeypatch.setattr(train_mod, "init_mlflow", lambda cfg: None)
    monkeypatch.setattr(train_mod, "log_config", lambda cfg: None)
    monkeypatch.setattr(mlflow, "start_run", lambda **kw: DummyRun())
    metrics = []
    artifacts = []
    monkeypatch.setattr(
        mlflow,
        "log_metric",
        lambda k, v, step=None: metrics.append((k, v, step))
    )
    monkeypatch.setattr(
        mlflow,
        "log_artifact",
        lambda path: artifacts.append(path)
    )

    monkeypatch.setattr(
        train_mod, "log_dataset_stats", lambda *args, **kw: None)
    monkeypatch.setattr(
        train_mod, "infer_num_node_features", lambda loader: 4)
    monkeypatch.setattr(
        train_mod, "infer_num_classes", lambda loader: 2)

    def mock_evaluate(model, loader, device, cfg):
        return {"accuracy": 0.85, "val_loss": 0.25, "macro_f1": 0.80}

    monkeypatch.setattr(
        "graph_transformer_benchmark.evaluation.evaluate",
        mock_evaluate
    )
    monkeypatch.setattr(
        "graph_transformer_benchmark.training."
        "graph_transformer_trainer.evaluate",
        mock_evaluate
    )
    pytorch_mock = MagicMock()
    pytorch_mock.log_model = MagicMock()
    monkeypatch.setattr(mlflow, "pytorch", pytorch_mock)
    monkeypatch.setattr(mlflow, "log_param", lambda k, v: None)
    monkeypatch.setattr(mlflow, "log_params", lambda d: None)
    monkeypatch.setattr(mlflow, "set_tag", lambda k, v: None)
    monkeypatch.setattr(mlflow, "set_tags", lambda d: None)
    monkeypatch.setattr(mlflow, "end_run", lambda: None)
    monkeypatch.setattr(mlflow, "active_run", lambda: MagicMock())

    return SimpleNamespace(metrics=metrics, artifacts=artifacts)
