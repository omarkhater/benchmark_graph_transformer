import random
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import Subset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from graph_transformer_benchmark.train import (
    _infer_num_classes,
    _infer_num_node_features,
    run_training,
    train_one_epoch,
    worker_init_fn,
)


def make_cfg(tmp_path, epochs: int = 3, log_artifacts: bool = False):
    """Construct a minimal DictConfig for run_training."""
    return OmegaConf.create({
        "data": {
            "dataset": "MUTAG",
            "root": str(tmp_path),
            "batch_size": 1,
            "num_workers": 0,
        },
        "model": {
            "type": "GCN",
            "hidden_dim": 4,
            "training": {
                "mlflow": {
                    "run_name": None,
                    "description": None,
                    "log_artifacts": log_artifacts,
                }
            },
        },
        "training": {
            "seed": 0,
            "device": "cpu",
            "epochs": epochs,
            "lr": 0.01,
            "mlflow": {
                    "run_name": None,
                    "description": None,
                    "log_artifacts": log_artifacts,
                }
        },
    })


def test_worker_init_fn_reproducibility():
    """worker_init_fn should seed Python random and NumPy deterministically."""
    # Capture sequences for worker 0 and worker 1
    seeds = []
    for wid in (0, 1):
        # Reset to unknown state
        random.seed(None)
        np.random.seed(None)
        worker_init_fn(wid)
        # Draw a couple of numbers
        seeds.append((random.random(), np.random.rand()))

    # Re-run and assert same outputs
    for wid, (r0, n0) in zip((0, 1), seeds):
        random.seed(None)
        np.random.seed(None)
        worker_init_fn(wid)
        assert random.random() == pytest.approx(r0)
        assert np.random.rand() == pytest.approx(n0)


def test_train_one_epoch_decreases_loss(
    dummy_model, generic_loader, optimizer, device
):
    # Compute loss before any training
    dummy_model.eval()
    with torch.no_grad():
        initial_logits = dummy_model(next(iter(generic_loader)))
        initial_loss = F.cross_entropy(
            initial_logits, next(iter(generic_loader)).y.view(-1)
        ).item()

    # Perform one epoch of training
    avg_loss = train_one_epoch(
        dummy_model, generic_loader, optimizer, device, epoch=1
    )

    # Loss after training should be <= initial loss
    assert avg_loss <= initial_loss, "Training did not decrease loss"
    assert avg_loss >= 0.0, "Loss should never be negative"


def test_run_training_logs_metrics(
    tmp_path: Any, patch_training_dependencies: Any
) -> None:
    """run_training should log expected metrics without artifacts."""
    cfg = make_cfg(tmp_path, epochs=2, log_artifacts=False)
    run_training(cfg)

    keys = [metric[0] for metric in patch_training_dependencies.metrics]
    assert keys.count("train_loss") == 2
    assert keys.count("val_acc") == 2
    assert keys.count("test_acc") == 1
    assert patch_training_dependencies.artifacts == []


def test_run_training_artifacts(
    tmp_path: Any,
    patch_training_dependencies: Any,
    monkeypatch: Any,
) -> None:
    """When log_artifacts=True, model.pth should be saved and logged once."""
    cfg = make_cfg(tmp_path, epochs=1, log_artifacts=True)
    monkeypatch.setattr(
        torch, "save", lambda state, path: open(path, "w").close()
    )
    run_training(cfg)
    assert patch_training_dependencies.artifacts == ["model.pth"]

# ----------------------------------------------------------------------
# Tests for _infer_num_node_features
# ----------------------------------------------------------------------


class DummyDataset(list):
    """Simple dataset wrapping a list with metadata attributes"""


def test_infer_num_node_features_from_dataset_attr():
    data = Data(x=torch.randn(2, 7), y=torch.tensor([0, 1]))
    ds = DummyDataset([data])
    ds.num_node_features = 7
    loader = DataLoader(ds, batch_size=1)
    assert _infer_num_node_features(loader) == 7


def test_infer_num_node_features_from_subset_attr():
    data = Data(x=torch.randn(3, 5), y=torch.tensor([0, 1, 2]))
    ds = DummyDataset([data])
    ds.num_node_features = 5
    subset = Subset(ds, [0])
    loader = DataLoader(subset, batch_size=1)
    assert _infer_num_node_features(loader) == 5


def test_infer_num_node_features_from_batch():
    data = Data(x=torch.randn(4, 9), y=torch.tensor([0, 1, 2, 3]))
    loader = DataLoader([data], batch_size=1)
    assert _infer_num_node_features(loader) == 9


# ----------------------------------------------------------------------
# Tests for _infer_num_classes
# ----------------------------------------------------------------------

def test_infer_num_classes_from_dataset_attr():
    data = Data(x=torch.randn(2, 4), y=torch.tensor([0, 1]))
    ds = DummyDataset([data])
    ds.num_classes = 4
    loader = DataLoader(ds, batch_size=1)
    assert _infer_num_classes(loader) == 4


def test_infer_num_classes_from_subset_attr():
    data = Data(x=torch.randn(3, 3), y=torch.tensor([0, 2, 1]))
    ds = DummyDataset([data])
    ds.num_classes = 3
    subset = Subset(ds, [0])
    loader = DataLoader(subset, batch_size=1)
    assert _infer_num_classes(loader) == 3


def test_infer_num_classes_from_batch_scalar_labels():
    data = Data(x=torch.randn(2, 6), y=torch.tensor([0, 2]))
    loader = DataLoader([data], batch_size=1)
    assert _infer_num_classes(loader) == 3


def test_infer_num_classes_from_batch_one_hot_labels():
    one_hot = torch.eye(5)
    data = Data(x=torch.randn(5, 5), y=one_hot)
    loader = DataLoader([data], batch_size=1)
    assert _infer_num_classes(loader) == 5
