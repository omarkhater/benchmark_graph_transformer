import random
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from graph_transformer_benchmark.train import (
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
            },
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
    dummy_model, generic_loader, optimizer, device, cfg_data
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
        dummy_model, generic_loader, optimizer, device, cfg_data, epoch=1
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
