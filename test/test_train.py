from typing import Any

import torch
from omegaconf import OmegaConf

from graph_transformer_benchmark.train import (
    run_training,
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
            "val_frequency": 1,
            "patience": 1,
            "mlflow": {
                    "run_name": None,
                    "description": None,
                    "log_artifacts": log_artifacts,
                }
        },
    })


def test_run_training_logs_metrics(
    tmp_path: Any, patch_training_dependencies: Any
) -> None:
    """run_training should log expected metrics without artifacts."""
    cfg = make_cfg(tmp_path, epochs=2, log_artifacts=False)
    run_training(cfg)

    keys = [metric[0] for metric in patch_training_dependencies.metrics]
    assert keys.count("loss/train/total") == 2
    assert keys.count("val_accuracy") == 1
    assert keys.count("train_accuracy") == 1
    assert patch_training_dependencies.artifacts == ["best_model.pth"]


def test_run_training_artifacts(
    tmp_path: Any,
    patch_training_dependencies: Any,
    monkeypatch: Any,
) -> None:
    """When log_artifacts=True, model.pth should be saved and logged once."""
    cfg = make_cfg(tmp_path, epochs=1, log_artifacts=True)
    # Update mock to handle pickle_module argument
    monkeypatch.setattr(
        torch, "save",
        lambda state, path, **kwargs: open(path, "w").close()
    )
    run_training(cfg)
    assert patch_training_dependencies.artifacts == ["best_model.pth"]
