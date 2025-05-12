# tests/test_train.py
from __future__ import annotations

from typing import Any

import torch
from omegaconf import OmegaConf

from graph_transformer_benchmark.train import run_training


# --------------------------------------------------------------------------- #
# config helper
# --------------------------------------------------------------------------- #
def _make_cfg(tmp_path, *, epochs: int = 2,
              log_artifacts: bool = False) -> Any:
    return OmegaConf.create(
        {
            "training": {
                "seed": 0,
                "device": "cpu",
                "lr": 0.01,
                "epochs": epochs,
                "val_frequency": 1,
                "patience": 3,
                "log_artifacts": log_artifacts,
            },
            "model": {
                "training": {
                    "mlflow": {
                        "run_name": "pytest-run",
                        "description": "CI unit-test",
                    }
                }
            },
            "data": {"dataset": "MUTAG", "root": str(tmp_path)},
        }
    )


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_run_training_logs_metrics(tmp_path, patch_training_dependencies):
    """Exactly one validation-accuracy metric should be emitted."""
    cfg = _make_cfg(tmp_path, epochs=2, log_artifacts=False)
    run_training(cfg)

    keys: list[str] = [k for k, *_ in patch_training_dependencies.metrics]
    val_acc = [
        k for k in keys
        if k.startswith("val/") and k.endswith("/accuracy")
    ]
    assert len(val_acc) == 1, f"logged keys={keys}"


def test_run_training_artifacts(
    tmp_path, monkeypatch, patch_training_dependencies
):
    """`best_model.pth` must be logged once when artifacts are enabled."""
    monkeypatch.setattr(
        torch, "save", lambda obj, path, **kw: open(path, "w").close()
    )

    cfg = _make_cfg(tmp_path, epochs=1, log_artifacts=True)
    run_training(cfg)

    paths = [p for p in patch_training_dependencies.artifacts if p]
    assert paths == ["best_model.pth"]


def test_return_value_is_finite(tmp_path):
    """`run_training` should return a finite float best-loss."""
    cfg = _make_cfg(tmp_path, epochs=1, log_artifacts=False)
    best = run_training(cfg)
    assert isinstance(best, float) and not torch.isnan(torch.tensor(best))
