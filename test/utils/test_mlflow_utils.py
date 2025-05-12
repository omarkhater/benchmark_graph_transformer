import os
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

import graph_transformer_benchmark.utils.mlflow_utils as utils


@pytest.fixture(autouse=True)
def mlflow_env(monkeypatch):
    """Isolate MLflow side effects by stubbing its functions."""
    calls = {
        "uri": None,
        "exp": None,
        "params": {},
        "artifacts": [],
        "metrics": []
    }
    monkeypatch.setattr(
        utils.mlflow,
        "set_tracking_uri",
        lambda uri: calls.update(uri=uri),
    )
    monkeypatch.setattr(
        utils.mlflow,
        "set_experiment",
        lambda name: calls.update(exp=name),
    )
    monkeypatch.setattr(
        utils.mlflow,
        "log_params",
        lambda params: calls["params"].update(params),
    )
    monkeypatch.setattr(
        utils.mlflow,
        "log_artifact",
        lambda path: calls["artifacts"].append(path),
    )
    return calls


def test_init_mlflow_sets_uri_and_experiment(mlflow_env):
    cfg = OmegaConf.create({
        "training": {
            "mlflow": {
                "tracking_uri": "file:///tmp/mlruns",
                "experiment_name": "test_exp",
            }
        }
    })
    utils.init_mlflow(cfg)
    assert mlflow_env["uri"] == "file:///tmp/mlruns"
    assert mlflow_env["exp"] == "test_exp"


def test_log_config_writes_yaml_and_params(tmp_path, mlflow_env):
    cfg = OmegaConf.create({
        "data": {"foo": "bar"},
        "model": {"hidden": 16},
        "training": {"lr": 0.1},
    })
    cwd = Path.cwd()
    os.chdir(tmp_path)
    utils.log_config(cfg)
    expected = {
        "data.foo":     "bar",
        "model.hidden": 16,
        "training.lr":  0.1,
    }
    assert mlflow_env["params"] == expected
    # Assert config.yaml exists and was logged
    assert (tmp_path / "config.yaml").exists()
    assert mlflow_env["artifacts"] == ["config.yaml"]
    os.chdir(cwd)


def test_log_health_metrics_records_grad_and_weight_norms(monkeypatch):
    class Dummy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(4, 2)

        def forward(self, x):
            return self.lin(x)

    model = Dummy()
    inp = torch.randn(1, 4)
    out = model(inp)
    out.sum().backward()

    class Opt:
        param_groups = [{"lr": 0.01}, {"lr": 0.02}]

    optimizer = Opt()
    calls = []
    monkeypatch.setattr(
        utils.mlflow,
        "log_metrics",
        lambda metrics, step=None: calls.append((metrics, step)),
    )
    monkeypatch.setattr(
        utils.mlflow,
        "log_metric",
        lambda key, value, step=None: calls.append((key, value, step)),
    )
    utils.log_health_metrics(model, optimizer, epoch=5)
    dict_calls = [c for c in calls if isinstance(c[0], dict)]
    assert len(dict_calls) == 2
    # Assert learning rates
    tuple_calls = [
        c for c in calls if isinstance(c, tuple) and len(c) == 3
    ]
    lr_calls = [
        c for c in tuple_calls if c[0].startswith("lr/group_")
    ]
    assert lr_calls == [
        ("lr/group_0", 0.01, 5),
        ("lr/group_1", 0.02, 5),
    ]
