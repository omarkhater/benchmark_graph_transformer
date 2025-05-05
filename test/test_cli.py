from omegaconf import DictConfig

import graph_transformer_benchmark.cli as cli_mod


def test_main_invokes_run_training(monkeypatch):
    """Verify that cli.main(cfg) calls run_training(cfg) exactly once."""
    seen = {}

    def fake_run_training(cfg):
        # record that we got called with the same cfg instance
        seen['cfg'] = cfg

    # patch out the real run_training
    monkeypatch.setattr(cli_mod, "run_training", fake_run_training)

    # build a dummy DictConfig
    dummy = DictConfig({"some": "value", "nested": {"x": 42}})

    # the hydra decorator wraps your function;
    # the original is in main.__wrapped__
    cli_mod.main.__wrapped__(dummy)

    # assert we actually called fake_run_training with our dummy config
    assert "cfg" in seen
    assert seen["cfg"] is dummy
