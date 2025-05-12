from omegaconf import OmegaConf

from graph_transformer_benchmark.utils import flatten_cfg


def test_flatten_cfg_nested_and_scalars():
    cfg = OmegaConf.create({
        "section": {
            "a": 1,
            "b": {"x": 2, "y": 3},
            "c": 4,
        }
    })
    flat = flatten_cfg(cfg.section, prefix="sec.")
    assert flat == {
        "sec.a": 1,
        "sec.b.x": 2,
        "sec.b.y": 3,
        "sec.c": 4,
    }
