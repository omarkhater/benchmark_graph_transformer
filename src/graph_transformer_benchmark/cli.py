#!/usr/bin/env python
"""
CLI entrypoint for GraphTransformer Benchmark.
Loads configuration via Hydra and invokes the training pipeline.
"""

import warnings

import hydra
from omegaconf import DictConfig

from graph_transformer_benchmark.train import run_training

# ——————————————————————————————————————————————————————————————————————
# Suppress the “experimental code” UserWarning from torch_geometric.contrib
warnings.filterwarnings(
    "ignore",
    message=".*experimental code and is subject to change.*",
    category=UserWarning,
    module="torch_geometric\\.contrib.*",
)
# ——————————————————————————————————————————————————————————————————————


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="default",
)
def main(cfg: DictConfig) -> None:
    """
    Hydra entrypoint.

    Args:
        cfg (DictConfig): Merged configuration from configs/default.yaml.
    """
    run_training(cfg)


if __name__ == "__main__":
    main()
