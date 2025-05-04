#!/usr/bin/env python
"""
CLI entrypoint for GraphTransformer Benchmark.
Loads configuration via Hydra and invokes the training pipeline.
"""

import warnings

import hydra
from omegaconf import DictConfig

from graph_transformer_benchmark.train import run_training

warnings.filterwarnings(
    "ignore",
    message=r".*torch_geometric\.contrib.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r".*acceptable character detection dependency.*",
)


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="default",
)
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
