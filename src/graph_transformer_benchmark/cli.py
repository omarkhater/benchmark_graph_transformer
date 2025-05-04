#!/usr/bin/env python
"""
CLI entrypoint for GraphTransformer Benchmark.
Loads configuration via Hydra and invokes the training pipeline.
"""

import warnings

warnings.filterwarnings(
    "ignore",
    message=r".*torch_geometric\.contrib.*",
    category=UserWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r".*acceptable character detection dependency.*",
)
import hydra  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from graph_transformer_benchmark.train import run_training  # noqa: E402


@hydra.main(
    version_base=None,
    config_path="../../configs",
    config_name="default",
)
def main(cfg: DictConfig) -> None:
    try:
        run_training(cfg)
    except Exception:
        return float("inf")


if __name__ == "__main__":
    main()
