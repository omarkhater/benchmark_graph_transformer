#!/usr/bin/env python
"""
Utility functions for MLflow setup and config logging.
"""
import logging
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import mlflow
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from graph_transformer_benchmark.data import enrich_batch


def init_mlflow(cfg: DictConfig) -> None:
    """
    Initialize MLflow using tracking URI and experiment name.

    Args:
        cfg (DictConfig): Contains `training.mlflow` settings.
    """
    mlflow.set_tracking_uri(cfg.training.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.training.mlflow.experiment_name)


def _flatten_cfg(section: DictConfig | Dict[str, Any],
                 prefix: str = "") -> Dict[str, Any]:
    """
    Recursively flatten a DictConfig (or plain dict).

    Each leaf key becomes "<prefix>key[.subkey...]" so that
    all parameter names are globally unique.
    """
    flat = {}
    for k, v in section.items():
        full_key = f"{prefix}{k}"
        if isinstance(v, (DictConfig, dict)):
            flat.update(_flatten_cfg(v, prefix=f"{full_key}."))
        else:
            if hasattr(v, "item"):
                v = v.item()
            flat[full_key] = v
    return flat


def log_config(cfg: DictConfig) -> None:
    """
    Log *all* run‑time configuration parameters to MLflow with
    unique keys and store the merged config file as an artifact.

    Args:
        cfg (DictConfig): Full merged run configuration.
    """
    # 1. Build a single flattened param dict
    params = {}
    for section_name in ("data", "model", "training"):
        if section_name in cfg:
            params.update(_flatten_cfg(cfg[section_name],
                                       prefix=f"{section_name}."))

    mlflow.log_params(params)

    config_path = Path("config.yaml")
    OmegaConf.save(config=cfg, f=config_path)
    mlflow.log_artifact(str(config_path))


def set_seed(seed: int) -> None:
    """
    Seed all relevant RNGs for reproducibility.

    Args:
        seed (int): The seed to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Make CuDNN deterministic (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def get_device(preferred: str) -> torch.device:
    """
    Return the compute device based on availability and preference.

    Args:
        preferred (str): 'cuda' or 'cpu'.

    Returns:
        torch.device: Selected device.
    """
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def log_health_metrics(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
) -> None:
    """
    Log gradient norms, weight norms, and learning rates.

    Args:
        model (nn.Module): Trained model.
        optimizer (Optimizer): Optimizer.
        epoch (int): Current epoch number.
    """
    grad_norms: Dict[str, float] = {}
    weight_norms: Dict[str, float] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[f"grad_norm/{name}"] = param.grad.norm().item()
        weight_norms[f"weight_norm/{name}"] = param.data.norm().item()

    mlflow.log_metrics(grad_norms, step=epoch)
    mlflow.log_metrics(weight_norms, step=epoch)
    for idx, group in enumerate(optimizer.param_groups):
        lr = group["lr"]
        mlflow.log_metric(f"lr/group_{idx}", lr, step=epoch)


class BatchEnrichedModel(nn.Module):
    """Wraps a GraphTransformer so every batch is first enriched."""
    def __init__(self,
                 base_model: nn.Module,
                 model_cfg: DictConfig) -> None:
        super().__init__()
        self.base_model = base_model
        self.model_cfg = model_cfg

    def forward(self, batch: Data):
        batch = enrich_batch(batch, self.model_cfg)
        return self.base_model(batch)


def build_run_name(cfg: DictConfig) -> str:
    """
    Build a descriptive MLflow run name based on the model config.

    Args:
        cfg (DictConfig): Hydra-configured run settings, including
            `cfg.model` with flags for biases, encoders, and GNN.

    Returns:
        str: A run name such as
            "GraphTransformer-Bias-Spatial-Edge-Hop",
            "GraphTransformer-Encoding-Degree-Eig-SVD",
            or "GraphTransformer-GNN-GCN-Pre".
    """
    parts: list[str] = [cfg.model.type]

    # Collect any attention-bias components
    bias_components: list[str] = []
    if cfg.model.get("with_spatial_bias", False):
        bias_components.append("Spatial")
    if cfg.model.get("with_edge_bias", False):
        bias_components.append("Edge")
    if cfg.model.get("with_hop_bias", False):
        bias_components.append("Hop")
    if bias_components:
        parts.append("Bias-" + "-".join(bias_components))

    # Collect any positional-encoding components
    enc_components: list[str] = []
    if cfg.model.get("with_degree_enc", False):
        enc_components.append("Degree")
    if cfg.model.get("with_eig_enc", False):
        enc_components.append("Eig")
    if cfg.model.get("with_svd_enc", False):
        enc_components.append("SVD")
    if enc_components:
        parts.append("Encoding-" + "-".join(enc_components))

    # Append GNN block info if configured
    gnn_conv = cfg.model.get("gnn_conv_type")
    if gnn_conv:
        position = cfg.model.get("gnn_position", "pre").capitalize()
        parts.append(f"GNN-{gnn_conv.upper()}-{position}")

    # Default to "Vanilla" when no special features are enabled
    if len(parts) == 1:
        parts.append("Vanilla")

    return "-".join(parts)


def log_dataset_stats(  # noqa: D401  ← flake8 pydocstyle (short-summary rule)
    loader: DataLoader,
    split_name: str = "train",
    *,
    log_to_mlflow: bool = False,
) -> None:
    """Log basic structural statistics of a graph DataLoader.

    Computes per-graph node / edge counts and prints:
      • #Graphs
      • Avg / Min / Max #Nodes
      • Avg / Min / Max #Edges
      • Mode node-count (helpful to spot fixed-size batches)

    Args
    ----
    loader:
        PyG‐style ``DataLoader`` that yields ``torch_geometric.data.Data``.
        The underlying ``dataset`` may be a ``Subset`` or any object that
        supports ``__len__`` and ``__getitem__``.
    split_name:
        Friendly name that appears in the log line (e.g. ``"train"``).
    log_to_mlflow:
        If *True*, the numbers are also recorded as MLflow parameters with
        keys like ``train_num_graphs`` and ``val_avg_nodes``.
    """
    ds = loader.dataset
    num_graphs = len(ds)

    # Gather node/edge counts -------------------------------------------------
    n_nodes: List[int] = []
    n_edges: List[int] = []
    for i in range(num_graphs):
        g = ds[i]
        n_nodes.append(int(g.num_nodes))
        n_edges.append(int(g.edge_index.size(1)))

    def _basic_stats(vals: List[int]) -> Tuple[float, int, int]:
        return (statistics.mean(vals), min(vals), max(vals))

    avg_n, min_n, max_n = _basic_stats(n_nodes)
    avg_e, min_e, max_e = _basic_stats(n_edges)
    mode_n = Counter(n_nodes).most_common(1)[0][0]

    msg = (
        f"[{split_name.upper():5}] "
        f"#Graphs={num_graphs:<5d} "
        f"Nodes (avg/min/max)={avg_n:.1f}/{min_n}/{max_n} "
        f"Edges (avg/min/max)={avg_e:.1f}/{min_e}/{max_e} "
        f"Mode Nodes={mode_n}"
    )
    logging.info(msg)

    if log_to_mlflow:  # ── optional experiment tracking ────────────────────
        import mlflow
        prefix = f"{split_name}_"
        mlflow.log_params(
            {
                f"{prefix}num_graphs": num_graphs,
                f"{prefix}avg_nodes": round(avg_n, 2),
                f"{prefix}min_nodes": min_n,
                f"{prefix}max_nodes": max_n,
                f"{prefix}avg_edges": round(avg_e, 2),
                f"{prefix}min_edges": min_e,
                f"{prefix}max_edges": max_e,
            }
        )
