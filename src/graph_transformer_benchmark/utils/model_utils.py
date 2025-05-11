from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from graph_transformer_benchmark.data import enrich_batch

TaskKind = Literal["multilabel", "multiclass", "regression"]

__all__ = ["infer_task_and_loss", "TaskKind"]


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


def infer_task_and_loss(loader: DataLoader) -> tuple[TaskKind, nn.Module]:
    """
    Inspect the first batch to decide task type + suitable loss.

    Returns
    -------
    task_kind : {'multilabel', 'multiclass', 'regression'}
    loss_fn   : nn.Module
    """
    batch = next(iter(loader))
    y = batch.y
    # Flatten except final dimension (handles node graph label shapes)
    y = y.view(-1, *y.shape[-1:])

    if torch.is_floating_point(y):
        unique = torch.unique(y)
        # ➊ multi-label: float {0,1}, more than one column
        if y.ndim == 2 and unique.le(1).all() and unique.ge(0).all():
            return "multilabel", nn.BCEWithLogitsLoss()
        # ➋ regression: arbitrary floats
        return "regression", nn.MSELoss()
    else:  # integer labels
        # ➌ multi-class single-label
        return "multiclass", nn.CrossEntropyLoss()
