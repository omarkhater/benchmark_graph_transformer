from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch_geometric.data import Data

from graph_transformer_benchmark.data import enrich_batch

TaskKind = Literal["multilabel", "multiclass", "regression"]


def create_model(
    model_fn: callable,
    model_cfg: DictConfig,
    sample_batch: Data,
    num_classes: int,
    device: torch.device
) -> nn.Module:
    """Create a model with correct input dimensions and batch enrichment.

    Args:
        model_fn: Function that builds the base model
        model_cfg: Model configuration
        sample_batch: Sample batch to determine feature dimensions
        num_classes: Number of output classes/targets
        device: Device to create model on
    """
    enriched = enrich_batch(sample_batch, model_cfg)
    num_features = enriched.x.size(1) if enriched.x is not None else 0
    base_model = model_fn(model_cfg, num_features, num_classes).to(device)
    return BatchEnrichmentWrapper(base_model, model_cfg, device)


class BatchEnrichmentWrapper(nn.Module):
    """Simple wrapper that enriches batches during forward pass."""

    def __init__(
            self,
            model: nn.Module,
            cfg: DictConfig,
            device: torch.device
            ):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.device = device

    def forward(self, batch: Data):
        """Forward pass with automatic batch enrichment."""
        batch = enrich_batch(batch, self.cfg)
        batch = batch.to(self.device)
        return self.model(batch)


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
