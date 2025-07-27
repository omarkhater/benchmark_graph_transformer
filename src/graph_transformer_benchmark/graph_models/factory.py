"""
Primary factory for instantiating GNN and GraphTransformer models.
"""
from typing import Any

import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv

from .backbones import ConvBackbone, TransformerBackbone
from .model import GraphModel


def build_model(
        cfg: dict[str, Any],
        num_features: int,
        out_channels: int
        ) -> nn.Module:
    """
    Instantiate a model based on cfg.

    Args:
        cfg: Configuration object with fields:
            - type: 'gcn', 'sage', 'gat', 'gin', or 'graphtransformer'
            - task: 'node' or 'graph'
            - objective: 'classification' or 'regression'
            - hidden_dim: embedding dimension
            - use_batch_norm: bool
            - use_residual: bool
            - encoder_cfg: dict for transformer backbones
            - gnn_cfg: dict for transformer backbones
            - cache_masks, cast_bias: bool for transformer backbones
        num_features: Number of input features per node.
        out_channels: Number of output classes or regression targets.

    Returns:
        nn.Module: Configured model.

    Raises:
        ValueError: If type, task, or objective is unsupported.
    """
    model_type = cfg.get("type", "").lower()
    task = cfg.get("task", "").lower()
    objective = cfg.get("objective", "").lower()

    if model_type not in {"gcn", "sage", "gat", "gin", "graphtransformer"}:
        raise ValueError(f"Unsupported model type: {cfg.get('type')}")
    if task not in {"node", "graph"}:
        raise ValueError(f"Unsupported task type: {cfg.get('task')}")
    if objective not in {"classification", "regression"}:
        raise ValueError(f"Unsupported objective: {cfg.get('objective')}")

    hidden_dim = cfg.get("hidden_dim")

    if hidden_dim is None:
        raise ValueError("`hidden_dim` must be specified in cfg")

    conv_map = {
        "gcn": GCNConv,
        "sage": SAGEConv,
        "gat": GATConv,
        "gin": GINConv
    }
    if model_type in conv_map:
        backbone = ConvBackbone(
            conv_map[model_type],
            num_features,
            hidden_dim
        )
    else:
        backbone = TransformerBackbone(
            cfg,
            num_features,
            hidden_dim
        )

    use_batch_norm = bool(getattr(cfg, "use_batch_norm", False))
    use_residual = bool(getattr(cfg, "use_residual", False))

    return GraphModel(
        backbone=backbone,
        task=task,
        objective=objective,
        out_channels=out_channels,
        use_batch_norm=use_batch_norm,
        use_residual=use_residual
    )
