"""
Primary factory for instantiating GNN models.
"""
import torch.nn as nn
from omegaconf import DictConfig
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv

from .builders import (
    build_gin_classifier,
    build_gnn_classifier,
    build_node_classifier,
)
from .transformer import NodeGraphTransformer, build_graph_transformer


def build_model(
    cfg: DictConfig,
    num_features: int,
    num_classes: int,
) -> nn.Module:
    """
    Instantiate a GNN or GraphTransformer for either graph or node-level tasks.

    Expects:
      cfg.type: "gcn" | "sage" | "gat" | "gin" | "graphtransformer"
      cfg.task: "graph" | "node"
      plus builder-specific hyperparams under cfg (hidden_dim, dropout, etc.)
    """
    if not hasattr(cfg, "hidden_dim"):
        object.__setattr__(cfg, "hidden_dim", num_features)
    if not hasattr(cfg, "dropout"):
        object.__setattr__(cfg, "dropout", 0.0)
    if not hasattr(cfg, "use_batch_norm"):
        object.__setattr__(cfg, "use_batch_norm", False)
    if not hasattr(cfg, "use_residual"):
        object.__setattr__(cfg, "use_residual", False)

    model_type = cfg.type.lower()
    task = cfg.task.lower()
    use_bn = bool(getattr(cfg, "use_batch_norm", False))
    use_res = bool(getattr(cfg, "use_residual",   False))

    if model_type == "graphtransformer":
        if task == "graph":
            return build_graph_transformer(cfg, num_features, num_classes)
        else:  # node-level
            base = build_graph_transformer(cfg, num_features, num_classes)
            ffn = cfg.ffn_hidden_dim or cfg.hidden_dim
            return NodeGraphTransformer(
                hidden_dim=cfg.hidden_dim,
                num_encoder_layers=cfg.num_layers,
                num_heads=cfg.num_heads,
                dropout=cfg.dropout,
                ffn_hidden_dim=ffn,
                activation=cfg.activation,
                attn_bias_providers=base.attn_bias_providers,
                positional_encoders=base.positional_encoders,
                use_super_node=cfg.use_super_node,
                node_feature_encoder=base.node_feature_encoder,
                gnn_block=base.gnn_block,
                gnn_position=cfg.gnn_position,
                num_node_classes=num_classes,
            )

    conv_map = {"gcn": GCNConv, "sage": SAGEConv, "gat": GATConv}
    if model_type in conv_map:
        if task == "node":
            builder = build_node_classifier
        else:
            builder = build_gnn_classifier
        return builder(
            conv_map[model_type],
            num_features,
            cfg.hidden_dim,
            num_classes,
            use_bn,
            use_res,
        )

    if model_type == "gin":
        if task == "graph":
            return build_gin_classifier(
                num_features, cfg.hidden_dim, num_classes,
                use_bn, use_res
            )
        else:
            def gin_factory(in_c, out_c):
                mlp = nn.Sequential(
                    nn.Linear(in_c, out_c),
                    nn.ReLU(),
                    nn.Linear(out_c, out_c),
                )
                return GINConv(mlp)
            return build_node_classifier(
                gin_factory,
                num_features,
                cfg.hidden_dim,
                num_classes,
                use_bn,
                use_res,
            )
    raise ValueError(
        f"Unsupported combination: type={cfg.type}, task={cfg.task}")
