"""
Primary factory for instantiating GNN models.
"""
import torch.nn as nn
from torch_geometric.nn import GATConv, GCNConv, GINConv, SAGEConv

from .builders import (
    build_gin_classifier,
    build_gnn_classifier,
    build_node_classifier,
)
from .transformer import GraphLevelGraphTransformer, build_graph_transformer


def build_model(
    cfg: dict,
    num_features: int,
    out_channels: int,
) -> nn.Module:
    """
    Instantiate a GNN or GraphTransformer for either graph or node-level tasks.

    Expects:
      cfg.type: "gcn" | "sage" | "gat" | "gin" | "graphtransformer"
      cfg.task: "graph" | "node"
      plus builder-specific hyperparams under cfg (hidden_dim, dropout, etc.)
    """
    necessary_keys = {
        "hidden_dim": num_features,
        "dropout": 0.0,
        "use_batch_norm": False,
        "use_residual": False
    }

    for key in necessary_keys:
        if key not in cfg:
            cfg.update({key: necessary_keys[key]})

    allowed_model_types = {
        "gcn", "sage", "gat", "gin", "graphtransformer"
    }
    if cfg.get("type") and cfg.get("type").lower() not in allowed_model_types:
        raise ValueError(
            f"Unsupported model type: {cfg.get('type')}. "
            f"Allowed types: {allowed_model_types}"
        )

    allowed_tasks = {"graph", "node"}
    if cfg.get("task") and cfg.get("task").lower() not in allowed_tasks:
        raise ValueError(
            f"Unsupported task type: {cfg.get('task')}. "
            f"Allowed tasks: {allowed_tasks}"
        )
    model_type = cfg.get("type").lower()
    task = cfg.get("task").lower()
    use_bn = bool(getattr(cfg, "use_batch_norm", False))
    use_res = bool(getattr(cfg, "use_residual",   False))

    if model_type == "graphtransformer":
        if task == "graph":
            model = GraphLevelGraphTransformer(
                hidden_dim=cfg.get("hidden_dim"),
                encoder_cfg=cfg.get("encoder_cfg"),
                gnn_cfg=cfg.get("gnn_cfg"),
                out_channels=out_channels
            )
            return model

        model = build_graph_transformer(
            cfg=cfg,
            num_features=num_features,
            out_channels=out_channels
        )
        return model

    conv_map = {"gcn": GCNConv, "sage": SAGEConv, "gat": GATConv}
    if model_type in conv_map:
        if task == "node":
            builder = build_node_classifier
        else:
            builder = build_gnn_classifier
        return builder(
            conv_map[model_type],
            num_features,
            cfg.get("hidden_dim"),
            out_channels,
            use_bn,
            use_res,
        )

    if model_type == "gin":
        if task == "graph":
            return build_gin_classifier(
                num_features,
                cfg.get("hidden_dim"),
                out_channels,
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
                cfg.get("hidden_dim"),
                out_channels,
                use_bn,
                use_res,
            )
    raise ValueError(
        f"Unsupported combination: type={cfg.type}, task={cfg.task}")
