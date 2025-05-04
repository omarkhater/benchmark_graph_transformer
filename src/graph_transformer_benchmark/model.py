#!/usr/bin/env python
"""
Model module: build GraphTransformer from configuration.
"""
from typing import Callable, List, Optional

import torch.nn as nn
from omegaconf import DictConfig
from torch_geometric.contrib.nn.bias import (
    GraphAttnEdgeBias,
    GraphAttnHopBias,
    GraphAttnSpatialBias,
)
from torch_geometric.contrib.nn.models import GraphTransformer
from torch_geometric.contrib.nn.positional import (
    DegreeEncoder,
    EigEncoder,
    SVDEncoder,
)
from torch_geometric.nn import GATConv, GCNConv, SAGEConv


def build_model(
    cfg_model: DictConfig,
    num_features: int,
    num_classes: int,
) -> GraphTransformer:
    """
    Construct a GraphTransformer based on model config.

    Args:
        cfg_model (DictConfig): Model hyperparameters.
        num_features (int): Number of input node features.
        num_classes (int): Number of target classes.

    Returns:
        GraphTransformer: Instantiated model.
    """
    # Node feature encoder
    encoder = nn.Sequential(
        nn.Linear(num_features, cfg_model.hidden_dim),
        nn.ReLU(),
        nn.LayerNorm(cfg_model.hidden_dim),
        nn.Dropout(cfg_model.dropout),
    )

    # Attention bias providers
    bias_providers: List[Callable] = []
    if cfg_model.with_spatial_bias:
        bias_providers.append(
            GraphAttnSpatialBias(
                num_heads=cfg_model.num_heads,
                num_spatial=cfg_model.num_spatial,
                use_super_node=cfg_model.use_super_node,
            )
        )
    if cfg_model.with_edge_bias:
        bias_providers.append(
            GraphAttnEdgeBias(
                num_heads=cfg_model.num_heads,
                num_edges=cfg_model.num_edges,
                use_super_node=cfg_model.use_super_node,
            )
        )
    if cfg_model.with_hop_bias:
        bias_providers.append(
            GraphAttnHopBias(
                num_heads=cfg_model.num_heads,
                num_hops=cfg_model.num_hops,
                use_super_node=cfg_model.use_super_node,
            )
        )

    # Positional encoders
    pos_encoders: List[Callable] = []
    if cfg_model.with_degree_enc:
        pos_encoders.append(
            DegreeEncoder(
                num_in=cfg_model.max_degree,
                num_out=cfg_model.max_degree,
                hidden_dim=cfg_model.hidden_dim,
            )
        )
    if cfg_model.with_eig_enc:
        pos_encoders.append(
            EigEncoder(
                num_eigvec=cfg_model.num_eigenc,
                hidden_dim=cfg_model.hidden_dim,
            )
        )
    if cfg_model.with_svd_enc:
        pos_encoders.append(
            SVDEncoder(
                r=cfg_model.num_svdenc,
                hidden_dim=cfg_model.hidden_dim,
            )
        )

    # Optional GNN hook
    gnn_block: Optional[Callable] = None
    if cfg_model.gnn_conv_type:
        conv_map = {
            "gcn": GCNConv,
            "sage": SAGEConv,
            "gat": GATConv,
        }
        ConvClass = conv_map[cfg_model.gnn_conv_type]
        conv_layer = ConvClass(cfg_model.hidden_dim, cfg_model.hidden_dim)

        def gnn_hook(data, x):  # noqa: E731
            return conv_layer(x, data.edge_index)

        gnn_block = gnn_hook

    return GraphTransformer(
        hidden_dim=cfg_model.hidden_dim,
        num_class=num_classes,
        use_super_node=cfg_model.use_super_node,
        node_feature_encoder=encoder,
        num_encoder_layers=cfg_model.num_layers,
        num_heads=cfg_model.num_heads,
        dropout=cfg_model.dropout,
        ffn_hidden_dim=cfg_model.ffn_hidden_dim,
        activation=cfg_model.activation,
        attn_bias_providers=bias_providers,
        positional_encoders=pos_encoders,
        gnn_block=gnn_block,
        gnn_position=cfg_model.gnn_position,
    )
