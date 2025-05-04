#!/usr/bin/env python
"""
Model definitions for GraphTransformer Benchmark.
"""
from typing import Callable, List, Optional

import torch
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
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_mean_pool


class GNNClassifier(nn.Module):
    """Wrap a single GNN conv layer for whole-graph classification.

    Applies one convolution, global mean pooling, then a linear head.
    """

    def __init__(
        self,
        conv_cls: Callable,
        in_channels: int,
        hidden_dim: int,
        num_classes: int,
    ) -> None:
        """
        Args:
            conv_cls (Callable): GNN convolution class (e.g. GCNConv).
            in_channels (int): Size of input node feature vectors.
            hidden_dim (int): Hidden dimension for convolution and pool.
            num_classes (int): Number of target classes.
        """
        super().__init__()
        self.conv = conv_cls(in_channels, hidden_dim)
        self.lin = nn.Linear(hidden_dim, num_classes)

    def forward(self, data: torch.Tensor) -> torch.Tensor:  # type: ignore
        """Compute logits for a batch of graphs.

        Args:
            data: A torch_geometric.data.Data object with attributes:
                - x: [N, in_channels] node features
                - edge_index: [2, E] edge list
                - batch: [N] batch assignment vector

        Returns:
            Tensor: [batch_size, num_classes] raw logits per graph.
        """
        x = self.conv(data.x, data.edge_index)
        x = global_mean_pool(x, data.batch)
        return self.lin(x)


def _build_graph_transformer(
    cfg_model: DictConfig, num_features: int, num_classes: int
) -> GraphTransformer:
    """Build a GraphTransformer with bias and positional encoders.

    Args:
        cfg_model (DictConfig): GraphTransformer-specific hyperparameters.
        num_features (int): Dimensionality of input node features.
        num_classes (int): Number of output classes.

    Returns:
        GraphTransformer: Configured and uninitialized model.
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

    gnn_block: Optional[Callable] = None
    if cfg_model.gnn_conv_type:
        conv_map = {"gcn": GCNConv, "sage": SAGEConv, "gat": GATConv}
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


def build_model(
    cfg_model: DictConfig, num_features: int, num_classes: int
) -> nn.Module:
    """Instantiate the model specified in the config.

    Supports GraphTransformer and GCN/SAGE/GAT wrapped by GNNClassifier.

    Args:
        cfg_model (DictConfig): Includes field `type` specifying model.
        num_features (int): Input node feature dimension.
        num_classes (int): Number of output classes.

    Returns:
        nn.Module: The requested model instance.

    Raises:
        ValueError: If cfg_model.type is not recognized.
    """
    model_type = cfg_model.type
    if model_type == "GraphTransformer":
        return _build_graph_transformer(cfg_model, num_features, num_classes)
    if model_type == "GCN":
        return GNNClassifier(
            GCNConv,
            num_features,
            cfg_model.hidden_dim,
            num_classes
        )
    if model_type == "SAGE":
        return GNNClassifier(
            SAGEConv,
            num_features,
            cfg_model.hidden_dim,
            num_classes
        )
    if model_type == "GAT":
        return GNNClassifier(
            GATConv,
            num_features,
            cfg_model.hidden_dim,
            num_classes
        )

    raise ValueError(f"Unsupported model type: {model_type}")
