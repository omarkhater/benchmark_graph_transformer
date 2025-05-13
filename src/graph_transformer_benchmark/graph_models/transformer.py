"""
GraphTransformer builders and node-adapted subclass.
"""
from typing import Callable, Optional, Sequence

import torch.nn as nn
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import ModuleList
from torch_geometric.contrib.nn.bias import (
    BaseBiasProvider,
    GraphAttnEdgeBias,
    GraphAttnHopBias,
    GraphAttnSpatialBias,
)
from torch_geometric.contrib.nn.models import GraphTransformer
from torch_geometric.contrib.nn.positional import (
    BasePositionalEncoder,
    DegreeEncoder,
    EigEncoder,
    SVDEncoder,
)
from torch_geometric.contrib.utils.mask_utils import build_key_padding
from torch_geometric.data import Data


def build_graph_transformer(
    cfg: DictConfig,
    num_features: int,
    num_classes: int,
) -> GraphTransformer:
    """
    Build a graph-level GraphTransformer with configurable biases and encoders.

    Args:
        cfg (DictConfig): Model hyperparameters and flags.
        num_features (int): Dimensionality of input node features.
        num_classes (int): Number of graph classes.

    Returns:
        GraphTransformer: Instantiated transformer model.
    """
    encoder = nn.Sequential(
        nn.Linear(num_features, cfg.hidden_dim),
        nn.ReLU(),
        nn.LayerNorm(cfg.hidden_dim),
        nn.Dropout(cfg.dropout),
    )
    biases = []
    if cfg.get("with_spatial_bias", None):
        biases.append(
            GraphAttnSpatialBias(
                num_heads=cfg.num_heads,
                num_spatial=cfg.num_spatial,
                use_super_node=cfg.use_super_node,
            )
        )
    if cfg.get("with_edge_bias", None):
        biases.append(
            GraphAttnEdgeBias(
                num_heads=cfg.num_heads,
                num_edges=cfg.num_edges,
                use_super_node=cfg.use_super_node,
            )
        )
    if cfg.get("with_hop_bias", None):
        biases.append(
            GraphAttnHopBias(
                num_heads=cfg.num_heads,
                num_hops=cfg.num_hops,
                use_super_node=cfg.use_super_node,
            )
        )

    encoders = []
    if cfg.get("with_degree_enc", None):
        encoders.append(
            DegreeEncoder(cfg.max_degree, cfg.max_degree, cfg.hidden_dim))
    if cfg.get("with_eig_enc", None):
        encoders.append(EigEncoder(cfg.num_eigenc, cfg.hidden_dim))
    if cfg.get("with_svd_enc", None):
        encoders.append(SVDEncoder(cfg.num_svdenc, cfg.hidden_dim))

    gnn_block = None
    if cfg.get("gnn_conv_type", None) is not None:
        from torch_geometric.nn import GATConv, GCNConv, SAGEConv
        conv_map = {"gcn": GCNConv, "sage": SAGEConv, "gat": GATConv}
        layer = conv_map[cfg.gnn_conv_type](cfg.hidden_dim, cfg.hidden_dim)

        def _hook(data: Data, x: Tensor) -> Tensor:
            return layer.to(x.device)(x, data.edge_index)

        gnn_block = _hook

    return GraphTransformer(
        hidden_dim=cfg.hidden_dim,
        num_class=num_classes,
        use_super_node=cfg.get("use_super_node", False),
        node_feature_encoder=encoder,
        num_encoder_layers=cfg.get("num_layers", 1),
        num_heads=cfg.get("num_heads", None),
        dropout=cfg.dropout,
        ffn_hidden_dim=cfg.get("ffn_hidden_dim", None),
        activation=cfg.get("activation", "gelu"),
        attn_bias_providers=ModuleList(biases),
        positional_encoders=encoders,
        gnn_block=gnn_block,
        gnn_position=cfg.get("gnn_position", "pre"),
    )


class NodeGraphTransformer(GraphTransformer):
    """
    GraphTransformer adapted for node classification.

    Removes global pooling and adds a per-node linear head.
    """

    def __init__(
        self,
        *,
        hidden_dim: int,
        num_encoder_layers: int,
        num_heads: int,
        dropout: float,
        ffn_hidden_dim: int,
        activation: str,
        attn_bias_providers: Sequence[BaseBiasProvider],
        positional_encoders: Sequence[BasePositionalEncoder],
        use_super_node: bool,
        node_feature_encoder: nn.Module,
        gnn_block: Optional[Callable],
        gnn_position: str,
        num_node_classes: int,
    ) -> None:
        ffn_hidden = ffn_hidden_dim or hidden_dim
        biases = list(attn_bias_providers or [])
        pos_encs = list(positional_encoders or [])

        super().__init__(
            hidden_dim=hidden_dim,
            num_class=num_node_classes,
            use_super_node=use_super_node,
            node_feature_encoder=node_feature_encoder,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            ffn_hidden_dim=ffn_hidden,
            activation=activation,
            attn_bias_providers=biases,
            positional_encoders=pos_encs,
            gnn_block=gnn_block,
            gnn_position=gnn_position,
        )

        object.__setattr__(self, "ffn_hidden_dim", ffn_hidden)
        object.__setattr__(self, "attn_bias_providers", biases)
        object.__setattr__(self, "positional_encoders", pos_encs)
        object.__setattr__(
            self,
            "node_lin",
            nn.Linear(hidden_dim, num_node_classes)
        )

    def forward(self, data: Data) -> Tensor:
        """
        Forward pass producing per-node logits.

        Args:
            data (Data): Batched graph data with attributes:
                - x: [N, in_channels]
                - edge_index: [2, E]
                - batch: [N] graph indices
        Returns:
            Tensor: [N, num_node_classes] node logits.
        """
        x = self.node_feature_encoder(data.x)
        if self.gnn_block is not None and self.gnn_position == "pre":
            x = self.gnn_block(data, x)
        if self.is_encoder_stack:
            enc_layers = self.encoder
        else:
            enc_layers = [self.encoder]

        key_pad = None
        current_heads = None
        struct_mask = None

        for layer in enc_layers:
            if key_pad is None or layer.num_heads != current_heads:
                key_pad = build_key_padding(
                    data.batch,
                    num_heads=layer.num_heads
                )
                current_heads = layer.num_heads

            x = layer(x, data.batch, struct_mask, key_pad)

            if (self.gnn_block is not None and
                    self.gnn_position == "after_attn"):
                x = self.gnn_block(data, x)

        return self.node_lin(x)
