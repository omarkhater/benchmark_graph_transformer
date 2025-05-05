"""
Base graph-level GNN classifier with optional BatchNorm & residual connections.
"""
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool


class BaseGNN(nn.Module):
    """
    Graph-level GNN classifier.

    Applies a single GNN convolution, optional batch normalization, and
    an optional residual connection, then pools over nodes to produce
    graph-level logits.
    """

    def __init__(
        self,
        conv: nn.Module,
        in_channels: int,
        hidden_dim: int,
        num_classes: int,
        use_batch_norm: bool,
        use_residual: bool,
    ) -> None:
        super().__init__()
        self.conv = conv
        if use_batch_norm:
            self.bn = nn.BatchNorm1d(hidden_dim)
        else:
            self.bn = nn.Identity()
        if use_residual:
            if in_channels != hidden_dim:
                self.res_proj = nn.Linear(in_channels, hidden_dim)
            else:
                self.res_proj = nn.Identity()
        else:
            self.res_proj = None
        self.lin = nn.Linear(hidden_dim, num_classes)
        self.use_residual = use_residual

    def forward(self, data: Data) -> Tensor:
        """
        Forward pass for graph classification.

        Args:
            data (Data): Batch of graphs with attributes:
                - x: [N, in_channels] node features
                - edge_index: [2, E] edge indices
                - batch: [N] mapping nodes to graph IDs

        Returns:
            Tensor: [num_graphs, num_classes] graph logits.
        """
        x0 = data.x
        x = self.conv(x0, data.edge_index)
        x = self.bn(x)

        if self.use_residual and self.res_proj is not None:
            x = x + self.res_proj(x0)

        x = global_mean_pool(x, data.batch)
        return self.lin(x)
