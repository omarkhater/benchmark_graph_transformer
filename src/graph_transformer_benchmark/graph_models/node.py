"""
Node-level GNN classifier emitting per-node logits.
"""
from torch import Tensor
from torch_geometric.data import Data

from .base import BaseGNN


class NodeGNN(BaseGNN):
    """
    GNN classifier for node-level tasks.

    Inherits BaseGNN but omits the global pooling step,
    producing one logit per node.
    """

    def forward(self, data: Data) -> Tensor:
        """
        Forward pass for node classification.

        Args:
            data (Data): Graph batch with node features and edge_index.

        Returns:
            Tensor: [N, num_classes] node logits.
        """
        x0 = data.x
        x = self.conv(x0, data.edge_index)
        x = self.bn(x)

        if self.use_residual and self.res_proj is not None:
            x = x + self.res_proj(x0)

        return self.lin(x)
