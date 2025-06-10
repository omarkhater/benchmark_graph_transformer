"""Model fixtures."""
import pytest
import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader


class DummyModel(Module):
    """Model that returns perfect one‐hot logits based on batch.y,
    scaled by a learnable weight."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = Parameter(torch.tensor(1.0))

    def forward(self, batch) -> Tensor:
        labels = batch.y.view(-1)
        num_classes = int(labels.max().item()) + 1
        logits = torch.zeros(
            (labels.size(0), num_classes), device=labels.device
        )
        logits[
            torch.arange(labels.size(0), device=labels.device), labels
        ] = 1.0
        return logits * self.scale


class DummyNodeClassifier(Module):
    """Perfect node-classifier (identity logits)."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, data: Data):
        y = data.y.view(-1)
        if self.num_classes == 1:                         # binary → (N, 1)
            return y.float().unsqueeze(1)
        logits = torch.zeros(y.size(0), self.num_classes, device=y.device)
        logits[torch.arange(y.size(0), device=y.device), y] = 1.0
        return logits


class DummyGraphClassifier(Module):
    """Perfect graph-classifier."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, batch: Batch):
        y = batch.y
        if self.num_classes == 1:
            return y.float().unsqueeze(1) if y.ndim == 1 else y.float()
        logits = torch.zeros(y.size(0), self.num_classes, device=y.device)
        logits[torch.arange(y.size(0), device=y.device), y] = 1.0
        return logits


class DummyRegressor(Module):
    """Returns labels untouched – perfect regressor."""

    def __init__(self, out_dim: int = 1):
        super().__init__()
        self.out_dim = out_dim

    def forward(self, batch: Batch):
        y = batch.y.clone().float()
        # For multi-target cases, we need to ensure 2D output
        # and preserve the original shape if present
        if self.out_dim > 1:
            # If input is already 2D with matching out_dim, preserve it
            if y.ndim == 2 and y.size(1) == self.out_dim:
                return y
            # Otherwise reshape to match expected (N, out_dim) shape
            return y.view(-1, self.out_dim)
        # For single target, always return (N, 1)
        return y.view(-1, 1)


@pytest.fixture
def dummy_model() -> DummyModel:
    """Provide a DummyModel instance for testing."""
    return DummyModel()


@pytest.fixture(autouse=True)
def ensure_model_has_parameter(dummy_model: DummyModel) -> None:
    """Ensure DummyModel has at least one parameter for optimizer tests."""
    if not any(True for _ in dummy_model.parameters()):
        dummy_model.register_parameter(
            "dummy_param", Parameter(torch.zeros(1))
        )


@pytest.fixture
def graph_loader() -> DataLoader:
    """Provide a DataLoader for two graph‐level samples with float targets."""
    d0 = Data(x=torch.randn(1, 4), y=torch.tensor([0.5], dtype=torch.float32))
    d1 = Data(x=torch.randn(1, 4), y=torch.tensor([1.5], dtype=torch.float32))
    return DataLoader([d0, d1], batch_size=2)


@pytest.fixture
def node_loader() -> DataLoader:
    """Provide a DataLoader for a single graph with 4 nodes [0,1,0,1]."""

    labels = torch.tensor([0, 1, 0, 1]).unsqueeze(1)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],   # "chain" or self‑loops, etc.
            [1, 2, 3, 0]
        ],  # here we connect 0→1,1→2,2→3,3→0
        dtype=torch.long)
    graph = Data(
        x=torch.randn(4, 4),
        y=labels,
        edge_index=edge_index
        )
    return DataLoader([graph], batch_size=1)


@pytest.fixture
def generic_loader() -> DataLoader:
    """Provide DataLoader for generic graph classification with 1D targets."""
    # two singleton graphs, each with a self‑loop edge:
    g0 = Data(
        x=torch.randn(1, 4),
        y=torch.tensor(0),
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),
    )
    g1 = Data(
        x=torch.randn(1, 4),
        y=torch.tensor(1),
        edge_index=torch.tensor([[0], [0]], dtype=torch.long),
    )
    return DataLoader([g0, g1], batch_size=2)


@pytest.fixture
def regression_loader() -> DataLoader:
    """Provide DataLoader for regression task with float targets."""
    # Create a small graph with float node labels
    graph = Data(
        x=torch.randn(4, 4),
        y=torch.randn(4),  # float targets for regression
        edge_index=torch.tensor(
            [[0, 1, 2, 3], [1, 2, 3, 0]],
            dtype=torch.long
        )
    )
    return DataLoader([graph], batch_size=1)


@pytest.fixture
def regression_none_x_loader() -> DataLoader:
    """Provide DataLoader for regression datasets with None node features.

    This fixture reproduces the structure found in some regression datasets
    where batch.x is None but edge_index and regression targets exist.
    This is needed to test edge cases in data enrichment functions.
    """
    # Create graph with None node features (like QM7b dataset)
    graph = Data(
        x=None,  # This is the key condition that causes issues
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        y=torch.tensor([42.5])  # float regression target
    )
    return DataLoader([graph], batch_size=1)


@pytest.fixture
def device() -> torch.device:
    """Provide a CPU device for testing."""
    return torch.device("cpu")
