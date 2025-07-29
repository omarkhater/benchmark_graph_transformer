"""Model fixtures."""
import pytest
import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch_geometric.data import Batch, Data


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
        y = batch.y.view(-1)
        if self.num_classes == 1:
            return torch.ones(
                (y.size(0), 1),
                device=y.device,
                dtype=torch.float
            )
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
def device() -> torch.device:
    """Provide a CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def node_classifier_model():
    """Factory for DummyNodeClassifier.

    It returns a callable that creates a new instance
    with the specified number of classes at runtime.

    """
    def _make(num_classes: int) -> DummyNodeClassifier:
        return DummyNodeClassifier(num_classes)
    return _make


@pytest.fixture
def graph_classifier_model():
    """Factory for DummyGraphClassifier.

    It returns a callable that creates a new instance
    with the specified number of classes at runtime.

    """
    def _make(num_classes: int) -> DummyGraphClassifier:
        return DummyGraphClassifier(num_classes)
    return _make


@pytest.fixture
def regressor_model():
    """Factory for DummyRegressor.

    It returns a callable that creates a new instance
    with the specified output dimension at runtime.

    """
    def _make(out_dim: int = 1) -> DummyRegressor:
        return DummyRegressor(out_dim)
    return _make
