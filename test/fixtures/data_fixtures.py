"""create synthetic data for testing specific functionalities."""
from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader


class DummyDataset:
    """Wraps a list of Data objects and exposes length and metadata."""
    def __init__(self, data_list: list[Data]) -> None:
        self._list = data_list
        first = data_list[0]
        self.num_node_features = first.x.size(1)
        # assume y is a 1D or 2D tensor of labels
        y = first.y.view(-1)
        self.num_classes = int(y.max().item()) + 1

    def __len__(self) -> int:
        return len(self._list)

    def __getitem__(self, idx: int) -> Data:
        return self._list[idx]


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
def make_node_data():
    """Factory fixture for creating node-level task data."""
    def _make_data(
        num_nodes: int = 4,
        in_features: int = 3,
        num_targets: int = 2,
        task_type: str = 'classification',
        is_binary: bool = True
    ) -> Data:
        SUPPORTED_TASKS = ['classification', 'regression']
        if task_type not in SUPPORTED_TASKS:
            raise ValueError(
                f"Task type must be one of {SUPPORTED_TASKS}, got {task_type}"
            )
        x = torch.randn(num_nodes, in_features)
        edge_index = torch.stack([
            torch.arange(num_nodes),
            torch.roll(torch.arange(num_nodes), -1)
        ])
        if task_type == 'classification':
            if is_binary:
                y = torch.randint(0, 2, (num_nodes, 1))
            else:
                y = torch.randint(0, num_targets, (num_nodes,))
        elif task_type == 'regression':
            if num_targets == 1:
                y = torch.randn(num_nodes)
            else:
                y = torch.randn(num_nodes, num_targets)
        return Data(x=x, y=y, edge_index=edge_index)
    return _make_data


@pytest.fixture
def make_graph_dataset():
    """Factory fixture for creating graph-level task datasets."""
    def _create_graph(
            num_nodes: int,
            in_features: int,
            task_type: str,
            num_targets: int,
            is_binary: bool
            ) -> Data:
        """Create a single graph with specified properties."""
        x = torch.randn(num_nodes, in_features)
        edge_index = torch.stack([
            torch.arange(num_nodes),
            torch.roll(torch.arange(num_nodes), -1)
        ])
        if task_type == 'classification':
            y = torch.randint(0, 2 if is_binary else num_targets, (1,))
        else:
            y = torch.randn(1 if num_targets == 1 else num_targets)
        return Data(x=x, y=y, edge_index=edge_index)

    def _make_dataset(
        num_graphs: int = 4,
        in_features: int = 3,
        num_targets: int = 2,
        task_type: str = 'classification',
        is_binary: bool = True,
        min_nodes: int = 1,
        max_nodes: int = 4
    ) -> list[Data]:
        """Create a dataset of graphs with specified properties."""
        if task_type not in ['classification', 'regression']:
            raise ValueError(
                "Task type must be 'classification' or 'regression'"
            )
        return [
            _create_graph(
                torch.randint(min_nodes, max_nodes + 1, (1,)).item(),
                in_features, task_type, num_targets, is_binary
            )
            for _ in range(num_graphs)
        ]
    return _make_dataset


@pytest.fixture
def binary_node_data(make_node_data) -> Data:
    """Node-level binary classification with 4 nodes, 3 features"""
    return make_node_data(
        num_nodes=4,
        in_features=3,
        num_targets=2,
        task_type='classification',
        is_binary=True
    )


@pytest.fixture
def multiclass_node_data(make_node_data) -> Data:
    """Node-level 3-class classification with 4 nodes, 3 features"""
    return make_node_data(
        num_nodes=4,
        in_features=3,
        num_targets=3,
        task_type='classification',
        is_binary=False
    )


@pytest.fixture
def node_reg_single_target(make_node_data) -> Data:
    """Node-level single target regression with 4 nodes, 3 features"""
    return make_node_data(
        num_nodes=4,
        in_features=3,
        num_targets=1,
        task_type='regression'
    )


@pytest.fixture
def node_reg_multi_target(make_node_data) -> Data:
    """Node-level multi-target regression with 4 nodes, 3 features"""
    return make_node_data(
        num_nodes=4,
        in_features=3,
        num_targets=3,
        task_type='regression'
    )


@pytest.fixture
def binary_graph_dataset(make_graph_dataset) -> list[Data]:
    """Graph-level binary classification with varied nodes, 3 features"""
    return make_graph_dataset(
        num_graphs=4,
        in_features=3,
        num_targets=2,
        task_type='classification',
        is_binary=True,
        min_nodes=1,
        max_nodes=3
    )


@pytest.fixture
def multiclass_graph_dataset(make_graph_dataset) -> list[Data]:
    """Graph-level 3-class classification with varied nodes, 3 features"""
    return make_graph_dataset(
        num_graphs=4,
        in_features=3,
        num_targets=3,
        task_type='classification',
        is_binary=False,
        min_nodes=1,
        max_nodes=4
    )


@pytest.fixture
def graph_reg_single_target(make_graph_dataset) -> list[Data]:
    """Graph-level single-target regression with varied nodes, 3 features"""
    return make_graph_dataset(
        num_graphs=4,
        in_features=3,
        num_targets=1,
        task_type='regression',
        min_nodes=1,
        max_nodes=4
    )


@pytest.fixture
def graph_reg_multi_target(make_graph_dataset) -> list[Data]:
    """Graph-level multi-target regression with varied nodes, 3 features"""
    return make_graph_dataset(
        num_graphs=4,
        in_features=3,
        num_targets=3,
        task_type='regression',
        min_nodes=1,
        max_nodes=4
    )


@pytest.fixture
def simple_graph() -> Data:
    """A toy 2‑node graph with an edge between them."""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    edge_index = torch.tensor([[0, 1], [1, 0]])
    # give it a graph‐level label (e.g. class 0)
    data = Data(x=x, edge_index=edge_index, y=torch.tensor([0]))
    data.batch = torch.zeros(x.size(0), dtype=torch.long)
    return data


@pytest.fixture
def simple_batch(simple_graph: Data) -> Batch:
    return Batch.from_data_list([simple_graph, simple_graph])


@pytest.fixture
def graph_batch(simple_graph: Data) -> Batch:
    """Batch of two identical graphs, for whole‐graph models."""
    return Batch.from_data_list([simple_graph, simple_graph])
