"""create synthetic data for testing specific functionalities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader


def create_train_val_test_masks(
    num_nodes: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Creates boolean masks for train/val/test splits of nodes.

    Args:
        num_nodes: Total number of nodes to split
        train_ratio: Fraction of nodes for training (default: 0.6)
        val_ratio: Fraction of nodes for validation (default: 0.2)
            Test ratio is inferred as 1 - train_ratio - val_ratio

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Boolean masks for
            train, validation and test sets respectively
    """
    indices = torch.randperm(num_nodes)
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size + val_size]] = True
    test_mask[indices[train_size + val_size:]] = True

    return train_mask, val_mask, test_mask


def create_node_features(
    num_nodes: int,
    in_features: int,
    sparse: bool = False
) -> torch.Tensor:
    """Creates node feature matrix.

    Args:
        num_nodes: Number of nodes
        in_features: Number of features per node
        sparse: If True, return sparse features as indices

    Returns:
        torch.Tensor: Node features
    """
    if sparse:
        return torch.randint(0, in_features, (num_nodes,))
    return torch.randn(num_nodes, in_features)


def create_targets(
    num_nodes: int,
    num_targets: int,
    task_type: str,
    is_binary: bool = True
) -> torch.Tensor:
    """Creates target values for nodes.

    Args:
        num_nodes: Number of nodes
        num_targets: Number of target classes/dimensions
        task_type: Either 'classification' or 'regression'
        is_binary: For classification, whether binary or multiclass

    Returns:
        torch.Tensor: Target values
    """
    if task_type == 'classification':
        if is_binary:
            return torch.randint(0, 2, (num_nodes, 1))
        return torch.randint(0, num_targets, (num_nodes,))

    if num_targets == 1:
        return torch.randn(num_nodes)
    return torch.randn(num_nodes, num_targets)


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
            [0, 1, 2, 3],
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
    """Factory fixture for creating node-level task data.

    Args:
        num_nodes: Number of nodes in the graph
        in_features: Number of input features per node
        num_targets: Number of target classes/dimensions
        task_type: Either 'classification' or 'regression'
        is_binary: For classification, whether binary or multiclass
        edge_attr_dim: Optional dimension for edge attributes
        sparse: If True, use sparse node features as indices

    Returns:
        Data: A PyG Data object with synthetic graph data
    """
    def _make_data(
        num_nodes: int = 4,
        in_features: int = 3,
        num_targets: int = 2,
        task_type: str = 'classification',
        is_binary: bool = True,
        edge_attr_dim: int | None = None,
        sparse: bool = False
    ) -> Data:
        if task_type not in ['classification', 'regression']:
            raise ValueError(
                f"Task type must be 'classification' or 'regression'"
                f", got {task_type}"
            )

        # Create features and targets
        x = create_node_features(num_nodes, in_features, sparse)
        y = create_targets(num_nodes, num_targets, task_type, is_binary)

        # Create edges - circular graph
        edge_index = torch.stack([
            torch.arange(num_nodes),
            torch.roll(torch.arange(num_nodes), -1)
        ])

        # Optional edge attributes
        edge_attr = None
        if edge_attr_dim is not None:
            edge_attr = torch.randn(edge_index.size(1), edge_attr_dim)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    return _make_data


@pytest.fixture
def make_graph_dataset():
    """Factory fixture for creating graph-level task datasets.

    Args:
        num_graphs: Number of graphs to generate
        in_features: Number of input features per node
        num_targets: Number of target classes/dimensions
        task_type: Either 'classification' or 'regression'
        is_binary: For classification, whether binary or multiclass
        edge_attr_dim: Optional dimension for edge attributes
        sparse: If True, use sparse node features as indices
        min_nodes: Minimum nodes per graph
        max_nodes: Maximum nodes per graph

    Returns:
        list[Data]: List of PyG Data objects representing graphs
    """
    def _create_graph(
        num_nodes: int,
        in_features: int,
        task_type: str,
        num_targets: int,
        is_binary: bool,
        edge_attr_dim: int | None,
        sparse: bool
    ) -> Data:
        """Create a single graph with specified properties."""
        x = create_node_features(num_nodes, in_features, sparse)
        edge_index = torch.stack([
            torch.arange(num_nodes),
            torch.roll(torch.arange(num_nodes), -1)
        ])

        # Graph-level targets
        if task_type == 'classification':
            y = torch.randint(0, 2 if is_binary else num_targets, (1,))
        else:
            y = torch.randn(1 if num_targets == 1 else num_targets)

        edge_attr = None
        if edge_attr_dim is not None:
            edge_attr = torch.randn(edge_index.size(1), edge_attr_dim)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    def _make_dataset(
        num_graphs: int = 4,
        in_features: int = 3,
        num_targets: int = 2,
        task_type: str = 'classification',
        is_binary: bool = True,
        edge_attr_dim: int | None = None,
        sparse: bool = False,
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
                in_features, task_type, num_targets, is_binary,
                edge_attr_dim, sparse
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


@pytest.fixture
def masked_node_loader() -> DataLoader:
    """Provides a DataLoader for a graph with train/val/test node masks.

    Creates a single graph with 100 nodes and adds boolean mask tensors
    covering ~10% of nodes each for train/val/test splits.

    Returns:
        DataLoader: Loader with batch_size=1 containing the masked graph
    """
    data = make_node_data()(
        num_nodes=100,
        in_features=8,
        num_targets=2,
        task_type='classification'
    )
    train_mask, val_mask, test_mask = create_train_val_test_masks(
        100, train_ratio=0.1, val_ratio=0.1
    )
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return DataLoader([data], batch_size=1)


@pytest.fixture
def cora_style_loader() -> DataLoader:
    """Provides a DataLoader mimicking the Cora citation network structure.

    Creates a single graph with Cora-like dimensions:
    - 2708 nodes (papers)
    - 1433 features (word presence/absence)
    - 7 classes
    - 60/20/20 train/val/test split

    Returns:
        DataLoader: Loader with batch_size=1 containing the Cora-like graph
    """
    data = make_node_data()(
        num_nodes=2708,
        in_features=1433,
        num_targets=7,
        task_type='classification',
        is_binary=False,
        sparse=True  # Use sparse features like Cora
    )
    train_mask, val_mask, test_mask = create_train_val_test_masks(
        2708, train_ratio=0.6, val_ratio=0.2
    )
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return DataLoader([data], batch_size=1)


@dataclass
class DataConfig:
    """Configuration for synthetic data generation.

    Args:
        task_type: Either "node" or "graph"
        is_regression: If True, regression task, otherwise classification
        num_targets: Number of target classes or dimensions
        num_features: Number of input features per node
        sparse: Whether to use sparse node features
        edge_attr_dim: Optional dimension for edge attributes
        num_nodes: Number of nodes (node tasks) or max nodes (for graph tasks)
        num_graphs: Number of graphs to generate (for graph tasks)
    """
    task_type: Literal["node", "graph"]
    is_regression: bool
    num_targets: int = 1
    num_features: int = 3
    sparse: bool = False
    edge_attr_dim: int | None = None
    num_nodes: int = 4
    num_graphs: int = 4


class DataManager:
    """Manager class providing unified access to all data fixtures.

    This class simplifies test data generation by providing:
    1. Consistent interface for both node and graph tasks
    2. Support for all target types (binary/multiclass/regression)
    3. Optional edge attributes and sparse features
    4. Both individual graphs and batched data
    5. Train/val/test splits for node tasks

    Example:
        ```python
        def test_model_handles_all_cases(data_manager):
            # Test node binary classification
            cfg = DataConfig(
                task_type="node", is_regression=False, num_targets=1
            )
            node_data = data_manager.get_data(cfg)
            assert model(node_data).shape == expected_shape

            # Test graph regression
            cfg = DataConfig(task_type="graph", is_regression=True)
            graph_data = data_manager.get_data(cfg)
            assert model(graph_data).shape == expected_shape
        ```
    """
    def __init__(
        self,
        make_node_data: Callable,
        make_graph_dataset: Callable
    ) -> None:
        self._make_node = make_node_data
        self._make_graph = make_graph_dataset

    def get_data(self, cfg: DataConfig) -> Data | list[Data]:
        """Get data for specific task and target type.

        Args:
            cfg: DataConfig specifying the data requirements

        Returns:
            Data for node tasks, list[Data] for graph tasks
        """
        classification = not cfg.is_regression
        is_binary = classification and cfg.num_targets == 1
        task_str = "classification" if classification else "regression"

        if cfg.task_type == "node":
            return self._make_node(
                num_nodes=cfg.num_nodes,
                in_features=cfg.num_features,
                num_targets=cfg.num_targets,
                task_type=task_str,
                is_binary=is_binary,
                edge_attr_dim=cfg.edge_attr_dim,
                sparse=cfg.sparse
            )

        return self._make_graph(
            num_graphs=cfg.num_graphs,
            in_features=cfg.num_features,
            num_targets=cfg.num_targets,
            task_type=task_str,
            is_binary=is_binary,
            edge_attr_dim=cfg.edge_attr_dim,
            sparse=cfg.sparse
        )

    def get_loader(self, cfg: DataConfig, batch_size: int = 2) -> DataLoader:
        """Get DataLoader for specific task and target type.

        Args:
            cfg: DataConfig specifying the data requirements
            batch_size: Batch size for the loader

        Returns:
            DataLoader containing the requested data
        """
        data = self.get_data(cfg)
        if isinstance(data, list):
            return DataLoader(data, batch_size=batch_size)
        return DataLoader([data], batch_size=1)

    def get_masked_loader(
        self,
        cfg: DataConfig,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2
    ) -> DataLoader:
        """Get loader with train/val/test node masks.

        Args:
            cfg: DataConfig specifying the data requirements
                (must be node task)
            train_ratio: Fraction of nodes for training
            val_ratio: Fraction of nodes for validation

        Returns:
            DataLoader with masked node data
        """
        if cfg.task_type != "node":
            raise ValueError("get_masked_loader only supports node tasks")

        data = self.get_data(cfg)
        train_mask, val_mask, test_mask = create_train_val_test_masks(
            data.num_nodes, train_ratio, val_ratio
        )
        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask
        return DataLoader([data], batch_size=1)


@pytest.fixture
def data_manager(make_node_data, make_graph_dataset) -> DataManager:
    """Provides DataManager instance for unified data access."""
    return DataManager(make_node_data, make_graph_dataset)


@pytest.fixture
def graph_regression_suite(request) -> dict[str, DataLoader]:
    """Provides a suite of graph regression test cases.

    Returns a dictionary containing different DataLoader configurations for
    graph regression tasks:
        - single_target: One regression target per graph
        - multi_target: Multiple regression targets per graph (e.g. QM9)
        - varied_graphs: Graphs with different numbers of nodes
        - no_features: Graphs without node features (e.g. QM7b)
        - edge_attr: Graphs with edge attributes (e.g. distance matrices)

    This comprehensive suite helps test model behavior across different
    graph regression scenarios found in real-world datasets.

    Returns:
        dict[str, DataLoader]: Named test cases mapping to DataLoaders
    """
    make_dataset = request.getfixturevalue('make_graph_dataset')

    # Single target case (e.g. ZINC-like)
    single_target = make_dataset(
        num_graphs=4,
        in_features=3,
        num_targets=1,
        task_type='regression'
    )

    # Multi-target case (e.g. QM9-like)
    multi_target = make_dataset(
        num_graphs=4,
        in_features=3,
        num_targets=5,  # Multiple properties per molecule
        task_type='regression'
    )

    # Case with edge attributes (e.g. molecular graphs)
    edge_attr = make_dataset(
        num_graphs=4,
        in_features=3,
        num_targets=2,
        task_type='regression',
        edge_attr_dim=4  # e.g. distance, bond type encoding
    )

    # No node features case (e.g. QM7b-like)
    graph_no_x = Data(
        x=None,
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        y=torch.randn(3)  # 3 quantum properties
    )

    return {
        'single_target': DataLoader(single_target, batch_size=2),
        'multi_target': DataLoader(multi_target, batch_size=2),
        'edge_attr': DataLoader(edge_attr, batch_size=2),
        'no_features': DataLoader([graph_no_x], batch_size=1),
        'varied_graphs': DataLoader(
            make_dataset(
                num_graphs=4,
                in_features=3,
                num_targets=2,
                task_type='regression',
                min_nodes=2,
                max_nodes=10  # More variation in graph sizes
            ),
            batch_size=2
        )
    }
