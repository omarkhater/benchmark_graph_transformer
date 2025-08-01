"""create synthetic data for testing specific functionalities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import pytest
import torch
from torch.utils.data import Subset
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import degree


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
        return torch.randint(
            0, in_features, (num_nodes, 1)
        ).float()
    return torch.randn(num_nodes, in_features)


def create_targets(
    num_nodes: int,
    num_targets: int,
    task_type: str
) -> torch.Tensor:
    """Creates target values for nodes.

    Args:
        num_nodes: Number of nodes
        num_targets: Number of target classes/dimensions
        task_type: Either 'classification' or 'regression'

    Returns:
        torch.Tensor: Target values
    """
    if task_type == 'classification':
        if num_targets == 1:
            # Binary classification: return labels in {0, 1}
            # with shape (num_nodes, 1)
            return torch.randint(0, 2, (num_nodes, ))
        # Multiclass classification: return class indices
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


class ClassificationDataset:
    """Wrapper for classification data with correct num_classes."""

    def __init__(self, data_list: list[Data], num_classes: int):
        self._data_list = data_list
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self._data_list)

    def __getitem__(self, idx: int) -> Data:
        return self._data_list[idx]


class DatasetWithAttributes:
    """Dataset wrapper that mimics real PyG datasets with attributes."""

    def __init__(
        self,
        data_list: list[Data],
        *,
        num_node_features: int | None = None,
        num_classes: int | None = None
    ):
        self._data_list = data_list
        if num_node_features is not None:
            self.num_node_features = num_node_features
        if num_classes is not None:
            self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self._data_list)

    def __getitem__(self, idx: int) -> Data:
        return self._data_list[idx]


@pytest.fixture
def graph_loader() -> DataLoader:
    """Provide a DataLoader for two graph‐level samples with float targets."""
    d0 = Data(x=torch.randn(1, 4), y=torch.tensor([0.5], dtype=torch.float32))
    d1 = Data(x=torch.randn(1, 4), y=torch.tensor([1.5], dtype=torch.float32))
    return DataLoader([d0, d1], batch_size=2)


@pytest.fixture(params=[8, 16])
def node_loader(request) -> DataLoader:
    """
    Provide a DataLoader for a single graph with `N` nodes
    (where N comes from request.param).
    """
    feature_dim = 4
    N = request.param
    # simple binary labels alternating 0/1
    labels = torch.arange(N, dtype=torch.long) % 2
    labels = labels.unsqueeze(1)

    # make a simple cycle graph: 0→1→2→…→(N-1)→0
    idx = torch.arange(N, dtype=torch.long)
    edge_index = torch.stack([idx, (idx + 1) % N], dim=0)

    # random features, here we keep feature-dim=4
    x = torch.randn(N, feature_dim)

    graph = Data(x=x, y=labels, edge_index=edge_index)

    # spatial_pos: dummy zero–distance matrix
    graph.spatial_pos = torch.zeros((N, N), dtype=torch.long)
    # ptr: single graph, so [0, N]
    graph.ptr = torch.tensor([0, N], dtype=torch.long)
    graph.hop_dist = torch.zeros((N, N), dtype=torch.long)

    row, col = graph.edge_index
    graph.out_degree = degree(row, N, dtype=torch.long)
    graph.in_degree = degree(col, N, dtype=torch.long)

    graph.eig_pos_emb = torch.zeros((N, feature_dim), dtype=torch.float)
    graph.svd_pos_emb = torch.zeros((N, 2 * 3), dtype=torch.float)

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


@pytest.fixture(params=[8, 16])
def regression_loader(request) -> DataLoader:
    """
    Provide a DataLoader for node‐level regression on a single cycle graph
    of size N (where N comes from request.param), with float targets.
    """
    N = request.param
    feature_dim = 4

    # float targets, one per node
    y = torch.randn(N, 1, dtype=torch.float)

    # cycle graph 0→1→…→(N-1)→0
    idx = torch.arange(N, dtype=torch.long)
    edge_index = torch.stack([idx, (idx + 1) % N], dim=0)

    x = torch.randn(N, feature_dim, dtype=torch.float)
    graph = Data(x=x, y=y, edge_index=edge_index)

    # structural tensors for all features / biases / positional encoders
    graph.spatial_pos = torch.zeros((N, N), dtype=torch.long)
    graph.hop_dist = torch.zeros((N, N), dtype=torch.long)
    graph.ptr = torch.tensor([0, N], dtype=torch.long)

    # degrees for DegreeEncoder
    row, col = graph.edge_index
    graph.out_degree = degree(row, N, dtype=torch.long)
    graph.in_degree = degree(col, N, dtype=torch.long)

    # positional embeddings for EigEncoder & SVDEncoder
    # match your cfg: num_eigvec=4, num_svdenc=3 → concatenated size=6
    graph.eig_pos_emb = torch.zeros((N, 4), dtype=torch.float)
    graph.svd_pos_emb = torch.zeros((N, 2 * 3), dtype=torch.float)

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
        y = create_targets(num_nodes, num_targets, task_type)

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
            y = torch.randint(0, 2 if num_targets == 1 else num_targets, (1,))
        else:
            y = torch.randn(1 if num_targets == 1 else num_targets)

        edge_attr = None
        if edge_attr_dim is not None:
            edge_attr = torch.randn(edge_index.size(1), edge_attr_dim)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        return data

    def _make_dataset(
        num_graphs: int = 4,
        in_features: int = 3,
        num_targets: int = 2,
        task_type: str = 'classification',
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
                in_features, task_type, num_targets,
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
        num_targets=1,
        task_type='classification'
    )


@pytest.fixture
def multiclass_node_data(make_node_data) -> Data:
    """Node-level 3-class classification with 4 nodes, 3 features"""
    return make_node_data(
        num_nodes=4,
        in_features=3,
        num_targets=3,
        task_type='classification'
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
        num_targets=1,
        task_type='classification',
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
def graph_batch(simple_graph: Data) -> Batch:
    """Batch of two identical graphs, for whole‐graph models."""
    return Batch.from_data_list([simple_graph, simple_graph])


@pytest.fixture
def masked_node_loader(make_node_data) -> DataLoader:
    """Provides a DataLoader for a graph with train/val/test node masks.

    Creates a single graph with 100 nodes and adds boolean mask tensors
    covering ~10% of nodes each for train/val/test splits.

    Returns:
        DataLoader: Loader with batch_size=1 containing the masked graph
    """
    data = make_node_data(
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
def cora_style_loader(make_node_data) -> DataLoader:
    """Provides a DataLoader mimicking the Cora citation network structure.

    Creates a single graph with Cora-like dimensions:
    - 2708 nodes (papers)
    - 1433 features (word presence/absence)
    - 7 classes
    - 60/20/20 train/val/test split

    Returns:
        DataLoader: Loader with batch_size=1 containing the Cora-like graph
    """
    data = make_node_data(
        num_nodes=2708,
        in_features=1433,
        num_targets=7,
        task_type='classification',
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
        task_str = "classification" if classification else "regression"

        if cfg.task_type == "node":
            return self._make_node(
                num_nodes=cfg.num_nodes,
                in_features=cfg.num_features,
                num_targets=cfg.num_targets,
                task_type=task_str,
                edge_attr_dim=cfg.edge_attr_dim,
                sparse=cfg.sparse
            )

        return self._make_graph(
            num_graphs=cfg.num_graphs,
            in_features=cfg.num_features,
            num_targets=cfg.num_targets,
            task_type=task_str,
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
def graph_classification_suite(request) -> dict[str, DataLoader]:
    """Provides a suite of classification test cases.

    The fixture can be parameterized to control the numbers used in the
    data generation.
    Use pytest.mark.parametrize with a dict of config overrides:

    @pytest.mark.parametrize('graph_classification_suite', [
        {'node_binary_targets': 1, 'multiclass_targets': 4}
    ], indirect=True)

    Returns a dictionary containing different DataLoader configurations for
    classification tasks:
        - node_binary: Node-level binary classification
        - node_multiclass: Node-level multiclass classification
        - graph_binary: Graph-level binary classification
        - graph_multiclass: Graph-level multiclass classification
        - masked_nodes: Node classification with train/val/test masks
        - sparse_features: Classification with sparse node features
        - edge_attr: With edge attributes for attention testing

    This comprehensive suite helps test behavior across different
    classification scenarios found in real-world datasets.

    Returns:
        dict[str, DataLoader]: Named test cases mapping to DataLoaders
    """
    # Default configuration - can be overridden by test parameters
    default_config = {
        'node_binary_targets': 2,
        'multiclass_targets': 3,
        'edge_attr_targets': 2,
        'sparse_targets': 2,
        'masked_targets': 3,
        'num_nodes': 4,
        'in_features': 3,
        'num_graphs': 4,
        'masked_nodes': 100,
        'sparse_features': 8,
        'edge_attr_dim': 4,
    }

    # Override with test-specific parameters if provided
    config = default_config.copy()
    if hasattr(request, 'param') and request.param:
        config.update(request.param)

    make_node = request.getfixturevalue('make_node_data')
    make_graph = request.getfixturevalue('make_graph_dataset')

    # Node-level binary classification
    node_binary = make_node(
        num_nodes=config['num_nodes'],
        in_features=config['in_features'],
        num_targets=config['node_binary_targets'],
        task_type='classification'
    )

    # Node-level multiclass
    node_multi = make_node(
        num_nodes=config['num_nodes'],
        in_features=config['in_features'],
        num_targets=config['multiclass_targets'],
        task_type='classification'
    )

    # Graph-level binary classification
    graph_binary = make_graph(
        num_graphs=config['num_graphs'],
        in_features=config['in_features'],
        num_targets=config['node_binary_targets'],
        task_type='classification'
    )

    # Graph-level multiclass classification
    graph_multi = make_graph(
        num_graphs=config['num_graphs'],
        in_features=config['in_features'],
        num_targets=config['multiclass_targets'],
        task_type='classification'
    )

    # With edge attributes
    edge_attr = make_graph(
        num_graphs=config['num_graphs'],
        in_features=config['in_features'],
        num_targets=config['edge_attr_targets'],
        task_type='classification',
        edge_attr_dim=config['edge_attr_dim']
    )

    # With sparse features (like citation networks)
    sparse_node = make_node(
        num_nodes=config['num_nodes'],
        in_features=config['sparse_features'],
        num_targets=config['sparse_targets'],
        task_type='classification',
        sparse=True
    )

    # Add train/val/test masks
    masked = make_node(
        num_nodes=config['masked_nodes'],
        in_features=config['sparse_features'],
        num_targets=config['masked_targets'],
        task_type='classification'
    )
    train_mask, val_mask, test_mask = create_train_val_test_masks(
        config['masked_nodes'], train_ratio=0.6, val_ratio=0.2
    )
    masked.train_mask = train_mask
    masked.val_mask = val_mask
    masked.test_mask = test_mask

    # Create datasets with num_node_features attribute (real PyG usage)
    # Use the actual feature dimension from the graph data
    actual_features = (
        graph_binary[0].x.size(-1) if graph_binary[0].x is not None else 0
    )

    # Create a Subset case to test parent dataset access
    pyg_style_dataset = DatasetWithAttributes(
        graph_binary,
        num_node_features=actual_features,
        num_classes=2  # Binary classification for PyG style
    )
    subset_dataset = Subset(pyg_style_dataset, [0, 1])

    # Create DataLoaders and add expected_classes attribute
    node_binary_loader = DataLoader([node_binary], batch_size=1)
    # Binary with (N, 1) shape -> 1 class dimension
    node_binary_loader.expected_classes = config['node_binary_targets']

    node_multi_loader = DataLoader([node_multi], batch_size=1)
    # Multiclass targets
    node_multi_loader.expected_classes = config['multiclass_targets']

    # For ClassificationDataset, we pass the desired num_classes directly
    graph_binary_num_classes = config['node_binary_targets'] + 1
    graph_binary_loader = DataLoader(
        ClassificationDataset(graph_binary, graph_binary_num_classes),
        batch_size=2
    )
    graph_binary_loader.expected_classes = graph_binary_num_classes

    graph_multi_num_classes = config['multiclass_targets']
    graph_multi_loader = DataLoader(
        ClassificationDataset(graph_multi, graph_multi_num_classes),
        batch_size=2,
    )
    graph_multi_loader.expected_classes = graph_multi_num_classes

    edge_attr_num_classes = config['edge_attr_targets']
    edge_attr_loader = DataLoader(
        ClassificationDataset(edge_attr, edge_attr_num_classes), batch_size=2
    )
    edge_attr_loader.expected_classes = edge_attr_num_classes

    sparse_loader = DataLoader([sparse_node], batch_size=1)
    sparse_loader.expected_classes = config['sparse_targets']

    masked_loader = DataLoader([masked], batch_size=1)
    masked_loader.expected_classes = config['masked_targets']

    # For PyG style dataset, num_classes is set during creation
    pyg_style_num_classes = 2  # Set in DatasetWithAttributes creation above
    pyg_loader = DataLoader(pyg_style_dataset, batch_size=2)
    pyg_loader.expected_classes = pyg_style_num_classes

    subset_loader = DataLoader(subset_dataset, batch_size=2)
    # Subset inherits from parent dataset
    subset_loader.expected_classes = pyg_style_num_classes

    return {
        'node_binary': node_binary_loader,
        'node_multiclass': node_multi_loader,
        'graph_binary': graph_binary_loader,
        'graph_multiclass': graph_multi_loader,
        'edge_attr': edge_attr_loader,
        'sparse_features': sparse_loader,
        'masked_nodes': masked_loader,
        'pyg_style': pyg_loader,
        'subset_with_parent': subset_loader
    }


@pytest.fixture
def graph_regression_suite(request) -> dict[str, DataLoader]:
    """Provides a suite of graph regression test cases.

    The fixture can be parameterized to control the numbers used in
    data generation.
    Use pytest.mark.parametrize with a dict of config overrides:

    @pytest.mark.parametrize('graph_regression_suite', [
        {'single_targets': 2, 'multi_targets': 7}
    ], indirect=True)

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
    # Default configuration - can be overridden by test parameters
    default_config = {
        'single_targets': 1,
        'multi_targets': 5,
        'edge_attr_targets': 2,
        'varied_targets': 2,
        'no_features_targets': 3,
        'num_graphs': 4,
        'in_features': 3,
        'edge_attr_dim': 4,
        'min_nodes': 2,
        'max_nodes': 10,
    }

    # Override with test-specific parameters if provided
    config = default_config.copy()
    if hasattr(request, 'param') and request.param:
        config.update(request.param)

    make_dataset = request.getfixturevalue('make_graph_dataset')

    # Single target case (e.g. ZINC-like)
    single_target = make_dataset(
        num_graphs=config['num_graphs'],
        in_features=config['in_features'],
        num_targets=config['single_targets'],
        task_type='regression'
    )

    # Multi-target case (e.g. QM9-like)
    multi_target = make_dataset(
        num_graphs=config['num_graphs'],
        in_features=config['in_features'],
        num_targets=config['multi_targets'],  # Multiple properties per mol
        task_type='regression'
    )

    # Case with edge attributes (e.g. molecular graphs)
    edge_attr = make_dataset(
        num_graphs=config['num_graphs'],
        in_features=config['in_features'],
        num_targets=config['edge_attr_targets'],
        task_type='regression',
        edge_attr_dim=config['edge_attr_dim']  # e.g. distance, bond type
    )

    # No node features case (e.g. QM7b-like)
    graph_no_x = Data(
        x=None,
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        y=torch.randn(config['no_features_targets'])  # quantum props
    )

    # Create datasets with num_node_features attribute (real PyG usage)
    # Use the actual feature dimension from the graph data
    actual_features = (
        single_target[0].x.size(-1) if single_target[0].x is not None else 0
    )

    # Create a Subset case to test parent dataset access
    pyg_style_regression = DatasetWithAttributes(
        single_target,
        num_node_features=actual_features
    )
    subset_regression = Subset(pyg_style_regression, [0, 1])

    return {
        'single_target': DataLoader(single_target, batch_size=2),
        'multi_target': DataLoader(multi_target, batch_size=2),
        'edge_attr': DataLoader(edge_attr, batch_size=2),
        'no_features': DataLoader([graph_no_x], batch_size=1),
        'varied_graphs': DataLoader(
            make_dataset(
                num_graphs=config['num_graphs'],
                in_features=config['in_features'],
                num_targets=config['varied_targets'],
                task_type='regression',
                min_nodes=config['min_nodes'],
                max_nodes=config['max_nodes']  # More variation in graph sizes
            ),
            batch_size=2
        ),
        'pyg_style': DataLoader(pyg_style_regression, batch_size=2),
        'subset_with_parent': DataLoader(subset_regression, batch_size=2)
    }


@pytest.fixture
def generic_batch(generic_loader: DataLoader):
    """Turn the generic_loader into a single Batch."""
    return next(iter(generic_loader))


@pytest.fixture
def masked_node_batch(masked_node_loader: DataLoader):
    """Turn the masked_node_loader into a single Batch."""
    return next(iter(masked_node_loader))


@pytest.fixture
def cora_style_batch(cora_style_loader: DataLoader):
    """Turn the cora_style_loader into a single Batch."""
    return next(iter(cora_style_loader))


@pytest.fixture
def regression_none_x_batch(regression_none_x_loader: DataLoader):
    """Turn the regression_none_x_loader into a single Batch."""
    return next(iter(regression_none_x_loader))


@pytest.fixture
def data_batch(request):
    """
    Fixture for providing a single batch of graph data.
    This fixture is parameterized to allow different batch configurations
    to be tested by specifying the fixture name in the test function.
    """
    batch = request.getfixturevalue(request.param)
    if not isinstance(batch, Batch):
        raise ValueError(
            f"Fixture {request.param} must be a torch_geometric.data.Batch "
            f"got {type(batch)}"
        )
    return batch


@pytest.fixture
def node_regression_loader() -> DataLoader:
    """
    Provide a DataLoader for node‐level regression on a fixed-size graph.
    This fixture creates a simple cycle graph with 4 nodes, each having
    random float features and targets.
    """
    # fixed-size toy graph
    N = 4
    feature_dim = 4

    # random node features and float targets
    x = torch.randn(N, feature_dim, dtype=torch.float)
    y = torch.randn(N, 1, dtype=torch.float)

    # a simple 4‐node cycle: 0→1→2→3→0
    edge_index = torch.tensor(
        [[0, 1, 2, 3],
         [1, 2, 3, 0]],
        dtype=torch.long,
    )

    graph = Data(x=x, y=y, edge_index=edge_index)
    return DataLoader([graph], batch_size=1)


@pytest.fixture
def node_regression_batch(node_regression_loader: DataLoader) -> Batch:
    """Turn the single‐graph loader into a Batch."""
    return next(iter(node_regression_loader))
