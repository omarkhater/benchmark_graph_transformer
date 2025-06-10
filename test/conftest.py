import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generator, List
from unittest.mock import MagicMock

import mlflow
import pytest
import torch
from ogb.graphproppred import PygGraphPropPredDataset
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim import SGD
from torch_geometric.data import Batch, Data
from torch_geometric.datasets import Planetoid, TUDataset
from torch_geometric.loader import DataLoader

import graph_transformer_benchmark.train as train_mod


class DummyDataset:
    """Wraps a list of Data objects and exposes length and metadata."""
    def __init__(self, data_list: List[Data]) -> None:
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


class DummyGraphEvaluator:
    """Dummy evaluator for OGB graph‐level tasks, returns fixed ROC‐AUC."""

    def __init__(self, name: str) -> None:
        self.name = name

    def eval(self, inputs: dict) -> dict:
        return {"rocauc": 0.75}


class DummyNodeEvaluator:
    """Dummy evaluator for OGB node‐level tasks, returns fixed accuracy."""

    def __init__(self, name: str) -> None:
        self.name = name

    def eval(self, inputs: dict) -> dict:
        return {"acc": 0.5}


class DummyModel(Module):
    """Model that returns perfect one‐hot logits based on batch.y,
    scaled by a learnable weight."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = Parameter(torch.tensor(1.0))

    def forward(self, batch: Data) -> Tensor:
        labels = batch.y.view(-1)
        num_classes = int(labels.max().item()) + 1
        logits = torch.zeros(
            (labels.size(0), num_classes), device=labels.device
        )
        logits[
            torch.arange(labels.size(0), device=labels.device), labels
        ] = 1.0
        return logits * self.scale


class DummyTrainLoader:
    """Wraps a DataLoader and exposes num_node_features & num_classes."""
    def __init__(
            self,
            loader: DataLoader,
            num_feat: int,
            num_cls: int
            ) -> None:
        self._loader = loader
        self.dataset = SimpleNamespace(
            num_node_features=num_feat,
            num_classes=num_cls,
        )

    def __iter__(self):
        return iter(self._loader)

    def __len__(self) -> int:
        return len(self._loader.dataset)


@pytest.fixture(autouse=True)
def patch_ogb_evaluators(monkeypatch: Any) -> Generator[None, None, None]:
    """Monkeypatch OGB Evaluators to use dummy implementations."""
    yield


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
            [0, 1, 2, 3],   # “chain” or self‑loops, etc.
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
def cfg_graph() -> DictConfig:
    """Provide a DictConfig for an OGB graph‐level dataset."""
    return OmegaConf.create({"data": {"dataset": "ogbg-molhiv"}})


@pytest.fixture
def cfg_node() -> DictConfig:
    """Provide a DictConfig for an OGB node‐level dataset."""
    return OmegaConf.create({"data": {"dataset": "ogbn-arxiv"}})


@pytest.fixture
def cfg_generic() -> DictConfig:
    """Provide a DictConfig for a generic graph dataset."""
    return OmegaConf.create({"data": {"dataset": "MUTAG"}})


@pytest.fixture
def cfg_unsupported() -> DictConfig:
    """Provide a DictConfig for an unsupported dataset name."""
    return OmegaConf.create({"data": {"dataset": "unknown"}})


@pytest.fixture
def cfg_data() -> DictConfig:
    """Provide a minimal DictConfig for train_one_epoch."""
    return OmegaConf.create({"data": {}})


@pytest.fixture
def optimizer(dummy_model: DummyModel) -> SGD:
    """Provide an SGD optimizer for the dummy_model."""
    return SGD(
        dummy_model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)


@pytest.fixture
def device() -> torch.device:
    """Provide a CPU device for testing."""
    return torch.device("cpu")


class DummyRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture(autouse=True)
def patch_training_dependencies(
    monkeypatch: Any,
    dummy_model: Module,
    generic_loader: DataLoader,
    device: torch.device,
) -> SimpleNamespace:
    """Stub out data/model/MLflow for run_training tests."""
    data_list = list(generic_loader.dataset)
    dummy_dataset = DummyDataset(data_list)
    dummy_loader = DataLoader(
        dummy_dataset,
        batch_size=generic_loader.batch_size
    )

    # Monkeypatch build_dataloaders to return our dummy_loader three times
    monkeypatch.setattr(
        train_mod,
        "build_dataloaders",
        lambda cfg, **kw: (dummy_loader, dummy_loader, dummy_loader)
    )
    monkeypatch.setattr(
        train_mod,
        "build_model",
        lambda cfg_model, nf, nc: dummy_model
    )
    monkeypatch.setattr(train_mod, "get_device", lambda dev: device)
    monkeypatch.setattr(train_mod, "init_mlflow", lambda cfg: None)
    monkeypatch.setattr(train_mod, "log_config", lambda cfg: None)
    monkeypatch.setattr(mlflow, "start_run", lambda **kw: DummyRun())
    metrics = []
    artifacts = []
    monkeypatch.setattr(
        mlflow,
        "log_metric",
        lambda k, v, step=None: metrics.append((k, v, step))
    )
    monkeypatch.setattr(
        mlflow,
        "log_artifact",
        lambda path: artifacts.append(path)
    )

    # Mock data utility functions that can be slow
    monkeypatch.setattr(
        train_mod, "log_dataset_stats", lambda *args, **kw: None)
    monkeypatch.setattr(
        train_mod, "infer_num_node_features", lambda loader: 4)
    monkeypatch.setattr(
        train_mod, "infer_num_classes", lambda loader: 2)

    def mock_evaluate(model, loader, device, cfg):
        return {"accuracy": 0.85, "val_loss": 0.25, "macro_f1": 0.80}

    monkeypatch.setattr(
        "graph_transformer_benchmark.evaluation.evaluate",
        mock_evaluate
    )
    monkeypatch.setattr(
        "graph_transformer_benchmark.training."
        "graph_transformer_trainer.evaluate",
        mock_evaluate
    )
    pytorch_mock = MagicMock()
    pytorch_mock.log_model = MagicMock()
    monkeypatch.setattr(mlflow, "pytorch", pytorch_mock)
    monkeypatch.setattr(mlflow, "log_param", lambda k, v: None)
    monkeypatch.setattr(mlflow, "log_params", lambda d: None)
    monkeypatch.setattr(mlflow, "set_tag", lambda k, v: None)
    monkeypatch.setattr(mlflow, "set_tags", lambda d: None)
    monkeypatch.setattr(mlflow, "end_run", lambda: None)
    monkeypatch.setattr(mlflow, "active_run", lambda: MagicMock())

    return SimpleNamespace(metrics=metrics, artifacts=artifacts)


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
def cfg_transformer() -> Any:
    """Minimal DictConfig for GraphTransformer builder."""
    return OmegaConf.create({
        "type": "GraphTransformer",
        "hidden_dim": 8,
        "num_layers": 2,
        "num_heads": 2,
        "dropout": 0.0,
        "ffn_hidden_dim": None,
        "activation": "relu",
        "use_super_node": False,
        "with_spatial_bias": False,
        "with_edge_bias": False,
        "with_hop_bias": False,
        "with_degree_enc": False,
        "with_eig_enc": False,
        "with_svd_enc": False,
        "gnn_conv_type": None,
        "gnn_position": "pre",
        "max_degree": 0,
        "num_spatial": 0,
        "num_edges": 0,
        "num_hops": 0,
        "num_eigenc": 0,
        "num_svdenc": 0,
    })

# Test Data


@pytest.fixture(autouse=True)
def cleanup_all(tmp_path: Path):
    """
    Remove any downloaded subfolders under tmp_path to keep filesystem clean.
    """
    yield
    for sub in ("TUD", "Planetoid", "OGB"):
        shutil.rmtree(tmp_path / sub, ignore_errors=True)


@pytest.fixture(params=[
    ("MUTAG", TUDataset, "TUD"),
    ("proteins", TUDataset, "TUD"),
    ("cora", Planetoid, "Planetoid"),
    ("PubMed", Planetoid, "Planetoid"),
])
def dataset_info(request, tmp_path: Path):
    """
    Provides (name, expected_class, expected_subdir, tmp_path) tuples.
    """
    name, expected_cls, subdir = request.param
    return name, expected_cls, subdir, tmp_path


@pytest.fixture(params=[
    ("MUTAG", TUDataset),
    ("PubMed", Planetoid),
])
def generic_cfg_and_cls(request, tmp_path: Path):
    """
    Provides (cfg, expected_class) for generic dataset splits.
    """
    name, expected_cls = request.param
    cfg = OmegaConf.create({
        "data": {
            "dataset": name,
            "root": str(tmp_path),
            "batch_size": 3,
            "num_workers": 0,
        }
    })
    return cfg, expected_cls


@pytest.fixture
def ogb_graph_dataset():
    """
    Fixtures a MagicMock OGB graph-level dataset with three sample graphs
    and explicit train/valid/test splits.
    """
    mock_ds = MagicMock(spec=PygGraphPropPredDataset)

    # Create sample graphs with all required attributes
    g1 = Data(
        x=torch.randn(2, 5),
        edge_index=torch.tensor([[0], [1]]),
        edge_attr=torch.randn(1, 4),
        y=torch.tensor([[0]])
    )
    g2 = Data(
        x=torch.randn(3, 5),
        edge_index=torch.tensor([[0, 1], [1, 2]]),
        edge_attr=torch.randn(2, 4),
        y=torch.tensor([[1]])
    )
    g3 = Data(
        x=torch.randn(4, 5),
        edge_index=torch.tensor([[0, 2], [2, 3]]),
        edge_attr=torch.randn(2, 4),
        y=torch.tensor([[1]])
    )
    graphs = [g1, g2, g3]

    # Mock get_idx_split to return proper indices
    mock_ds.get_idx_split.return_value = {
        "train": torch.tensor([0]),
        "valid": torch.tensor([1]),
        "test": torch.tensor([2])
    }

    # Mock data loading behavior
    mock_ds.data = graphs[0]  # First graph as data
    mock_ds.slices = None  # Not needed for our test

    # Handle __getitem__ directly without file loading
    def getitem(idx):
        if isinstance(idx, (list, torch.Tensor)):
            if isinstance(idx, torch.Tensor):
                idx = idx.tolist()
            return [graphs[i] for i in idx]
        if isinstance(idx, slice):
            return graphs[idx]
        return graphs[idx]

    mock_ds.__getitem__.side_effect = getitem
    mock_ds.__len__.return_value = len(graphs)

    # Mock OGB-specific attributes
    mock_ds.num_tasks = 1
    mock_ds.task_type = "binary classification"
    mock_ds.eval_metric = "rocauc"

    return mock_ds, graphs


@pytest.fixture
def ogb_node_dataset():
    """
    Fixtures a MagicMock OGB node-level dataset (single graph)
    with train/valid/test masks across nodes.
    """
    mock_ds = MagicMock()
    num_nodes = 4
    data = Data(
        x=torch.randn(num_nodes, 8),
        edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        y=torch.randint(0, 3, (num_nodes,)),
    )
    train_idx = torch.tensor([0, 2])
    valid_idx = torch.tensor([1])
    test_idx = torch.tensor([3])
    mock_ds.get_idx_split.return_value = {
        "train": train_idx,
        "valid": valid_idx,
        "test":  test_idx,
    }
    mock_ds.__getitem__.return_value = data
    mock_ds.__len__.return_value = 1
    return mock_ds, data, train_idx, valid_idx, test_idx


# ----------------------------------------------------------------------
# Data Generation Factory Fixtures
# ----------------------------------------------------------------------

@pytest.fixture
def make_node_data():
    """Factory fixture for creating node-level task data.

    Args:
        num_nodes: Number of nodes in the graph
        in_features: Number of input features per node
        num_targets: For classification = number of classes
                    For regression = number of target dimensions
        task_type: Either 'binary', 'multiclass', or 'regression'
    """
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
    """Factory fixture for creating graph-level task datasets.

    Args:
        num_graphs: Number of graphs in dataset
        in_features: Number of input features per node
        num_targets: For classification = number of classes
                    For regression = number of target dimensions
        task_type: Either 'classification' or 'regression'
        is_binary: If True & task_type='classification', creates binary targets
        min_nodes: Minimum nodes per graph
        max_nodes: Maximum nodes per graph
    """
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
