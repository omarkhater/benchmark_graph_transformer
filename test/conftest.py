from types import SimpleNamespace
from typing import Any, Generator, List

import mlflow
import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim import SGD
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

import graph_transformer_benchmark.evaluate as eval_mod
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
    monkeypatch.setattr(eval_mod, "GraphEvaluator", DummyGraphEvaluator)
    monkeypatch.setattr(eval_mod, "NodeEvaluator", DummyNodeEvaluator)
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
    """Provide a DataLoader for two graph‐level samples with labels [0],[1]."""
    d0 = Data(x=torch.randn(1, 4), y=torch.tensor([[0]]))
    d1 = Data(x=torch.randn(1, 4), y=torch.tensor([[1]]))
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
    return SGD(dummy_model.parameters(), lr=0.1)


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

    # Monkeypatch build_dataloaders to return our dummy_loader twice
    monkeypatch.setattr(
        train_mod,
        "build_dataloaders",
        lambda cfg, **kw: (dummy_loader, dummy_loader),
    )
    monkeypatch.setattr(
        train_mod,
        "build_model",
        lambda cfg_model, nf, nc: dummy_model
    )
    monkeypatch.setattr(train_mod, "get_device", lambda dev: device)
    monkeypatch.setattr(train_mod, "init_mlflow", lambda cfg: None)
    monkeypatch.setattr(train_mod, "log_config", lambda cfg: None)
    monkeypatch.setattr(
        train_mod,
        "log_health_metrics",
        lambda m, o, e: None
    )
    monkeypatch.setattr(
        train_mod,
        "evaluate",
        lambda model, loader, dev, cfg: 0.42
    )

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
    """Batch of two identical graphs, for whole‑graph models."""
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
