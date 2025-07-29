import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import Batch, Data

from graph_transformer_benchmark.utils.model_utils import (
    BatchEnrichmentWrapper,
    build_run_name,
    infer_task_and_loss,
)


class TestBatchEnrichmentWrapper:
    """Validate BatchEnrichmentWrapper on node/graph × cls/reg tasks."""

    # ------------------------------ constants --------------------------------
    _CFG_MODES = ["minimal", "bias_only", "positional_only",
                  "gnn_only", "all_features"]

    _NODE_CLASS_BATCHES = ["masked_node_batch", "cora_style_batch"]
    _GRAPH_CLASS_BATCHES = ["graph_batch", "generic_batch"]
    _NODE_REG_BATCHES = ["node_regression_batch"]
    _GRAPH_REG_BATCHES = ["regression_none_x_batch"]

    # ------------------------------ helpers ----------------------------------
    @staticmethod
    def _one_hot(labels: torch.Tensor, device) -> torch.Tensor:
        labels = labels.view(-1)
        k = int(labels.max().item()) + 1
        logits = torch.zeros(labels.size(0), k, device=device)
        logits[torch.arange(labels.size(0), device=device), labels] = 1.0
        return logits

    def _run_cls(
        self, model_factory, cfg_transformer, data_batch: Batch, device
    ):
        """Run classification model and validate output."""
        labels = data_batch.y.view(-1)
        num_classes = int(labels.max().item()) + 1
        model = model_factory(num_classes)
        wrapper = BatchEnrichmentWrapper(model, cfg_transformer, device=device)

        out = wrapper(data_batch)
        expected = self._one_hot(labels, out.device)

        assert out.shape == expected.shape
        assert torch.allclose(out, expected)

    def _run_reg(
        self, model_factory, cfg_transformer, data_batch: Batch, device
    ):
        """Run regression model and validate output."""
        y_true = data_batch.y.float()
        if y_true.ndim == 1:
            y_true = y_true.view(-1, 1)

        model = model_factory(out_dim=y_true.size(1))
        wrapper = BatchEnrichmentWrapper(model, cfg_transformer, device=device)

        out = wrapper(data_batch)
        assert out.shape == y_true.shape
        assert torch.allclose(out, y_true)

    # ------------------------------ tests ------------------------------------
    @pytest.mark.parametrize("cfg_transformer", _CFG_MODES, indirect=True)
    @pytest.mark.parametrize("data_batch", _NODE_CLASS_BATCHES, indirect=True)
    def test_node_classification(
        self, cfg_transformer, data_batch, device, node_classifier_model
    ):
        self._run_cls(node_classifier_model, cfg_transformer,
                      data_batch, device)

    @pytest.mark.parametrize("cfg_transformer", _CFG_MODES, indirect=True)
    @pytest.mark.parametrize("data_batch", _GRAPH_CLASS_BATCHES, indirect=True)
    def test_graph_classification(
        self, cfg_transformer, data_batch, device, graph_classifier_model
    ):
        self._run_cls(graph_classifier_model, cfg_transformer,
                      data_batch, device)

    @pytest.mark.parametrize("cfg_transformer", _CFG_MODES, indirect=True)
    @pytest.mark.parametrize("data_batch", _NODE_REG_BATCHES, indirect=True)
    def test_node_regression(
        self, cfg_transformer, data_batch, device, regressor_model
    ):
        self._run_reg(regressor_model, cfg_transformer, data_batch, device)

    @pytest.mark.parametrize("cfg_transformer", _CFG_MODES, indirect=True)
    @pytest.mark.parametrize("data_batch", _GRAPH_REG_BATCHES, indirect=True)
    def test_graph_regression(
        self, cfg_transformer, data_batch, device, regressor_model
    ):
        self._run_reg(regressor_model, cfg_transformer, data_batch, device)


@pytest.mark.parametrize("config,expected_name", [
    ({"type": "GraphTransformer"}, "GraphTransformer-Vanilla"),
    (
        {
            "type": "GraphTransformer",
            "with_spatial_bias": True,
            "with_edge_bias": True
        },
        "GraphTransformer-Bias-Spatial-Edge"
    ),
    (
        {
            "type": "GraphTransformer",
            "with_degree_enc": True,
            "with_eig_enc": True
        },
        "GraphTransformer-Encoding-Degree-Eig"
    ),
    (
        {
            "type": "GraphTransformer",
            "gnn_conv_type": "gcn",
            "gnn_position": "pre"
        },
        "GraphTransformer-GNN-GCN-Pre"
    ),
])
def test_build_run_name(config, expected_name):
    cfg = OmegaConf.create({"model": config})
    assert build_run_name(cfg) == expected_name


@pytest.mark.parametrize("labels,expected_task,expected_loss", [
    # Multi-label case
    (
        torch.tensor([[0., 1.], [1., 0.]]),
        "multilabel",
        nn.BCEWithLogitsLoss
    ),
    # Multi-class case
    (
        torch.tensor([0, 1, 2]),
        "multiclass",
        nn.CrossEntropyLoss
    ),
    # Regression case
    (
        torch.tensor([0.5, 1.5, 2.5]),
        "regression",
        nn.MSELoss
    ),
])
def test_infer_task_and_loss(labels, expected_task, expected_loss):
    data = Data(x=torch.randn(3, 4), y=labels)
    loader = [Batch.from_data_list([data])]
    task, loss_fn = infer_task_and_loss(iter(loader))

    assert task == expected_task
    assert isinstance(loss_fn, expected_loss)
