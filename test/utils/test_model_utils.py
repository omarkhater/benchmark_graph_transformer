import pytest
import torch
from omegaconf import OmegaConf
from torch import nn
from torch_geometric.data import Batch, Data

from graph_transformer_benchmark.utils.model_utils import (
    BatchEnrichedModel,
    build_run_name,
    infer_task_and_loss,
)


def test_batch_enriched_model_forwards_correctly(
        dummy_model,
        cfg_transformer,
        simple_graph
        ):
    model = BatchEnrichedModel(dummy_model, cfg_transformer, device="cpu")
    out = model(simple_graph)
    # DummyModel returns one-hot logits based on label value
    expected = torch.zeros((simple_graph.num_nodes, simple_graph.y.max() + 1))
    expected[torch.arange(simple_graph.num_nodes), 0] = 1.0
    assert torch.allclose(out, expected)


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
