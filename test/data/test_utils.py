"""Unit tests for utils module."""
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data, Dataset

from graph_transformer_benchmark.data import utils


def test_is_node_level_graph(one_graph_dataset):
    """Test node-level graph detection functionality."""
    assert utils.is_node_level_graph(one_graph_dataset)
    many = [one_graph_dataset[0] for _ in range(3)]
    many_ds = type(
        "Many",
        (Dataset,),
        {"__len__": lambda self: 3, "__getitem__": lambda s, i: many[i]}
    )()
    assert not utils.is_node_level_graph(many_ds)


def test_enrich_batch_smoke():
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 2, 3], [1, 3, 4]])
    data = Data(x=x, edge_index=edge_index, y=torch.zeros(5, dtype=torch.long))
    batch = Batch.from_data_list([data])
    cfg = OmegaConf.create({"with_degree_enc": True})
    enriched = utils.enrich_batch(batch, cfg)
    assert hasattr(enriched, "in_degree")
    assert enriched.x.shape == x.shape
