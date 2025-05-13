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


def test_enrich_batch_all_encodings():
    """Test batch enrichment with all encoding options enabled."""
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 2, 3], [1, 3, 4]])
    data = Data(x=x, edge_index=edge_index, y=torch.zeros(5, dtype=torch.long))
    batch = Batch.from_data_list([data])

    cfg = OmegaConf.create({
        "with_degree_enc": True,
        "with_eig_enc": True,
        "num_eigenc": 4,
        "with_svd_enc": True,
        "num_svdenc": 3,
        "with_spatial_bias": True,
        "with_edge_bias": True,
        "with_hop_bias": True
    })

    enriched = utils.enrich_batch(batch, cfg)

    # Check all encodings are present
    assert hasattr(enriched, "in_degree")
    assert hasattr(enriched, "out_degree")
    assert hasattr(enriched, "eig_pos_emb")
    assert hasattr(enriched, "svd_pos_emb")
    assert hasattr(enriched, "spatial_pos")
    assert hasattr(enriched, "edge_dist")
    assert hasattr(enriched, "hop_dist")

    # Check shapes
    assert enriched.eig_pos_emb.shape == (5, 4)  # num_nodes x num_eigenc
    assert enriched.svd_pos_emb.shape == (5, 6)  # num_nodes x (2 * num_svdenc)
    assert enriched.spatial_pos.shape == (5, 5)   # num_nodes x num_nodes
    assert enriched.edge_dist.shape == (5, 5)     # num_nodes x num_nodes
    assert enriched.hop_dist.shape == (5, 5)      # num_nodes x num_nodes


def test_enrich_batch_no_encodings():
    """Test batch enrichment with no encoding options."""
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 2, 3], [1, 3, 4]])
    data = Data(x=x, edge_index=edge_index, y=torch.zeros(5, dtype=torch.long))
    batch = Batch.from_data_list([data])

    cfg = OmegaConf.create({})  # Empty config means no encodings
    enriched = utils.enrich_batch(batch, cfg)

    # Verify original attributes are preserved
    assert torch.equal(enriched.x, batch.x)
    assert torch.equal(enriched.edge_index, batch.edge_index)

    # Verify no extra encodings were added
    assert not hasattr(enriched, "in_degree")
    assert not hasattr(enriched, "eig_pos_emb")
    assert not hasattr(enriched, "svd_pos_emb")
    assert not hasattr(enriched, "spatial_pos")
