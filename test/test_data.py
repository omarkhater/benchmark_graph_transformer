import pytest
import torch
from omegaconf import OmegaConf
from torch_geometric.data import Batch, Data

from graph_transformer_benchmark.data import (
    _get_dataset,
    _load_graph_level,
    _load_node_level,
    build_dataloaders,
    enrich_batch,
)


def test_get_dataset_basic(dataset_info):
    name, expected_cls, subdir, tmp_path = dataset_info
    ds = _get_dataset(name, tmp_path)
    assert isinstance(ds, expected_cls)
    root_str = getattr(ds, "root", None) or getattr(ds, "raw_dir", "")
    assert subdir in root_str


def test_get_dataset_unsupported(tmp_path):
    with pytest.raises(ValueError) as exc:
        _get_dataset("not_real", tmp_path)
    assert "Unsupported dataset 'not_real'" in str(exc.value)


def test_build_dataloaders_generic(generic_cfg_and_cls):
    cfg, expected_cls = generic_cfg_and_cls
    train_ld, val_ld, test_ld = build_dataloaders(cfg)
    for loader in (train_ld, val_ld, test_ld):
        assert isinstance(loader.dataset, expected_cls)
        assert loader.batch_size == cfg.data.batch_size


def test_load_graph_level_split(ogb_graph_dataset):
    mock_ds, graphs = ogb_graph_dataset
    g1, g2, g3 = graphs
    train_ld, val_ld, test_ld = _load_graph_level(
        mock_ds, {"batch_size": 1, "num_workers": 0})
    assert torch.equal(train_ld.dataset.x, g1.x)
    assert torch.equal(val_ld.dataset.x,   g2.x)
    assert torch.equal(test_ld.dataset.x,  g3.x)


def test_load_node_level_masks(ogb_node_dataset):
    mock_ds, data, train_idx, valid_idx, test_idx = ogb_node_dataset
    train_ld, val_ld, test_ld = _load_node_level(
        mock_ds, {"batch_size": 1, "num_workers": 0})
    tb = next(iter(train_ld))
    vb = next(iter(val_ld))
    tb2 = next(iter(test_ld))
    assert tb.train_mask.sum().item() == len(train_idx)
    assert vb.val_mask.sum().item() == len(valid_idx)
    assert tb2.test_mask.sum().item() == len(test_idx)


def test_enrich_batch_various_flags():
    x = torch.randn(5, 3)
    edge_index = torch.tensor([[0, 2, 3], [1, 3, 4]])
    data = Data(x=x, edge_index=edge_index, y=torch.zeros(5, dtype=torch.long))
    batch = Batch.from_data_list([data])
    cfg = OmegaConf.create({
        "with_degree_enc": True,
        "with_eig_enc":    True,
        "num_eigenc":      2,
        "with_svd_enc":    True,
        "num_svdenc":      1,
        "with_spatial_bias": True,
        "with_edge_bias":    True,
        "with_hop_bias":     True,
    })
    enriched = enrich_batch(batch, cfg)
    n = x.size(0)
    assert enriched.out_degree.shape == (n,)
    assert enriched.in_degree.shape == (n,)
    assert enriched.eig_pos_emb.shape == (n, 2)
    assert enriched.svd_pos_emb.shape == (n, 2)
    assert enriched.spatial_pos.shape == (n, n)
    assert enriched.edge_dist.shape == (n, n)
    assert enriched.hop_dist.shape == (n, n)
