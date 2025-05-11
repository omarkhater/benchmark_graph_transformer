import pytest
import torch

from graph_transformer_benchmark.utils.device import get_device


@pytest.mark.parametrize(
    "cuda_avail, expected",
    [
        (True,  torch.device("cuda")),
        (False, torch.device("cpu")),
    ],
)
def test_get_device_respects_preference(monkeypatch, cuda_avail, expected):
    monkeypatch.setattr(torch.cuda, "is_available",
                        lambda: cuda_avail)
    device = get_device("cuda")
    assert device == expected
