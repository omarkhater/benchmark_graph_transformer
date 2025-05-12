import random

import numpy as np
import pytest
import torch

import graph_transformer_benchmark.utils.seed as utils


def test_set_seed_reproducible_and_cudnn_flags(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    utils.set_seed(1234)
    r1, n1 = random.random(), np.random.rand()
    t1 = torch.randint(0, 10, (1,)).item()
    utils.set_seed(1234)  # Reset
    assert random.random() == pytest.approx(r1)
    assert np.random.rand() == pytest.approx(n1)
    assert torch.randint(0, 10, (1,)).item() == t1
    assert torch.backends.cudnn.deterministic
    assert not torch.backends.cudnn.benchmark


def test_worker_init_fn_seeds_correctly():
    """Test that worker_init_fn creates different but deterministic seeds."""
    utils.worker_init_fn(0)  # first worker
    r1 = random.random()
    n1 = np.random.rand()
    t1 = torch.randint(0, 10, (1,)).item()

    utils.worker_init_fn(0)  # reset first worker
    assert random.random() == pytest.approx(r1)
    assert np.random.rand() == pytest.approx(n1)
    assert torch.randint(0, 10, (1,)).item() == t1

    utils.worker_init_fn(1)  # second worker should get different values
    assert random.random() != pytest.approx(r1)
    assert np.random.rand() != pytest.approx(n1)
    assert torch.randint(0, 10, (1,)).item() != t1


@pytest.mark.parametrize("cuda_available", [True, False])
def test_configure_determinism(monkeypatch, cuda_available):
    """Test that configure_determinism enables deterministic settings."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)

    utils.configure_determinism(42, cuda=cuda_available)

    # Check deterministic settings
    assert torch.are_deterministic_algorithms_enabled()

    if cuda_available:
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark

    # Verify RNG states are seeded
    r1 = random.random()
    n1 = np.random.rand()
    t1 = torch.randint(0, 10, (1,)).item()

    utils.configure_determinism(42, cuda=cuda_available)
    assert random.random() == pytest.approx(r1)
    assert np.random.rand() == pytest.approx(n1)
    assert torch.randint(0, 10, (1,)).item() == t1


@pytest.mark.parametrize("cudnn_available", [True, False])
def test_configure_determinism_with_cudnn(monkeypatch, cudnn_available):
    """Test configure_determinism with different cuDNN availability."""

    # Mock CUDA and cuDNN availability
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    mock_backends = {"cudnn": cudnn_available}
    monkeypatch.setattr(utils, "_BACKENDS_SET", mock_backends)

    # Reset to known state
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    utils.configure_determinism(42, cuda=True)

    assert torch.are_deterministic_algorithms_enabled()

    if cudnn_available:
        assert torch.backends.cudnn.deterministic
        assert not torch.backends.cudnn.benchmark
    else:
        assert not torch.backends.cudnn.deterministic
        assert torch.backends.cudnn.benchmark
