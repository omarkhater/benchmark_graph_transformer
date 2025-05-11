"""Unit tests for BaseTrainer in graph_transformer_benchmark.training.base."""
import logging
import math
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from graph_transformer_benchmark.training.base import BaseTrainer


class SimpleModel(nn.Module):
    """A simple linear model for testing purposes."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass applying the linear transformation."""
        return self.linear(x)


class SimpleTrainer(BaseTrainer):
    """Simple implementation of BaseTrainer for testing purposes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.all_val_metrics: List[List[Any]] = []
        self.should_raise_exception_in_train_epoch: bool = False
        self.train_epoch_return_value: Dict[str, float] = {"total_loss": 0.5}
        self.validate_batch_return_value: Dict[str, Any] = {
            "total_loss": 0.1, "raw_losses": {"loss1": torch.tensor(0.1)}
        }
        self.after_validation_called_with_epoch: Optional[int] = None
        self.after_training_called: bool = False

    def train_epoch(self) -> Dict[str, float]:
        """Simulate training for one epoch.

        Returns:
            Dictionary containing loss values.

        Raises:
            ValueError: If should_raise_exception_in_train_epoch is True.
        """
        if self.should_raise_exception_in_train_epoch:
            raise ValueError("Test exception in train_epoch")
        return self.train_epoch_return_value

    def validate_batch(self, batch: Any) -> Dict[str, Any]:
        """Simulate validation for a batch of data.

        Args:
            batch: Input batch data.

        Returns:
            Dictionary with validation results.
        """
        return self.validate_batch_return_value

    def calculate_loss(self, *args: Any) -> float:
        """Simulate loss calculation.

        Returns:
            A fixed loss value for testing.
        """
        return 0.1

    def _gather_tensors(self, *args: Any) -> torch.Tensor:
        """Simulate tensor gathering.

        Returns:
            A fixed tensor for testing.
        """
        return torch.tensor([0.1])

    def after_validation(self, current_epoch: int) -> None:
        """Hook called after validation to record epoch number.

        Args:
            current_epoch: Current epoch number.
        """
        super().after_validation(current_epoch)
        self.after_validation_called_with_epoch = current_epoch

    def after_training(self) -> None:
        """Hook called after training to mark completion."""
        super().after_training()
        self.after_training_called = True


@pytest.fixture
def simple_trainer() -> SimpleTrainer:
    """Create a SimpleTrainer instance with standard test configuration.

    Returns:
        A configured SimpleTrainer instance.
    """
    device = torch.device("cpu")
    model = SimpleModel()
    x = torch.randn(10, 1)
    y = torch.randn(10, 1)
    dataset = TensorDataset(x, y)
    train_loader = DataLoader(dataset, batch_size=2)
    val_loader = DataLoader(dataset, batch_size=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    return SimpleTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        num_epochs=5,
        val_frequency=1,
        patience=2,
        curriculum_keys=["loss1"]
    )


@pytest.fixture
def simple_trainer_with_scheduler(
    simple_trainer: SimpleTrainer
) -> SimpleTrainer:
    """Add a mock scheduler to the simple_trainer fixture.

    Args:
        simple_trainer: Base SimpleTrainer instance.

    Returns:
        SimpleTrainer with mock scheduler added.
    """
    simple_trainer.scheduler = MagicMock()
    simple_trainer.scheduler.step = MagicMock()
    simple_trainer.scheduler.state_dict = MagicMock(return_value={})
    return simple_trainer


def test_initialization(simple_trainer: SimpleTrainer) -> None:
    """Test that SimpleTrainer initializes with correct attributes."""
    assert isinstance(simple_trainer.model, nn.Module)
    assert isinstance(simple_trainer.train_loader, DataLoader)
    assert isinstance(simple_trainer.val_loader, DataLoader)
    assert isinstance(simple_trainer.optimizer, torch.optim.Optimizer)
    assert simple_trainer.num_epochs == 5
    assert simple_trainer.val_frequency == 1
    assert simple_trainer.patience == 2


def test_input_validation(caplog: pytest.LogCaptureFixture) -> None:
    """Test input validation raises expected exceptions for invalid inputs."""
    # Test with invalid model type
    with pytest.raises(
        TypeError, match="Expected 'model' to be an instance of nn.Module"
    ):
        SimpleTrainer(
            model="not_a_model",
            train_loader=DataLoader(TensorDataset(torch.randn(1))),
            val_loader=DataLoader(TensorDataset(torch.randn(1))),
            optimizer=torch.optim.SGD([torch.randn(1)], lr=0.01),
            device=torch.device("cpu")
        )

    # Test with invalid train_loader type
    with pytest.raises(
        TypeError, match="Expected 'train_loader' to be a DataLoader"
    ):
        SimpleTrainer(
            model=SimpleModel(),
            train_loader="not_a_dataloader",
            val_loader=DataLoader(TensorDataset(torch.randn(1))),
            optimizer=torch.optim.SGD([torch.randn(1)], lr=0.01),
            device=torch.device("cpu")
        )

    # Test with invalid val_loader type
    with pytest.raises(
        TypeError, match="Expected 'val_loader' to be a DataLoader"
    ):
        SimpleTrainer(
            model=SimpleModel(),
            train_loader=DataLoader(TensorDataset(torch.randn(1))),
            val_loader="not_a_dataloader",
            optimizer=torch.optim.SGD([torch.randn(1)], lr=0.01),
            device=torch.device("cpu")
        )

    # Test with invalid optimizer type
    with pytest.raises(
        TypeError, match="Expected 'optimizer' to be a torch.optim.Optimizer"
    ):
        SimpleTrainer(
            model=SimpleModel(),
            train_loader=DataLoader(TensorDataset(torch.randn(1))),
            val_loader=DataLoader(TensorDataset(torch.randn(1))),
            optimizer="not_an_optimizer",
            device=torch.device("cpu")
        )

    # Test with invalid device type
    with pytest.raises(
        TypeError, match="Expected 'device' to be a torch.device"
    ):
        SimpleTrainer(
            model=SimpleModel(),
            train_loader=DataLoader(TensorDataset(torch.randn(1))),
            val_loader=DataLoader(TensorDataset(torch.randn(1))),
            optimizer=torch.optim.SGD([torch.randn(1)], lr=0.01),
            device="not_a_device"
        )

    # Test with invalid scheduler (missing step method)
    with pytest.raises(
        TypeError,
        match="Expected 'scheduler' to have a callable 'step\\(\\)' method.*"
    ):
        SimpleTrainer(
            model=SimpleModel(),
            train_loader=DataLoader(TensorDataset(torch.randn(1))),
            val_loader=DataLoader(TensorDataset(torch.randn(1))),
            optimizer=torch.optim.SGD([torch.randn(1)], lr=0.01),
            device=torch.device("cpu"),
            scheduler=object()
        )

    # Test with scheduler missing state_dict method
    class SchedulerWithStep:
        """Mock scheduler with step method but no state_dict."""
        def step(self, loss: Optional[float] = None) -> None:
            """Mock step method."""

    with pytest.raises(
        TypeError,
        match="Expected 'scheduler' to have a callable "
        "'state_dict\\(\\)' method.*"
    ):
        SimpleTrainer(
            model=SimpleModel(),
            train_loader=DataLoader(TensorDataset(torch.randn(1))),
            val_loader=DataLoader(TensorDataset(torch.randn(1))),
            optimizer=torch.optim.SGD([torch.randn(1)], lr=0.01),
            device=torch.device("cpu"),
            scheduler=SchedulerWithStep()
        )

    # Test val_frequency > num_epochs warning
    with caplog.at_level(logging.WARNING):
        trainer = SimpleTrainer(
            model=SimpleModel(),
            train_loader=DataLoader(
                TensorDataset(torch.randn(10, 1), torch.randn(10, 1))
            ),
            val_loader=DataLoader(
                TensorDataset(torch.randn(10, 1), torch.randn(10, 1))
            ),
            optimizer=torch.optim.SGD(SimpleModel().parameters(), lr=0.01),
            device=torch.device("cpu"),
            num_epochs=2,
            val_frequency=5  # greater than num_epochs
        )
    assert "val_frequency (5) is greater than num_epochs (2)." in caplog.text
    assert trainer.val_frequency == 1  # Should be adjusted to num_epochs - 1

    # Test negative num_epochs
    with pytest.raises(
        ValueError, match="num_epochs must be a positive integer"
    ):
        SimpleTrainer(
            model=SimpleModel(),
            train_loader=DataLoader(TensorDataset(torch.randn(1))),
            val_loader=DataLoader(TensorDataset(torch.randn(1))),
            optimizer=torch.optim.SGD([torch.randn(1)], lr=0.01),
            device=torch.device("cpu"),
            num_epochs=-1
        )

    # Test negative val_frequency
    with pytest.raises(
        ValueError, match="val_frequency must be a positive integer"
    ):
        SimpleTrainer(
            model=SimpleModel(),
            train_loader=DataLoader(TensorDataset(torch.randn(1))),
            val_loader=DataLoader(TensorDataset(torch.randn(1))),
            optimizer=torch.optim.SGD([torch.randn(1)], lr=0.01),
            device=torch.device("cpu"),
            val_frequency=-1
        )

    # Test negative patience
    with pytest.raises(
        ValueError, match="patience must be a positive integer"
    ):
        SimpleTrainer(
            model=SimpleModel(),
            train_loader=DataLoader(TensorDataset(torch.randn(1))),
            val_loader=DataLoader(TensorDataset(torch.randn(1))),
            optimizer=torch.optim.SGD([torch.randn(1)], lr=0.01),
            device=torch.device("cpu"),
            patience=-1
        )


def test_early_stopping(simple_trainer: SimpleTrainer) -> None:
    """Test early stopping logic under different improvement scenarios."""
    # Test improvement case - should reset num_bad
    stop, num_bad = simple_trainer._check_early_stopping(1, 0.5, 0)
    assert not stop
    assert num_bad == 0

    # Test no improvement but within patience
    stop, num_bad = simple_trainer._check_early_stopping(2, 1.0, 0)
    assert not stop
    assert num_bad == 1

    # Test just at patience limit
    stop, num_bad = simple_trainer._check_early_stopping(3, 1.0, 1)
    assert not stop
    assert num_bad == 2

    # Test exceeding patience - should trigger stop
    stop, num_bad = simple_trainer._check_early_stopping(4, 1.0, 2)
    assert stop
    assert num_bad == 3

    # Test improvement resets counter
    stop, num_bad = simple_trainer._check_early_stopping(5, 0.3, 3)
    assert not stop
    assert num_bad == 0


def test_ensure_on_device(simple_trainer: SimpleTrainer) -> None:
    """Test that ensure_on_device correctly handles different data types."""
    device = torch.device("cpu")

    # Test tensor
    tensor = torch.randn(2, 2)
    assert simple_trainer.ensure_on_device(tensor).device == device

    # Test list of tensors
    tensor_list = [torch.randn(2, 2), torch.randn(2, 2)]
    result_list = simple_trainer.ensure_on_device(tensor_list)
    assert all(t.device == device for t in result_list)

    # Test dict of tensors
    tensor_dict = {"a": torch.randn(2, 2), "b": torch.randn(2, 2)}
    result_dict = simple_trainer.ensure_on_device(tensor_dict)
    assert all(t.device == device for t in result_dict.values())

    # Test non-tensor types pass through unchanged
    assert simple_trainer.ensure_on_device(123) == 123
    assert simple_trainer.ensure_on_device("test") == "test"
    assert simple_trainer.ensure_on_device(None) is None


@patch('mlflow.active_run', MagicMock(return_value=True))
@patch('mlflow.log_metric', MagicMock())
def test_train_workflow(simple_trainer: SimpleTrainer) -> None:
    """Test the main training workflow completes successfully."""
    model, metrics = simple_trainer.train()

    # Validate training results
    assert model is not None
    assert "train_losses" in metrics
    assert "val_losses" in metrics
    assert "best_loss" in metrics
    assert "best_epoch" in metrics
    assert len(metrics["train_losses"]) > 0
    assert len(metrics["val_losses"]) > 0


@patch('mlflow.active_run', MagicMock(return_value=False))
def test_train_workflow_train_loss_none(
    simple_trainer: SimpleTrainer,
    caplog: pytest.LogCaptureFixture
) -> None:
    """Test training handles missing total_loss in train_epoch result."""
    simple_trainer.train_epoch_return_value = {"other_loss": 0.5}

    with caplog.at_level(logging.ERROR):
        model, metrics = simple_trainer.train()

    # Check error handling behavior
    assert model is None
    assert metrics == {}
    assert "Training loss not found in loss_dict" in caplog.text
    assert "Does train_epoch return a key = total_loss?" in caplog.text


@patch('mlflow.active_run', MagicMock(return_value=True))
@patch('mlflow.log_metric', MagicMock())
def test_train_workflow_with_scheduler(
    simple_trainer_with_scheduler: SimpleTrainer
) -> None:
    """Test that scheduler.step is called during training."""
    trainer = simple_trainer_with_scheduler
    trainer.val_frequency = 1
    trainer.num_epochs = 2

    model, metrics = trainer.train()

    assert model is not None
    assert trainer.scheduler.step.call_count > 0


@patch('mlflow.active_run', MagicMock(return_value=False))
def test_train_workflow_exception_handling(
    simple_trainer: SimpleTrainer,
    caplog: pytest.LogCaptureFixture
) -> None:
    """Test that exceptions in train_epoch are handled gracefully."""
    simple_trainer.should_raise_exception_in_train_epoch = True

    with caplog.at_level(logging.ERROR):
        model, metrics = simple_trainer.train()

    assert model is None
    assert metrics == {}
    assert "An error occurred during training: Test exception in train_epoch"\
        in caplog.text


def test_validate_workflow_loss_none(
    simple_trainer: SimpleTrainer
) -> None:
    """Test validation error when total_loss is missing."""
    simple_trainer.validate_batch_return_value = {
        "other_loss": 0.1, "raw_losses": {"loss1": torch.tensor(0.1)}
    }

    with pytest.raises(
            ValueError, match="Validation loss not found in losses"
            ):
        simple_trainer.validate(current_epoch=1)


def test_validate_workflow_raw_losses_none(
    simple_trainer: SimpleTrainer
) -> None:
    """Test validation error when raw_losses is missing."""
    simple_trainer.validate_batch_return_value = {"total_loss": 0.1}
    simple_trainer.curriculum_keys = ["loss1"]

    with pytest.raises(
        ValueError, match="Validation raw losses not found in losses"
    ):
        simple_trainer.validate(current_epoch=1)


def test_validate_workflow_raw_loss_not_tensor(
    simple_trainer: SimpleTrainer
) -> None:
    """Test validation error when a raw_loss value is not a tensor."""
    simple_trainer.curriculum_keys = ["loss1"]
    simple_trainer.validate_batch_return_value = {
        "total_loss": 0.1,
        "raw_losses": {"loss1": 0.1}  # Not a tensor
    }

    with pytest.raises(
        ValueError, match="Raw loss for loss1 must be a tensor"
            ):
        simple_trainer.validate(current_epoch=1)


def test_validate_workflow_key_not_in_raw_losses(
    simple_trainer: SimpleTrainer
) -> None:
    """Test validation error when no curriculum key in raw_losses."""
    simple_trainer.curriculum_keys = ["missing_key"]
    simple_trainer.validate_batch_return_value = {
        "total_loss": 0.1,
        "raw_losses": {"loss1": torch.tensor(0.1)}
    }

    with pytest.raises(
        ValueError, match="Key missing_key not found in raw_losses"
    ):
        simple_trainer.validate(current_epoch=1)


@patch('mlflow.active_run', MagicMock(return_value=False))
def test_train_loop_skipped_validation_coverage(
    simple_trainer: SimpleTrainer
) -> None:
    """Test all_val_metrics is updated correctly when validation is skipped."""
    simple_trainer.num_epochs = 2
    simple_trainer.val_frequency = 3  # Won't trigger validation in 2 epochs

    initial_all_val_metrics_len = len(simple_trainer.all_val_metrics)
    simple_trainer.train()

    # Check that empty lists were appended for each epoch
    assert len(simple_trainer.all_val_metrics) == \
        initial_all_val_metrics_len + 2
    assert simple_trainer.all_val_metrics == [[], []]


def test_validate_workflow_loss_is_tensor(
    simple_trainer: SimpleTrainer
) -> None:
    """Test validation when total_loss is a tensor."""
    simple_trainer.validate_batch_return_value = {
        "total_loss": torch.tensor(0.1),
        "raw_losses": {"loss1": torch.tensor(0.1)}
    }

    assert len(simple_trainer.val_loader) > 0
    val_loss = simple_trainer.validate(current_epoch=1)

    assert isinstance(val_loss, float)
    assert val_loss == pytest.approx(0.1)


def test_validate_workflow_empty_val_loader_with_curriculum(
    simple_trainer: SimpleTrainer
) -> None:
    """Test validation with empty validation loader and curriculum keys."""
    simple_trainer.curriculum_keys = ["loss1"]
    simple_trainer.val_loader = DataLoader(
        TensorDataset(torch.empty(0, 1), torch.empty(0, 1))
    )

    val_loss = simple_trainer.validate(current_epoch=1)

    # Check NaN handling
    assert "loss1" in simple_trainer._avg_val_raws
    assert math.isnan(simple_trainer._avg_val_raws["loss1"])
    assert math.isnan(val_loss)


@patch('mlflow.active_run', MagicMock(return_value=True))
@patch('mlflow.log_metric', MagicMock())
def test_hooks_called_during_train(simple_trainer: SimpleTrainer) -> None:
    """Test that after_validation and after_training hooks are called."""
    simple_trainer.num_epochs = 2
    simple_trainer.val_frequency = 1  # Ensure validation happens

    simple_trainer.train()

    # Check that hooks were called
    assert simple_trainer.after_validation_called_with_epoch == 2  # epoch + 1
    assert simple_trainer.after_training_called is True


def test_cleanup(simple_trainer: SimpleTrainer) -> None:
    """Test that cleanup_gpu moves model to CPU."""
    simple_trainer.cleanup_gpu()
    assert next(simple_trainer.model.parameters()).device == \
        torch.device("cpu")
