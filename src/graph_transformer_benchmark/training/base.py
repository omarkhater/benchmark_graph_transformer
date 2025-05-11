import copy
import logging
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from graph_transformer_benchmark.utils import log_health_metrics

from .tqdm_configuration import TqdmManager


class BaseTrainer(ABC):
    """
    Base trainer for Pytorch based neural networks.

    This class provides a standard training loop with optional validation,
    early stopping, and learning rate scheduling. It is designed to be
    inherited by specific model trainers that implement the training
    and validation logic.

    Args:
        model (nn.Module): The neural network model to be trained.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        device (torch.device): Device to run the model on ('cpu' or 'cuda').
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]):
            Learning rate scheduler.
        num_epochs (int): Maximum number of epochs to train.
        val_frequency (int): Frequency of validation (in epochs).
        patience (int): Number of epochs with no improvement after which
            training will be stopped.
        log_interval (int): Interval for logging training loss.

    Attributes:
        model (nn.Module): The neural network model.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        device (torch.device): Device to run the model on.
        scheduler (Optional[torch.optim.lr_scheduler._LRScheduler]):
            Learning rate scheduler.
        num_epochs (int): Maximum number of epochs to train.
        val_frequency (int): Frequency of validation (in epochs).
        patience (int): Number of epochs with no improvement after which
            training will be stopped.
        log_interval (int): Interval for logging training loss.
        best_loss (float): Best validation loss observed during training.
        best_epoch (int): Epoch at which the best validation loss was observed.
        best_state (Dict[str, Any]): State of the model at the best epoch.
        train_losses (List[float]): List of training losses for each epoch.
        val_losses (List[float]): List of validation losses for each epoch.

    Methods:

        train(): Main training loop.
        validate(): Validation loop.
        train_epoch(): Train for one epoch.
        validate_batch(): Validate a batch of data.
        calculate_loss(): Calculate loss for a batch of data.
        _gather_tensors(): Gather tensors to be used with calculate_loss.
        _check_early_stopping(): Check if early stopping criteria is met.
        after_validation(): Hook method called after validation is complete.
        cleanup_gpu(): Moves the model to CPU and clears GPU memory.

    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_epochs: int = 100,
        val_frequency: int = 5,
        patience: int = 5,
        curriculum_keys: Optional[List[str]] = None,
    ):
        self._validate_inputs(
            model, train_loader, val_loader, optimizer, device,
            scheduler, num_epochs, val_frequency, patience
        )

        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        if not (val_frequency > num_epochs):
            self.val_frequency = val_frequency
        self.patience = patience
        self.best_loss = float('inf')
        self.best_epoch = 0
        self.best_state = copy.deepcopy(self.model.state_dict())
        self.train_losses = []
        self.val_losses = []
        if curriculum_keys is not None:
            self.curriculum_keys = curriculum_keys
        else:
            self.curriculum_keys = []
        self._val_raws = {}

    def _validate_inputs(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        num_epochs: int,
        val_frequency: int,
        patience: int
    ):
        """
        Validates the inputs passed to the trainer.

        Raises:
            TypeError or ValueError: if wrong input type or invalid.
        """
        if not isinstance(model, nn.Module):
            raise TypeError(
                f"Expected 'model' to be an instance of nn.Module, "
                f"got {type(model)}"
            )
        if not isinstance(train_loader, DataLoader):
            raise TypeError(
                f"Expected 'train_loader' to be a DataLoader, "
                f"got {type(train_loader)}"
            )
        if not isinstance(val_loader, DataLoader):
            raise TypeError(
                f"Expected 'val_loader' to be a DataLoader, "
                f"got {type(val_loader)}"
            )
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                f"Expected 'optimizer' to be a torch.optim.Optimizer, "
                f"got {type(optimizer)}"
            )
        if not isinstance(device, torch.device):
            raise TypeError(
                f"Expected 'device' to be a torch.device, "
                f"got {type(device)}"
            )
        if scheduler is not None:
            if not hasattr(scheduler, "step") or not callable(scheduler.step):
                raise TypeError(
                    "Expected 'scheduler' to have a callable 'step()' method, "
                    f"got {scheduler}"
                )
            if not hasattr(scheduler, "state_dict") or not callable(
                scheduler.state_dict
            ):
                raise TypeError(
                    "Expected 'scheduler' to have a callable 'state_dict()' "
                    f"method, got {scheduler}"
                )
        if not isinstance(num_epochs, int) or num_epochs <= 0:
            raise ValueError("num_epochs must be a positive integer")
        if not isinstance(val_frequency, int) or val_frequency <= 0:
            raise ValueError("val_frequency must be a positive integer")
        if not isinstance(patience, int) or patience <= 0:
            raise ValueError("patience must be a positive integer")

        if val_frequency > num_epochs:
            self.val_frequency = num_epochs - 1
            logging.warning(
                f"val_frequency ({val_frequency}) is greater than "
                f"num_epochs ({num_epochs}). "
                f"Setting val_frequency to {self.val_frequency}."
            )

    @abstractmethod
    def train_epoch(self) -> float:
        """
        Train for one epoch.
        Must be implemented by the inherited class.
        """

    @abstractmethod
    def validate_batch(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Validate a batch of data.
        Must be implemented by the inherited class.
        """

    @abstractmethod
    def calculate_loss(self, *args) -> float:
        """
        Calculate loss for a batch of data.
        Must be implemented by the inherited class.
        """

    @abstractmethod
    def _gather_tensors(self, *args) -> torch.Tensor:
        """
        Gather tensors to be used with calculate_loss.
        Must be implemented by the inherited class.
        """

    def train(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        training loop with validation, early stopping, and LR scheduling.

        Returns:
            - model: The best trained model.
            - metrics: A dictionary containing training and validation losses.
        """
        try:
            prefix = "loss/train"
            num_bad = 0
            epoch_bar = TqdmManager(
                total=self.num_epochs,
                desc="Training Epochs",
                leave=True,
                unit="epoch",
            )
            for epoch in range(self.num_epochs):
                loss_dict = self.train_epoch()

                train_loss = loss_dict.get("total_loss")
                if train_loss is None:
                    raise ValueError(
                        "Training loss not found in loss_dict. "
                        "Does train_epoch return a key = total_loss?"
                    )

                self.train_losses.append(train_loss)
                epoch_bar.set_postfix(train_loss=f"{train_loss:.4f}")
                if mlflow.active_run():
                    mlflow.log_metric(
                        f"{prefix}/total", train_loss, step=epoch)
                    for key, value in loss_dict.items():
                        mlflow.log_metric(f"{prefix}/{key}", value, step=epoch)
                if epoch > 0 and epoch % self.val_frequency == 0:
                    val_loss = self.validate(epoch+1)
                    self.val_losses.append(val_loss)
                    logging.info(
                        f"Epoch {epoch+1}/{self.num_epochs}: "
                        f"train={train_loss:.4f}, val={val_loss:.4f}"
                    )
                    epoch_bar.set_postfix(
                        train_loss=f"{train_loss:.4f}",
                        val_loss=f"{val_loss:.4f}"
                    )
                    stop, num_bad = self._check_early_stopping(
                        epoch, val_loss, num_bad)
                    if stop:
                        break
                    if self.scheduler is not None:
                        logging.info(
                            f"Step scheduler at epoch {epoch+1} "
                            f"with val_loss={val_loss:.4f}"
                        )
                        self.scheduler.step(val_loss)
                else:
                    # Enable access all validation metrics by best epoch index
                    self.all_val_metrics.append([])
                log_health_metrics(self.model, self.optimizer, epoch)
                epoch_bar.update(1)

            epoch_bar.close()
            self.model.load_state_dict(self.best_state)
            metrics = {
                'train_losses': self.train_losses,
                'val_losses': self.val_losses,
                'best_loss': self.best_loss,
                'best_epoch': self.best_epoch
            }
            self.metrics = metrics
            self.after_training()
            return self.model, metrics
        except Exception as e:
            logging.error(f"An error occurred during training: {e}")
            traceback.print_exc()
            return None, {}
        finally:
            self.cleanup_gpu()

    def _check_early_stopping(
            self,
            epoch: int,
            val_loss: float,
            num_bad: int
            ) -> Tuple[bool, int]:
        """
        Checks if early stopping criteria is met.

        Args:
            epoch: Current epoch number
            val_loss: Current validation loss
            num_bad: Number of epochs without improvement

        Returns:
            Tuple of (should_stop, new_num_bad)
        """
        if val_loss < self.best_loss - 1e-4:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(self.model.state_dict())
            num_bad = 0
        else:
            num_bad += 1

        # Only stop if we've exceeded patience
        if num_bad > self.patience:
            logging.info(
                f"Early stopping at epoch {epoch+1} after "
                f"{num_bad} epochs without improvement."
            )
            return True, num_bad
        return False, num_bad

    def validate(self, current_epoch: int) -> float:
        """
        Validation loop that uses validate_batch to compute loss.
        Provides a standard validation workflow while allowing
        custom batch processing logic in child classes.

        Args:
            current_epoch (int): The current epoch number.

        Returns:
            float: Average validation loss or NaN if validation set is empty
        """
        if len(self.val_loader) == 0:
            self._avg_val_raws = {
                key: float("nan") for key in self.curriculum_keys
            }
            self.after_validation(current_epoch)
            return float("nan")

        self.model.eval()
        running_loss = 0.0
        self._val_raws = {k: [] for k in self.curriculum_keys}

        with torch.no_grad():
            for batch_data in self.val_loader:
                batch_data = self.ensure_on_device(batch_data)
                losses = self.validate_batch(batch_data)
                loss = losses.get("total_loss")
                if loss is None:
                    raise ValueError(
                        "Validation loss not found in losses. "
                        "Does validate_batch return a key = total_loss?"
                    )
                if isinstance(loss, float):
                    running_loss += loss
                else:
                    running_loss += loss.item()

                raw_losses = losses.get("raw_losses")
                if raw_losses is None:
                    raise ValueError(
                        "Validation raw losses not found in losses. "
                        "Does validate_batch return a key = raw_losses?"
                    )

                for key, value in self._val_raws.items():
                    if key in raw_losses:
                        value_tensor = raw_losses[key]
                        if not isinstance(value_tensor, torch.Tensor):
                            raise ValueError(
                                f"Raw loss for {key} must be a tensor")
                        value.append(value_tensor.item())
                    else:
                        raise ValueError(
                            f"Key {key} not found in raw_losses. "
                            f"Does calculate_loss return a key = {key} "
                            "under raw_losses?"
                        )

        self._avg_val_raws = {
            key: (sum(vals) / len(vals) if vals else float("nan"))
            for key, vals in self._val_raws.items()
        }
        self.after_validation(current_epoch)

        self.model.train()
        return running_loss / len(self.val_loader)

    def after_validation(self, current_epoch: int):
        """
        Hook method called after validation is complete.
        Child classes can override this to add visualizations
        or other post-validation processing.

        Args:
            current_epoch (int): The current epoch number.
        """

    def cleanup_gpu(self):
        """
        Moves the model to CPU and clears GPU memory.
        """
        self.model.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def ensure_on_device(self, data: Any) -> Any:
        """
        Ensures the data is on the correct device.

        Args:
            data: The data to move to the device.

        Returns:
            The data on the correct device.
        """
        if isinstance(data, torch.Tensor):
            return data.to(self.device)
        elif isinstance(data, (list, tuple)):
            return [self.ensure_on_device(item) for item in data]
        elif isinstance(data, dict):
            return {k: self.ensure_on_device(v) for k, v in data.items()}
        else:
            return data

    def after_training(self):
        """
        Hook method called after training is complete.
        Child classes can override this to add visualizations
        or other post-training processing.
        """
