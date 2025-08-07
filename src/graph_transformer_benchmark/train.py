"""GraphTransformer benchmark – single‑run training entry‑point.

This module is meant to be imported (or run) by CLI wrappers.  It builds the
PyG dataloaders, instantiates the requested model, and hands control over to
:class:`graph_transformer_benchmark.training.GraphTransformerTrainer`, which
contains the generic training / validation loop.
"""
from __future__ import annotations

import logging
import traceback

import mlflow
import torch
from torch_geometric.loader import DataLoader

from graph_transformer_benchmark.data import build_dataloaders
from graph_transformer_benchmark.evaluation import (
    detect_task_type,
)
from graph_transformer_benchmark.graph_models import build_model
from graph_transformer_benchmark.training.graph_transformer_trainer import (
    GraphTransformerTrainer,
)
from graph_transformer_benchmark.utils import (
    build_run_name,
    compute_max_degree,
    configure_determinism,
    create_model,
    get_device,
    infer_num_targets,
    init_mlflow,
    log_config,
    log_dataset_stats,
    set_seed,
    update_training_pipeline_config,
    worker_init_fn,
)


def update_max_degree(
        cfg: dict,
        data_loader: DataLoader,
        safety_buffer: int = 5
        ) -> None:
    """Update the max_degree in the configuration based on the training data.

    This function modifies the `degree_cfg` in the configuration to ensure that
    the `max_degree` is set to a value that is safe for training, based on
    the actual maximum degree observed in the training data. It adds a small
    buffer to the observed maximum degree to avoid CUDA device-side assert
    errors.

    Parameters
    ----------
    cfg : dict
        The configuration dictionary containing the model and encoder settings.
    data_loader : DataLoader
        The DataLoader containing the training data.
    safety_buffer : int, optional
        A small buffer to add to the observed maximum degree to ensure safety.
        Default is 5.

    Returns
    -------
    None
        The function modifies the `cfg` dictionary in place.
    """
    degree_cfg = cfg.get(
        "model", {}
        ).get(
            "encoder_cfg", {}
        ).get(
            "positional", {}
        ).get("degree", {})
    if degree_cfg.get("enabled", False):
        actual_max_degree = compute_max_degree(data_loader)
        degree_cfg["max_degree"] = actual_max_degree + safety_buffer
        logging.info(
            f"Updated max_degree to {degree_cfg['max_degree']} "
            f"based on training data. "
            f"Value set was {degree_cfg.get('max_degree', 'unknown')}"
        )


def run_training(cfg: dict) -> float:
    """Launch training for a single model/dataset pair.

    The function is designed to be called programmatically from a CLI front‑end
    that parses the Hydra configuration. All heavy lifting—early stopping,
    validation scheduling, MLflow logging—is performed by
    :class:`~graph_transformer_benchmark.training.GraphTransformerTrainer`.

    Parameters
    ----------
    cfg: dict
        Hydra configuration dictionary. It should contain the following keys:
        - `model`: model configuration, including type and task.
        - `training`: training configuration, including device, seed, and
            MLflow settings.
        - `data`: dataset configuration, including dataset name and split
            parameters.
        - `task`: task type, either "graph" or "node". This is used to
            determine the model's task and is expected to match the model's
            configuration.

    Returns
    -------
    float
        Best validation loss observed during training.
    """
    try:
        training_seed = cfg.get("training", {}).get("seed", 1)
        set_seed(training_seed)
        global _GLOBAL_SEED
        _GLOBAL_SEED = training_seed

        configure_determinism(training_seed, torch.cuda.is_available())

        init_mlflow(cfg)
        mlflow_cfg = cfg.get("training", {}).get("mlflow", {})
        run_name = mlflow_cfg.get("run_name", None)
        description = mlflow_cfg.get("description", None)
        if run_name is None:
            run_name = build_run_name(cfg)
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag("mlflow.note.content", description)
            log_config(cfg)
            generator = torch.Generator().manual_seed(training_seed)
            train_loader, val_loader, test_loader = build_dataloaders(
                cfg,
                generator=generator,
                worker_init_fn=worker_init_fn,
            )
            for split, loader in (
                ("train", train_loader),
                ("val", val_loader),
                ("test", test_loader),
            ):
                log_dataset_stats(loader, split, log_to_mlflow=True)

            device = get_device(cfg.get("training", {}).get("device", "cpu"))
            task = detect_task_type(train_loader)
            update_training_pipeline_config(cfg, task)
            num_classes = infer_num_targets(train_loader, task)
            logging.info(f"Detected task: {task}, num_classes: {num_classes}")
            model_cfg = cfg.get("model", {})
            update_max_degree(cfg, train_loader)
            sample_batch = next(iter(train_loader))
            model = create_model(
                model_fn=build_model,
                model_cfg=model_cfg,
                sample_batch=sample_batch,
                num_classes=num_classes,
                device=device)
            logging.info(f"Model created: {model}")
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=cfg.get("training", {}).get("lr", 0.001),
                weight_decay=cfg.get(
                    "training", {}).get("weight_decay", 0.0001),
            )

            trainer = GraphTransformerTrainer(
                cfg=cfg,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                test_loader=test_loader,
                optimizer=optimizer,
                device=device,
                num_epochs=cfg.get("training", {}).get("epochs", 1),
                val_frequency=cfg.get("training", {}).get("val_frequency", 1),
                patience=cfg.get("training", {}).get("patience", 1),
            )

            _, metrics = trainer.train()
            return float(metrics.get("best_loss", float("nan")))
    except Exception as e:
        logging.error(
            f"Training failed with exception: {e}",
            exc_info=True,
        )
        mlflow.log_param("training_status", "failed")
        mlflow.log_param("exception", str(e))
        mlflow.log_param("exception_type", type(e).__name__)
        mlflow.log_param("traceback", traceback.format_exc())
    return float("nan")
