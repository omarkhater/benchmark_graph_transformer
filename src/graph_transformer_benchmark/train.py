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

from graph_transformer_benchmark.data import build_dataloaders
from graph_transformer_benchmark.graph_models import build_model
from graph_transformer_benchmark.training.graph_transformer_trainer import (
    GraphTransformerTrainer,
)
from graph_transformer_benchmark.utils import (
    build_run_name,
    configure_determinism,
    create_model,
    get_device,
    infer_num_classes,
    init_mlflow,
    log_config,
    log_dataset_stats,
    set_seed,
    worker_init_fn,
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
            num_classes = infer_num_classes(train_loader)
            sample_batch = next(iter(train_loader))
            model = create_model(
                model_fn=build_model,
                model_cfg=cfg.get("model", {}),
                sample_batch=sample_batch,
                num_classes=num_classes,
                device=device)
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
