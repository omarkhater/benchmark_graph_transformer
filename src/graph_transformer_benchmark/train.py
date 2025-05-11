"""GraphTransformer benchmark – single‑run training entry‑point.

This module is meant to be imported (or run) by CLI wrappers.  It builds the
PyG dataloaders, instantiates the requested model, and hands control over to
:class:`graph_transformer_benchmark.training.GraphTransformerTrainer`, which
contains the generic training / validation loop.
"""
from __future__ import annotations

import mlflow
import torch
from omegaconf import DictConfig

from graph_transformer_benchmark.data import build_dataloaders
from graph_transformer_benchmark.graph_models import build_model
from graph_transformer_benchmark.training.graph_transformer_trainer import (
    GraphTransformerTrainer,
)
from graph_transformer_benchmark.utils import (
    build_run_name,
    configure_determinism,
    get_device,
    infer_num_classes,
    infer_num_node_features,
    init_mlflow,
    log_config,
    log_dataset_stats,
    set_seed,
    worker_init_fn,
)


def run_training(cfg: DictConfig) -> float:
    """Launch training for a single model/dataset pair.

    The function is designed to be called programmatically from a CLI front‑end
    that parses the Hydra configuration. All heavy lifting—early stopping,
    validation scheduling, MLflow logging—is performed by
    :class:`~graph_transformer_benchmark.training.GraphTransformerTrainer`.

    Parameters
    ----------
    cfg:
        Hydra configuration node produced by
        ``graph_transformer_benchmark/conf``.

    Returns
    -------
    float
        Best validation loss observed during training.
    """
    set_seed(cfg.training.seed)
    global _GLOBAL_SEED
    _GLOBAL_SEED = cfg.training.seed

    configure_determinism(cfg.training.seed, torch.cuda.is_available())

    init_mlflow(cfg)
    run_name = cfg.model.training.mlflow.run_name or build_run_name(cfg)
    with mlflow.start_run(run_name=run_name):
        mlflow.set_tag(
            "mlflow.note.content", cfg.model.training.mlflow.description)
        log_config(cfg)

        generator = torch.Generator().manual_seed(cfg.training.seed)
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

        device = get_device(cfg.training.device)
        num_features = infer_num_node_features(train_loader)
        num_classes = infer_num_classes(train_loader)
        model = build_model(cfg.model, num_features, num_classes).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

        trainer = GraphTransformerTrainer(
            cfg=cfg,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=cfg.training.epochs,
            val_frequency=cfg.training.val_frequency,
            patience=cfg.training.patience,
        )

        _, metrics = trainer.train()
        return float(metrics["best_loss"])
