"""Utilities: metrics, checkpointing, seeding."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    mean_absolute_error,
)


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Compute all evaluation metrics. Returns dict of floats."""
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, preds)),
        "qwk": float(cohen_kappa_score(labels, preds, weights="quadratic")),
        "mae": float(mean_absolute_error(labels, preds)),
    }


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    epoch: int,
    metrics: dict,
    feat_dim: int,
    model_name: str,
    hidden: int | None = None,
    top_k: int | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    **extra,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "feat_dim": feat_dim,
        "model_name": model_name,
        "metrics": metrics,
        "hidden": hidden,
        "top_k": top_k,
        **extra,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
) -> dict:
    """Load checkpoint. Restores model; optionally optimizer and scheduler."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and "scheduler_state_dict" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
