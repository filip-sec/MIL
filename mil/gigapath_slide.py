"""Helpers for the experimental Prov-GigaPath slide-encoder pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .utils import compute_metrics


def load_slide_embedding_cache(emb_dir: Path) -> dict[str, torch.Tensor]:
    """Load ``slide_id.pt`` embeddings from a cache dir."""
    cache: dict[str, torch.Tensor] = {}
    for path in sorted(Path(emb_dir).glob("*.pt")):
        tensor = torch.load(path, map_location="cpu", weights_only=True)
        if tensor.ndim != 1:
            raise ValueError(f"{path} must contain a 1D slide embedding, got shape {tuple(tensor.shape)}")
        cache[path.stem] = tensor.float()
    if not cache:
        raise ValueError(f"No .pt slide embeddings found under {emb_dir}")
    return cache


class SlideEmbeddingDataset(Dataset):
    """Small dataset of precomputed slide embeddings."""

    def __init__(self, cache: dict[str, torch.Tensor], slide_ids: list[str], labels: pd.DataFrame):
        self.cache = cache
        self.slide_ids = [sid for sid in slide_ids if sid in cache and sid in labels.index]
        self.labels = labels

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, str]:
        sid = self.slide_ids[idx]
        label = int(self.labels.loc[sid, "isup_grade"])
        return self.cache[sid], label, sid


def collate_slide_embeddings(batch: list[tuple[torch.Tensor, int, str]]) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """Batch fixed-size slide embeddings."""
    feats = torch.stack([row[0] for row in batch], dim=0)
    labels = torch.tensor([row[1] for row in batch], dtype=torch.long)
    ids = [row[2] for row in batch]
    return feats, labels, ids


class GigaPathSlideClassifier(nn.Module):
    """Simple slide-level head over precomputed Prov-GigaPath slide embeddings."""

    def __init__(self, input_dim: int, num_classes: int = 6, dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Dropout(dropout),
            nn.Linear(input_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class SlideTrainResult:
    """Fold summary for the slide-encoder experiment."""

    fold: int
    epoch: int
    qwk: float
    balanced_accuracy: float
    accuracy: float
    mae: float
    loss: float


def train_slide_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Return average training loss for one epoch."""
    model.train()
    total = 0.0
    for feats, labels, _ids in loader:
        feats = feats.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(feats), labels)
        loss.backward()
        optimizer.step()
        total += float(loss.item())
    return total / max(1, len(loader))


@torch.no_grad()
def evaluate_slide_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None,
    device: torch.device,
) -> dict:
    """Evaluate one slide-level classifier checkpoint."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0
    for feats, labels, _ids in loader:
        feats = feats.to(device)
        labels = labels.to(device)
        logits = model(feats)
        if criterion is not None:
            total_loss += float(criterion(logits, labels).item())
            n_batches += 1
        all_preds.append(logits.argmax(dim=1).cpu())
        all_labels.append(labels.cpu())
    preds_np = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()
    out = compute_metrics(preds_np, labels_np)
    if criterion is not None and n_batches > 0:
        out["loss"] = total_loss / n_batches
    return out
