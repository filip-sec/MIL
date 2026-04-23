#!/usr/bin/env python3
"""Train a lightweight classifier on cached Prov-GigaPath slide embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mil.config import merge_runtime_config
from mil.gigapath_slide import (
    GigaPathSlideClassifier,
    SlideEmbeddingDataset,
    collate_slide_embeddings,
    evaluate_slide_model,
    load_slide_embedding_cache,
    train_slide_epoch,
)
from mil.utils import save_checkpoint, seed_everything


def main() -> None:
    ap = argparse.ArgumentParser(description="Train a classifier over Prov-GigaPath slide embeddings")
    ap.add_argument("--config", type=Path, default=None, help="Optional YAML config (defaults to configs/pipelines/gigapath_slide_train.yaml)")
    ap.add_argument("--emb-dir", type=Path, default=None, help="Directory with slide_id.pt embeddings")
    ap.add_argument("--train-csv", type=Path, default=None)
    ap.add_argument("--splits-csv", type=Path, default=None)
    ap.add_argument("--checkpoint-dir", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--folds", type=int, nargs="+", default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    overrides = {key: value for key, value in vars(args).items() if key != "config" and value is not None}
    runtime = merge_runtime_config("gigapath_slide_train", config_path=args.config, overrides=overrides)

    emb_dir = runtime.get("emb_dir")
    if emb_dir is None:
        raise SystemExit("--emb-dir is required (or set emb_dir in the config file)")

    seed_everything(runtime.get("seed"))
    device_name = runtime.get("device")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    print(f"Device: {device}", flush=True)

    cache = load_slide_embedding_cache(emb_dir)
    sample_dim = next(iter(cache.values())).numel()
    print(f"Loaded {len(cache)} slide embeddings, dim={sample_dim}", flush=True)

    labels = pd.read_csv(runtime.get("train_csv")).set_index("image_id")
    splits = pd.read_csv(runtime.get("splits_csv"))
    folds = runtime.get("folds") or sorted(splits["fold"].unique())

    checkpoint_dir = runtime.get("checkpoint_dir")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    for fold in folds:
        print(f"\n{'=' * 60}\nGIGAPATH SLIDE FOLD {fold}\n{'=' * 60}", flush=True)
        train_ids = splits.loc[splits["fold"] != fold, "image_id"].astype(str).tolist()
        val_ids = splits.loc[splits["fold"] == fold, "image_id"].astype(str).tolist()

        train_ds = SlideEmbeddingDataset(cache, train_ids, labels)
        val_ds = SlideEmbeddingDataset(cache, val_ids, labels)
        if not len(train_ds) or not len(val_ds):
            print(f"Skipping fold {fold}: no overlapping slide embeddings", flush=True)
            continue

        train_loader = DataLoader(
            train_ds,
            batch_size=runtime.get("batch_size"),
            shuffle=True,
            num_workers=runtime.get("num_workers"),
            collate_fn=collate_slide_embeddings,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=runtime.get("batch_size"),
            shuffle=False,
            num_workers=runtime.get("num_workers"),
            collate_fn=collate_slide_embeddings,
        )

        model = GigaPathSlideClassifier(sample_dim, dropout=runtime.get("dropout")).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=runtime.get("lr"),
            weight_decay=runtime.get("weight_decay"),
        )
        criterion = nn.CrossEntropyLoss()

        best_qwk = -1.0
        best_metrics: dict | None = None
        wait = 0
        for epoch in range(1, runtime.get("epochs") + 1):
            train_loss = train_slide_epoch(model, train_loader, criterion, optimizer, device)
            metrics = evaluate_slide_model(model, val_loader, criterion, device)
            print(
                f"  Fold {fold} | Epoch {epoch:3d}/{runtime.get('epochs')} | "
                f"train_loss={train_loss:.4f} | val_loss={metrics['loss']:.4f} | "
                f"qwk={metrics['qwk']:.4f} | bal_acc={metrics['balanced_accuracy']:.4f} | "
                f"mae={metrics['mae']:.3f}",
                flush=True,
            )
            if metrics["qwk"] > best_qwk:
                best_qwk = float(metrics["qwk"])
                best_metrics = {**metrics, "epoch": epoch}
                save_checkpoint(
                    checkpoint_dir / f"fold{fold}_best.pt",
                    model,
                    epoch,
                    metrics,
                    feat_dim=sample_dim,
                    model_name="gigapath_slide_linear",
                    encoder_name="gigapath_slide",
                    config_name=runtime.config_name,
                    config_path=str(runtime.config_path),
                    optimizer=optimizer,
                    hidden=None,
                    top_k=None,
                    embedding_source_dir=str(emb_dir),
                )
                wait = 0
            else:
                wait += 1
                if runtime.get("patience") > 0 and wait >= runtime.get("patience"):
                    print(f"  Early stop at epoch {epoch}", flush=True)
                    break

        if best_metrics is not None:
            rows.append({"fold": fold, **best_metrics})

    if rows:
        df = pd.DataFrame(rows)
        print("\nPer-fold results:")
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
