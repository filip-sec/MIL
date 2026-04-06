#!/usr/bin/env python3
"""
Quick baseline: single-branch attention MIL + CE on PANDA .h5 features (one fold or 80/20).

For full 5-fold CV, gated/ordinal models, patch caps, and resume, use:

  python scripts/run_training.py --feat-dir ...

This script reuses ``mil`` datasets and models (no duplicated implementations).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

MIL_ROOT = Path(__file__).resolve().parent.parent
if str(MIL_ROOT) not in sys.path:
    sys.path.insert(0, str(MIL_ROOT))

import pandas as pd

from mil.data import H5BagDataset, collate_bags
from mil.model import AttentionMIL


def _path(s: str) -> Path:
    p = Path(s)
    if not p.is_absolute():
        p = MIL_ROOT / p
    return p


def main():
    p = argparse.ArgumentParser(description="Quick MIL baseline (attention + CE, one setup)")
    p.add_argument("--feat-dir", type=_path, required=True, help="Dir with slide_id.h5 patch features")
    p.add_argument("--train-csv", type=_path, default=MIL_ROOT / "data" / "raw" / "train.csv", help="train.csv")
    p.add_argument("--splits-csv", type=_path, default=MIL_ROOT / "data" / "splits" / "panda_5fold_stratified.csv")
    p.add_argument("--fold", type=int, default=0, help="Holdout fold 0–4 when splits-csv has fold column")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--checkpoint-dir", type=_path, default=MIL_ROOT / "checkpoints")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if not args.train_csv.exists():
        raise SystemExit(f"Missing {args.train_csv}")
    train_labels = pd.read_csv(args.train_csv).set_index("image_id")

    feat_dir = args.feat_dir
    if not feat_dir.exists():
        raise SystemExit(f"feat_dir not found: {feat_dir}")
    available_h5 = {p.stem for p in feat_dir.glob("*.h5")}
    slide_ids = [sid for sid in train_labels.index if sid in available_h5]
    labels_subset = train_labels.loc[slide_ids]
    print("Slides with features and labels:", len(slide_ids))

    if args.splits_csv.exists():
        splits = pd.read_csv(args.splits_csv)
        if "fold" in splits.columns and "image_id" in splits.columns:
            train_ids = splits[splits["fold"] != args.fold]["image_id"].tolist()
            val_ids = splits[splits["fold"] == args.fold]["image_id"].tolist()
            train_ids = [i for i in train_ids if i in available_h5]
            val_ids = [i for i in val_ids if i in available_h5]
            print(f"Fold {args.fold}: train {len(train_ids)}, val {len(val_ids)}")
        else:
            train_ids, val_ids = train_test_split(
                slide_ids, test_size=0.2, random_state=args.seed, stratify=labels_subset["isup_grade"]
            )
            print("Splits CSV missing fold/image_id; using 80/20 stratified")
    else:
        train_ids, val_ids = train_test_split(
            slide_ids, test_size=0.2, random_state=args.seed, stratify=labels_subset["isup_grade"]
        )
        print("No splits CSV; using 80/20 stratified")

    with h5py.File(feat_dir / f"{train_ids[0]}.h5", "r") as f:
        feat_dim = f["features"].shape[1]
    print("feat_dim:", feat_dim)

    train_ds = H5BagDataset(feat_dir, train_ids, train_labels)
    val_ds = H5BagDataset(feat_dir, val_ids, train_labels)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, collate_fn=collate_bags
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_bags)

    model = AttentionMIL(feat_dim=feat_dim, num_classes=6, hidden=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_bal_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for feats, mask, labels, _ids, _coords in train_loader:
            feats, mask, labels = feats.to(device), mask.to(device), labels.to(device)
            opt.zero_grad()
            logits = model(feats, mask)
            loss = criterion(logits, labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        model.eval()
        all_pred, all_label = [], []
        with torch.no_grad():
            for feats, mask, labels, _ids, _coords in val_loader:
                feats, mask, labels = feats.to(device), mask.to(device), labels.to(device)
                logits = model(feats, mask)
                pred = logits.argmax(dim=1)
                all_pred.append(pred.cpu())
                all_label.append(labels.cpu())
        all_pred = torch.cat(all_pred)
        all_label = torch.cat(all_label)
        acc = accuracy_score(all_label.numpy(), all_pred.numpy())
        bal_acc = balanced_accuracy_score(all_label.numpy(), all_pred.numpy())

        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  val_acc={acc:.4f}  val_bal_acc={bal_acc:.4f}")

        if bal_acc > best_bal_acc:
            best_bal_acc = bal_acc
            ckpt = args.checkpoint_dir / "mil_baseline_best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_bal_acc": bal_acc,
                    "feat_dim": feat_dim,
                    "model_name": "attention",
                },
                ckpt,
            )
            print(f"  -> saved {ckpt}")

    print("Done. Best val balanced accuracy:", best_bal_acc)


if __name__ == "__main__":
    main()
