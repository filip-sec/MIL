#!/usr/bin/env python3
"""
Thin CLI for MIL training on PANDA .h5 features (attention / gated / ordinal, 5-fold CV).

Usage examples:

  # Phase 1: baseline (single-branch attention + CE)
  python scripts/run_training.py --feat-dir data/trident_out/.../features_uni_v2 --model attention

  # Phase 2: gated attention
  python scripts/run_training.py --feat-dir ... --model gated

  # Phase 3: ordinal (CORAL + aux CE)
  python scripts/run_training.py --feat-dir ... --model ordinal

  # Phase 4: stochastic patch dropout + max patches cap (smart sampling)
  python scripts/run_training.py --feat-dir ... --model ordinal --patch-dropout 0.2 --max-patches 2000

  # Phase 5: attention-biased sampling (extract attention from trained model first)
  python scripts/extract_attention.py --feat-dir ... --checkpoint checkpoints/fold0_best.pt --out-dir attention_cache
  python scripts/run_training.py --feat-dir ... --max-patches 2000 --attention-dir attention_cache --model ordinal

  # Default: on-the-fly loading (lower RAM). Use --preload only for small subsets.
  python scripts/run_training.py --feat-dir ...  # H5BagDataset, no preload
  python scripts/run_training.py --feat-dir ... --preload  # CachedBagDataset, high RAM

  # Resume interrupted fold (e.g. after crash)
  python scripts/run_training.py --feat-dir ... --resume checkpoints/fold0_best.pt --folds 0

  # Single fold only
  python scripts/run_training.py --feat-dir ... --model ordinal --folds 0

  # All phases at once (5-fold, ordinal, patch dropout, early stopping)
  python scripts/run_training.py --feat-dir ... --model ordinal --patch-dropout 0.2 --patience 10 --epochs 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from mil.train import run_cv


def _load_attention_weights(attention_dir: Path) -> dict[str, torch.Tensor]:
    """Load attention weights from attention_dir/slide_id.pt. Returns dict slide_id -> [n] tensor."""
    attention_dir = Path(attention_dir)
    out = {}
    for p in attention_dir.glob("*.pt"):
        slide_id = p.stem
        out[slide_id] = torch.load(p, map_location="cpu", weights_only=True)
    return out


def main():
    p = argparse.ArgumentParser(description="MIL training (PANDA ISUP)")
    p.add_argument("--feat-dir", type=Path, required=True)
    p.add_argument("--train-csv", type=Path, default=Path("data/raw/train.csv"))
    p.add_argument("--splits-csv", type=Path, default=Path("data/splits/panda_5fold_stratified.csv"))
    p.add_argument("--model", type=str, default="attention", choices=["attention", "gated", "ordinal"])
    p.add_argument(
        "--hidden", type=int, default=None, help="Hidden dim (default: 128 for attention, 256 for gated/ordinal)"
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate (SGD: 0.01 typical)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (0=main process only, 4 typical)")
    p.add_argument("--warmup-epochs", type=int, default=5, help="Linear warmup epochs before cosine decay")
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=0, help="Early stopping patience (0=disabled)")
    p.add_argument(
        "--patch-dropout", type=float, default=0.0, help="Fraction of patches to drop per slide (0=disabled)"
    )
    p.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Cap for both train and val (deprecated; prefer --max-patches-train/val)",
    )
    p.add_argument(
        "--max-patches-train", type=int, default=512, help="Cap train bags (reduces RAM, collation, forward time)"
    )
    p.add_argument(
        "--max-patches-val", type=int, default=512, help="Cap val bags (512=conservative; try 768 if val is fast)"
    )
    p.add_argument(
        "--attention-dir",
        type=Path,
        default=None,
        help="Dir with attention .pt files (slide_id.pt) for biased sampling; use after extracting attention from a trained model",
    )
    p.add_argument("--ce-weight", type=float, default=0.3, help="Auxiliary CE weight for ordinal model")
    p.add_argument("--top-k", type=int, default=8, help="Top-k patches for focal branch (ordinal only)")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    p.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Resume from checkpoint (e.g. checkpoints/fold0_best.pt); use with --folds N to resume that fold",
    )
    p.add_argument(
        "--coords-dir",
        type=Path,
        default=None,
        help="Dir with coords .h5 (slide_id.h5 or slide_id_patches.h5) if not in feat .h5",
    )
    p.add_argument(
        "--preload",
        action="store_true",
        help="Preload all features to RAM (default: off, use for small subsets; off = on-the-fly H5BagDataset, lower RAM)",
    )
    p.add_argument("--folds", type=int, nargs="+", default=None, help="Which folds to run (default: all)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    mil_root = Path(__file__).resolve().parent.parent
    for attr in ("feat_dir", "train_csv", "splits_csv", "checkpoint_dir", "coords_dir", "attention_dir", "resume"):
        val = getattr(args, attr)
        if val is not None and not val.is_absolute():
            setattr(args, attr, mil_root / val)

    cfg = {
        "model_name": args.model,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "patch_dropout": args.patch_dropout,
        "ce_weight": args.ce_weight,
        "top_k": args.top_k,
        "checkpoint_dir": args.checkpoint_dir,
        "seed": args.seed,
        "warmup_epochs": args.warmup_epochs,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "preload": args.preload,
        "num_workers": args.num_workers,
    }
    if args.max_patches is not None:
        cfg["max_patches_train"] = cfg["max_patches_val"] = args.max_patches
    else:
        cfg["max_patches_train"] = args.max_patches_train
        cfg["max_patches_val"] = args.max_patches_val
    if args.attention_dir is not None:
        cfg["attention_weights"] = _load_attention_weights(args.attention_dir)
    if args.hidden is not None:
        cfg["hidden"] = args.hidden
    if args.coords_dir is not None:
        cfg["coords_dir"] = args.coords_dir
    if args.resume is not None:
        cfg["resume_from"] = args.resume

    df = run_cv(
        feat_dir=args.feat_dir,
        train_csv=args.train_csv,
        splits_csv=args.splits_csv,
        cfg=cfg,
        folds=args.folds,
    )
    print("\nPer-fold results:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
