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

  # CLAM-SB (gated attention + instance supervision; Lu et al. style)
  python scripts/run_training.py --feat-dir ... --model clam --clam-inst-weight 0.7

  # Phase 4: stochastic patch dropout + max patches cap (smart sampling)
  python scripts/run_training.py --feat-dir ... --model ordinal --patch-dropout 0.2 --max-patches 2000

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

from mil.config import merge_runtime_config
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
    p.add_argument("--config", type=Path, default=None, help="Optional YAML config (defaults to configs/pipelines/mil_training.yaml)")
    p.add_argument("--feat-dir", type=Path, default=None)
    p.add_argument("--train-csv", type=Path, default=None)
    p.add_argument("--splits-csv", type=Path, default=None)
    p.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["attention", "gated", "ordinal", "clam"],
    )
    p.add_argument(
        "--hidden", type=int, default=None, help="Hidden dim (default: 128 for attention, 256 for gated/ordinal)"
    )
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None, help="Learning rate (SGD: 0.01 typical)")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None, help="DataLoader workers (0=main process only, 4 typical)")
    p.add_argument("--warmup-epochs", type=int, default=None, help="Linear warmup epochs before cosine decay")
    p.add_argument("--momentum", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--patience", type=int, default=None, help="Early stopping patience (0=disabled)")
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
        "--max-patches-train", type=int, default=None, help="Cap train bags (reduces RAM, collation, forward time)"
    )
    p.add_argument(
        "--max-patches-val", type=int, default=None, help="Cap val bags (512=conservative; try 768 if val is fast)"
    )
    p.add_argument(
        "--attention-dir",
        type=Path,
        default=None,
        help="Dir with attention .pt files (slide_id.pt) for biased sampling; use after extracting attention from a trained model",
    )
    p.add_argument("--ce-weight", type=float, default=None, help="Auxiliary CE weight for ordinal model")
    p.add_argument("--top-k", type=int, default=None, help="Top-k patches for ordinal focal branch; CLAM instance sampling if --model clam")
    p.add_argument(
        "--clam-inst-weight",
        type=float,
        default=None,
        help="Weight for CLAM instance supervision vs bag CE (--model clam only)",
    )
    p.add_argument("--checkpoint-dir", type=Path, default=None)
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
        "--encoder-name",
        type=str,
        default=None,
        help="Optional encoder registry key for checkpoint metadata (otherwise inferred from feat-dir basename when possible)",
    )
    p.add_argument(
        "--preload",
        action="store_true",
        default=None,
        help="Preload all features to RAM (default: off, use for small subsets; off = on-the-fly H5BagDataset, lower RAM)",
    )
    p.add_argument("--folds", type=int, nargs="+", default=None, help="Which folds to run (default: all)")
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    overrides = {key: value for key, value in vars(args).items() if key != "config" and value is not None}
    runtime = merge_runtime_config("mil_training", config_path=args.config, overrides=overrides)

    feat_dir = runtime.get("feat_dir")
    if feat_dir is None:
        raise SystemExit("--feat-dir is required (or set feat_dir in the config file)")
    train_csv = runtime.get("train_csv")
    splits_csv = runtime.get("splits_csv")
    cfg = {
        "model_name": runtime.get("model"),
        "epochs": runtime.get("epochs"),
        "lr": runtime.get("lr"),
        "batch_size": runtime.get("batch_size"),
        "patience": runtime.get("patience"),
        "patch_dropout": runtime.get("patch_dropout"),
        "ce_weight": runtime.get("ce_weight"),
        "top_k": runtime.get("top_k"),
        "checkpoint_dir": runtime.get("checkpoint_dir"),
        "seed": runtime.get("seed"),
        "warmup_epochs": runtime.get("warmup_epochs"),
        "momentum": runtime.get("momentum"),
        "weight_decay": runtime.get("weight_decay"),
        "preload": runtime.get("preload"),
        "num_workers": runtime.get("num_workers"),
        "clam_inst_weight": runtime.get("clam_inst_weight"),
        "config_name": runtime.config_name,
        "config_path": str(runtime.config_path),
    }
    if runtime.get("max_patches") is not None:
        cfg["max_patches_train"] = cfg["max_patches_val"] = runtime.get("max_patches")
    else:
        cfg["max_patches_train"] = runtime.get("max_patches_train")
        cfg["max_patches_val"] = runtime.get("max_patches_val")
    if runtime.get("attention_dir") is not None:
        cfg["attention_weights"] = _load_attention_weights(runtime.get("attention_dir"))
    if runtime.get("hidden") is not None:
        cfg["hidden"] = runtime.get("hidden")
    if runtime.get("coords_dir") is not None:
        cfg["coords_dir"] = runtime.get("coords_dir")
    if runtime.get("resume") is not None:
        cfg["resume_from"] = runtime.get("resume")
    if runtime.get("encoder_name") is not None:
        cfg["encoder_name"] = runtime.get("encoder_name")

    df = run_cv(feat_dir=feat_dir, train_csv=train_csv, splits_csv=splits_csv, cfg=cfg, folds=runtime.get("folds"))
    print("\nPer-fold results:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
