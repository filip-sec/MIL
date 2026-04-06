#!/usr/bin/env python3
"""Extract attention weights from a trained MIL model for all slides.

Saves attention_dir/slide_id.pt (tensor [n_patches]) for use with --attention-dir
in run_training.py for attention-biased sampling in a second training phase.

Usage:
  # TRIDENT (MetaCentrum) canonical features dir:
  # /storage/brno2/home/filipsec/MIL/data/trident_out/panda_uni_v2_grandqc_20x_256_ov0/20x_256px_0px_overlap/features_uni_v2
  python scripts/extract_attention.py \
    --feat-dir /storage/brno2/home/filipsec/MIL/data/trident_out/panda_uni_v2_grandqc_20x_256_ov0/20x_256px_0px_overlap/features_uni_v2 \
    --checkpoint checkpoints/fold0_best.pt \
    --out-dir attention_cache

  # Then (optional) use the extracted attention for attention-biased sampling:
  python scripts/run_training.py \
    --feat-dir /storage/brno2/home/filipsec/MIL/data/trident_out/panda_uni_v2_grandqc_20x_256_ov0/20x_256px_0px_overlap/features_uni_v2 \
    --max-patches 2000 \
    --attention-dir attention_cache \
    --model ordinal
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import pandas as pd
import torch

from mil.model import build_model
from mil.attention_map import get_attention_for_slide


def main():
    p = argparse.ArgumentParser(description="Extract MIL attention for biased sampling")
    p.add_argument("--feat-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True, help="Model checkpoint (e.g. checkpoints/fold0_best.pt)")
    p.add_argument("--out-dir", type=Path, required=True, help="Output dir for slide_id.pt files")
    p.add_argument("--train-csv", type=Path, default=Path("data/raw/train.csv"))
    p.add_argument("--splits-csv", type=Path, default=None, help="If given, extract only for slides in splits")
    p.add_argument("--key", type=str, default="features")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    for attr in ("feat_dir", "train_csv", "splits_csv", "checkpoint", "out_dir"):
        val = getattr(args, attr)
        if val is not None and not val.is_absolute():
            setattr(args, attr, root / val)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict with model_state_dict")
    state = ckpt["model_state_dict"]
    model_name = ckpt.get("model_name", "ordinal")
    hidden = ckpt.get("hidden", 256 if model_name != "attention" else 128)
    top_k = ckpt.get("top_k", 8)
    feat_dim = ckpt.get("feat_dim")
    if feat_dim is None:
        sample = list((args.feat_dir).glob("*.h5"))[:1]
        if sample:
            with h5py.File(sample[0], "r") as f:
                feat_dim = f[args.key].shape[1]
        else:
            raise ValueError("Cannot infer feat_dim; provide checkpoint with config")

    # Only OrdinalMIL accepts top_k; AttentionMIL/GatedAttentionMIL ignore it.
    model_kwargs = dict(feat_dim=feat_dim, num_classes=6, hidden=hidden)
    if model_name == "ordinal":
        model_kwargs["top_k"] = top_k
    model = build_model(model_name, **model_kwargs)
    model.load_state_dict(state, strict=True)
    model = model.to(args.device)
    model.eval()

    if args.splits_csv:
        splits = pd.read_csv(args.splits_csv)
        slide_ids = splits["image_id"].unique().tolist()
    else:
        slide_ids = [p.stem for p in args.feat_dir.glob("*.h5")]

    n = 0
    for i, sid in enumerate(slide_ids):
        h5_path = args.feat_dir / f"{sid}.h5"
        if not h5_path.exists():
            continue
        with h5py.File(h5_path, "r") as f:
            feats = torch.from_numpy(f[args.key][:]).float()
        if feats.size(0) == 0:
            continue
        att = get_attention_for_slide(model, feats, device=args.device)
        out_path = args.out_dir / f"{sid}.pt"
        torch.save(att, out_path)
        n += 1
        if (i + 1) % 500 == 0:
            print(f"  Extracted {i + 1}/{len(slide_ids)}", flush=True)

    print(f"Saved {n} attention files to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
