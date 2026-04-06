#!/usr/bin/env python3
"""
Plot MIL attention map on a WSI thumbnail.

Usage:
  # After training, with checkpoint:
  python scripts/plot_attention_map.py \\
    --slide-id 005e66f06bce9c2e49142536caf2f6ee \\
    --checkpoint checkpoints/best.pt \\
    --wsi-dir data/raw/train_images \\
    --feat-dir data/trident_out/.../features_uni_v2 \\
    --coords-dir data/trident_out/.../20x_256px_0px_overlap  # optional, for spatial overlay

  # With pre-extracted raw per-tile attention (slide_id.pt), no checkpoint needed:
  python scripts/plot_attention_map.py \\
    --slide-id 005e66f06bce9c2e49142536caf2f6ee \\
    --attention-dir attention_cache_fold0 \\
    --wsi-dir data/raw/train_images \\
    --feat-dir data/trident_out/.../features_uni_v2 \\
    --coords-dir data/trident_out/.../20x_256px_0px_overlap

  # Without coords: shows 1D bar plot of attention per patch
  # With coords: overlays heatmap on WSI thumbnail
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root (parent of scripts/) so `import mil` works when run as `python scripts/plot_attention_map.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import torch

from mil.attention_map import get_attention_for_slide, plot_attention_map
from mil.model import build_model
from mil.utils import load_checkpoint


def main():
    ap = argparse.ArgumentParser(description="Plot MIL attention map on WSI")
    ap.add_argument("--slide-id", required=True, help="Slide ID (e.g. 005e66f06bce9c2e49142536caf2f6ee)")
    ap.add_argument("--checkpoint", type=Path, default=None, help="Model checkpoint .pt (optional if using --attention-dir)")
    ap.add_argument(
        "--attention-dir",
        type=Path,
        default=None,
        help="Dir with pre-extracted attention .pt files (slide_id.pt). If set, skips model inference.",
    )
    ap.add_argument("--feat-dir", type=Path, required=True, help="Features .h5 directory")
    ap.add_argument("--wsi-dir", type=Path, default=None, help="WSI .tiff directory for thumbnail")
    ap.add_argument("--coords-dir", type=Path, default=None, help="Patch coords (if not in feat .h5)")
    ap.add_argument("--output", type=Path, default=None, help="Save figure path")
    ap.add_argument("--thumb-size", type=int, default=512)
    args = ap.parse_args()

    if args.attention_dir is None and args.checkpoint is None:
        raise SystemExit("Provide either --checkpoint (to compute attention) or --attention-dir (to load pre-extracted).")
    if args.attention_dir is not None and args.checkpoint is not None:
        raise SystemExit("Provide only one of --checkpoint or --attention-dir (not both).")

    h5_path = args.feat_dir / f"{args.slide_id}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Features not found: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        feats = torch.from_numpy(f["features"][:]).float().unsqueeze(0)
        n = feats.size(1)
        mask = torch.ones(1, n, dtype=torch.bool)

        coords = None
        if "coords" in f:
            coords = f["coords"][:]
        elif args.coords_dir:
            for name in (f"{args.slide_id}.h5", f"{args.slide_id}_patches.h5"):
                path = args.coords_dir / name
                if path.exists():
                    with h5py.File(path, "r") as cf:
                        for k in ("coords", "coordinates", "patches"):
                            if k in cf and cf[k].shape[0] == n:
                                coords = cf[k][:]
                                break
                    break

    if args.attention_dir is not None:
        att_path = args.attention_dir / f"{args.slide_id}.pt"
        if not att_path.exists():
            raise FileNotFoundError(f"Attention not found: {att_path}")
        att = torch.load(att_path, map_location="cpu", weights_only=True)
        if att.dim() != 1 or att.numel() != n:
            raise ValueError(f"Attention shape mismatch: expected [N]={n}, got {tuple(att.shape)}")
    else:
        ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        feat_dim = ckpt["feat_dim"]
        model_name = ckpt.get("model_name", "attention")
        hidden = ckpt.get("hidden", 256 if model_name != "attention" else 128)
        top_k = ckpt.get("top_k", 8)
        model_kwargs = dict(feat_dim=feat_dim, num_classes=6, hidden=hidden)
        if model_name == "ordinal":
            model_kwargs["top_k"] = top_k
        model = build_model(model_name, **model_kwargs)
        load_checkpoint(args.checkpoint, model)
        model.eval()
        att = get_attention_for_slide(model, feats, mask)

    wsi_path = None
    if args.wsi_dir:
        wsi_path = args.wsi_dir / f"{args.slide_id}.tiff"
        if not wsi_path.exists():
            wsi_path = None

    ax = plot_attention_map(
        att,
        coords=coords,
        wsi_path=wsi_path,
        thumbnail_size=args.thumb_size,
        alpha=0.6,
    )
    ax.set_title(f"{args.slide_id}\nAttention map (pred: run inference for label)")

    if args.output:
        ax.figure.savefig(args.output, dpi=150, bbox_inches="tight")
        print(f"Saved: {args.output}")
    else:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
