#!/usr/bin/env python3
"""
Plot MIL attention map on a WSI thumbnail.

CLAM ``create_heatmaps.py``-style conveniences (without re-running the tile encoder):
optional bag logits / softmax printout, blockmap HDF5 (``attention_scores`` + ``coords``),
and batch processing from a CSV.

Usage:
  # After training, with checkpoint (attention / gated / ordinal / CLAM):
  python scripts/plot_attention_map.py \\
    --slide-id 005e66f06bce9c2e49142536caf2f6ee \\
    --checkpoint checkpoints/fold0_best.pt \\
    --wsi-dir data/raw/train_images \\
    --feat-dir data/trident_out/.../features_uni_v2

  # Save CLAM-style blockmap + figure; tune overlay (cf. heatmap_arguments in YAML workflows):
  python scripts/plot_attention_map.py \\
    --slide-id ... --checkpoint ... --feat-dir ... --wsi-dir ... \\
    --output figures/att.png \\
    --blockmap-dir figures/blockmaps \\
    --cmap jet --alpha 0.55

  # Batch (CSV column ``slide_id``; requires ``--output-dir``):
  python scripts/plot_attention_map.py --slides-csv heatmaps/process_list.csv \\
    --checkpoint ... --feat-dir ... --wsi-dir ... --output-dir figures/heat_batch \\
    --blockmap-dir figures/blockmaps

  # With pre-extracted attention .pt files (no checkpoint):
  python scripts/plot_attention_map.py --slide-id ... --attention-dir ... --feat-dir ...

  # Full-slide pyramid raster (CLAM drawHeatmap-style: blur + percentile, TIFF/PNG via Pillow):
  python scripts/plot_attention_map.py --slide-id ... --checkpoint ... --feat-dir ... --wsi-dir ... \\
    --raster-out figures/heatmap_full.tif \\
    --max-side 6144 --gaussian-sigma 16 --percentile-high 99.5 --cmap jet

  # Without coords: 1D bar plot of attention; without WSI: no thumbnail background
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root (parent of scripts/) so `import mil` works when run as `python scripts/plot_attention_map.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import numpy as np
import pandas as pd
import torch

from mil.attention_map import (
    get_attention_and_logits_for_slide,
    infer_patch_size_level0,
    plot_attention_map,
    read_trident_coord_attrs_from_h5,
    save_attention_blockmap_h5,
    save_fullslide_attention_raster,
)
from mil.model import build_model
from mil.utils import load_checkpoint


def _parse_gaussian_sigma(s: str) -> float | None:
    sl = s.strip().lower()
    if sl in ("", "auto", "none"):
        return None
    if sl in ("off", "no"):
        return 0.0
    return float(s)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA requested but not available; using CPU.", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(device_str)


def _init_model_from_checkpoint(ckpt_path: Path) -> tuple[torch.nn.Module, str, int]:
    """Load weights once; ``feat_dim`` is the encoder dimension expected by the checkpoint."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    feat_dim = int(ckpt["feat_dim"])
    model_name = ckpt.get("model_name", "attention")
    hidden = ckpt.get("hidden", 256 if model_name != "attention" else 128)
    top_k = ckpt.get("top_k", 8)
    sd = ckpt.get("model_state_dict", {})
    if model_name == "clam":
        mid_dim = int(sd["fc.0.weight"].shape[0]) if "fc.0.weight" in sd else 512
        model_kwargs = {
            "feat_dim": feat_dim,
            "num_classes": 6,
            "mid_dim": mid_dim,
            "k_sample": int(top_k),
        }
        if "hidden" in ckpt:
            model_kwargs["attn_dim"] = int(ckpt["hidden"])
    else:
        model_kwargs = dict(feat_dim=feat_dim, num_classes=6, hidden=hidden)
        if model_name == "ordinal":
            model_kwargs["top_k"] = top_k
    model = build_model(model_name, **model_kwargs)
    load_checkpoint(ckpt_path, model)
    model.eval()
    return model, model_name, feat_dim


def _load_feats_coords(
    h5_path: Path,
    slide_id: str,
    coords_dir: Path | None,
) -> tuple[torch.Tensor, torch.Tensor, int, np.ndarray | None, int | None]:
    with h5py.File(h5_path, "r") as f:
        feats = torch.from_numpy(f["features"][:]).float().unsqueeze(0)
        n = feats.size(1)
        mask = torch.ones(1, n, dtype=torch.bool)

        coords = None
        if "coords" in f:
            coords = f["coords"][:]
        elif coords_dir:
            for name in (f"{slide_id}.h5", f"{slide_id}_patches.h5"):
                path = coords_dir / name
                if path.exists():
                    with h5py.File(path, "r") as cf:
                        for k in ("coords", "coordinates", "patches"):
                            if k in cf and cf[k].shape[0] == n:
                                coords = cf[k][:]
                                break
                    break

    patch_size_level0: int | None = None
    if coords is not None:
        attrs = read_trident_coord_attrs_from_h5(h5_path)
        patch_size_level0 = infer_patch_size_level0(attrs, fallback_encoder_patch=224)

    return feats, mask, n, coords, patch_size_level0


def _run_one_slide(
    *,
    slide_id: str,
    feat_dir: Path,
    coords_dir: Path | None,
    checkpoint: Path | None,
    attention_dir: Path | None,
    wsi_dir: Path | None,
    output: Path | None,
    raster_out: Path | None,
    thumb_size: int,
    device: torch.device,
    cmap: str,
    alpha: float,
    blockmap_dir: Path | None,
    print_probs: bool,
    gaussian_sigma: float | None,
    percentile_low: float,
    percentile_high: float,
    use_percentile_rank: bool,
    max_side: int,
    vis_level: int | None,
    mask_mode: str,
    patch_mask_dilate: int,
    model: torch.nn.Module | None = None,
    model_name_cached: str | None = None,
    ckpt_feat_dim: int | None = None,
):
    h5_path = feat_dir / f"{slide_id}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Features not found: {h5_path}")

    feats, mask, n, coords, patch_size_level0 = _load_feats_coords(h5_path, slide_id, coords_dir)

    if attention_dir is not None:
        att_path = attention_dir / f"{slide_id}.pt"
        if not att_path.exists():
            raise FileNotFoundError(f"Attention not found: {att_path}")
        att = torch.load(att_path, map_location="cpu", weights_only=True)
        if att.dim() != 1 or att.numel() != n:
            raise ValueError(f"Attention shape mismatch: expected [N]={n}, got {tuple(att.shape)}")
        pred_str = ""
    else:
        if model is None:
            assert checkpoint is not None
            model, model_name_cached, ckpt_fd = _init_model_from_checkpoint(checkpoint)
        else:
            ckpt_fd = ckpt_feat_dim
        assert ckpt_fd is not None and model_name_cached is not None
        if feats.size(-1) != ckpt_fd:
            raise ValueError(
                f"Feature dim mismatch for {slide_id}: .h5 has D={feats.size(-1)}, checkpoint expects {ckpt_fd}"
            )
        model_name = model_name_cached
        model = model.to(device)
        att, logits = get_attention_and_logits_for_slide(model, feats, mask, device=device)
        probs = torch.softmax(logits, dim=0)
        pred = int(probs.argmax().item())
        pred_str = f" pred={pred} probs={[float(f'{p:.4f}') for p in probs.tolist()]}"
        if print_probs:
            print(f"{slide_id} ({model_name}){pred_str}")

    wsi_path = None
    if wsi_dir:
        for ext in (".tiff", ".tif", ".svs"):
            cand = wsi_dir / f"{slide_id}{ext}"
            if cand.exists():
                wsi_path = cand
                break

    title = f"{slide_id}\nAttention map{pred_str if pred_str else ''}"

    # Matplotlib thumbnail: skip when only saving full-slide raster (faster, no duplicate render).
    want_thumb = output is not None or raster_out is None
    ax = None
    if want_thumb:
        ax = plot_attention_map(
            att,
            coords=coords,
            wsi_path=wsi_path,
            thumbnail_size=thumb_size,
            alpha=alpha,
            cmap=cmap,
            patch_size_level0=patch_size_level0,
            gaussian_sigma=gaussian_sigma,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
            use_percentile_rank=use_percentile_rank,
            mask_mode=mask_mode,
            patch_mask_dilate=patch_mask_dilate,
        )
        ax.set_title(title[:200] + ("…" if len(title) > 200 else ""))

    if raster_out is not None:
        if wsi_path is None or not Path(wsi_path).exists():
            raise ValueError(f"Full-slide --raster-out requires a WSI under --wsi-dir for {slide_id}")
        if coords is None or patch_size_level0 is None:
            raise ValueError(
                "Full-slide raster needs patch coords in the feature .h5 and TRIDENT-style L0 patch size attrs "
                "(or correct coords + patch_size_level0 in metadata); add coords to bags or use --coords-dir."
            )
        lv, rh, rw = save_fullslide_attention_raster(
            wsi_path,
            att,
            coords,
            patch_size_level0,
            raster_out,
            vis_level=vis_level,
            max_side=max_side,
            cmap=cmap,
            alpha=alpha,
            gaussian_sigma=gaussian_sigma,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
            use_percentile_rank=use_percentile_rank,
            mask_mode=mask_mode,
            patch_mask_dilate=patch_mask_dilate,
        )
        if print_probs:
            print(f"  fullslide raster: level={lv} size={rw}x{rh} -> {raster_out}")

    if blockmap_dir is not None and coords is not None:
        bm_path = blockmap_dir / f"{slide_id}_blockmap.h5"
        save_attention_blockmap_h5(bm_path, att.detach().cpu().numpy(), coords)
        if print_probs:
            print(f"  blockmap: {bm_path}")

    if output is not None and ax is not None:
        ax.figure.savefig(output, dpi=150, bbox_inches="tight")
        print(f"Saved: {output}")
        import matplotlib.pyplot as plt

        plt.close(ax.figure)
    return ax


def main():
    ap = argparse.ArgumentParser(description="Plot MIL attention map on WSI (features + checkpoint à la CLAM heatmaps)")
    ap.add_argument("--slide-id", default=None, help="Slide ID (required unless --slides-csv)")
    ap.add_argument("--slides-csv", type=Path, default=None, help="CSV with slide_id column; sets batch mode")
    ap.add_argument("--checkpoint", type=Path, default=None, help="Model checkpoint .pt (optional if --attention-dir)")
    ap.add_argument(
        "--attention-dir",
        type=Path,
        default=None,
        help="Dir with pre-extracted attention .pt files (slide_id.pt). If set, skips model inference.",
    )
    ap.add_argument("--feat-dir", type=Path, required=True, help="Features .h5 directory")
    ap.add_argument("--wsi-dir", type=Path, default=None, help="WSI directory (.tiff / .tif / .svs)")
    ap.add_argument("--coords-dir", type=Path, default=None, help="Patch coords (if not in feat .h5)")
    ap.add_argument("--output", type=Path, default=None, help="Save figure path (single-slide mode)")
    ap.add_argument("--output-dir", type=Path, default=None, help="Batch mode: save <slide_id>.png here")
    ap.add_argument(
        "--blockmap-dir",
        type=Path,
        default=None,
        help="If set with coords, save CLAM-style *_blockmap.h5 (attention_scores + coords) here",
    )
    ap.add_argument("--thumb-size", type=int, default=512)
    ap.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or cuda — bag forward device when using --checkpoint",
    )
    ap.add_argument("--cmap", type=str, default="turbo", help="Matplotlib colormap for overlay")
    ap.add_argument("--alpha", type=float, default=0.6, help="Overlay alpha (0–1)")
    ap.add_argument(
        "--gaussian-sigma",
        type=str,
        default="auto",
        help="Gaussian blur on rasterized heatmap: auto | off | positive float (larger = smoother)",
    )
    ap.add_argument("--percentile-low", type=float, default=5.0, help="Contrast stretch low percentile")
    ap.add_argument("--percentile-high", type=float, default=99.0, help="Contrast stretch high percentile")
    ap.add_argument(
        "--percentile-rank",
        action="store_true",
        help="Rank-normalize positive pixels before colormap (spread dynamic range; needs scipy)",
    )
    ap.add_argument(
        "--raster-out",
        type=Path,
        default=None,
        help="Save CLAM-style full-slide RGB overlay at a pyramid level (TIFF/PNG; needs coords + --wsi-dir)",
    )
    ap.add_argument(
        "--raster-dir",
        type=Path,
        default=None,
        help="Batch: save <slide_id>.tif under this directory (with --slides-csv)",
    )
    ap.add_argument(
        "--max-side",
        type=int,
        default=4096,
        help="Auto pyramid level: max longer side of rendered level (smaller = faster, coarser)",
    )
    ap.add_argument(
        "--vis-level",
        type=int,
        default=None,
        help="Force OpenSlide pyramid index for full-slide raster (overrides --max-side if set)",
    )
    ap.add_argument(
        "--mask-mode",
        type=str,
        default="none",
        choices=("none", "patches", "tissue"),
        help="Clip heatmap: none | patches (union of patch squares) | tissue (Otsu on slide RGB, heuristic)",
    )
    ap.add_argument(
        "--patch-mask-dilate",
        type=int,
        default=0,
        help="With mask-mode=patches: binary dilation iters on mask (tiny fringe inside footprint)",
    )
    ap.add_argument(
        "--no-print-probs",
        action="store_true",
        help="Do not print per-slide predicted class / softmax (checkpoint mode only)",
    )
    args = ap.parse_args()

    if args.attention_dir is None and args.checkpoint is None:
        raise SystemExit("Provide either --checkpoint (to compute attention) or --attention-dir (to load pre-extracted).")
    if args.attention_dir is not None and args.checkpoint is not None:
        raise SystemExit("Provide only one of --checkpoint or --attention-dir (not both).")

    print_probs = not args.no_print_probs
    dev = _resolve_device(args.device)
    g_sigma = _parse_gaussian_sigma(args.gaussian_sigma)

    if args.slides_csv is not None:
        if args.output_dir is None and args.raster_dir is None:
            raise SystemExit("--slides-csv requires --output-dir and/or --raster-dir.")
        if args.output_dir is not None:
            args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.raster_dir is not None:
            args.raster_dir.mkdir(parents=True, exist_ok=True)
        if args.blockmap_dir:
            args.blockmap_dir.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(args.slides_csv)
        if "slide_id" not in df.columns:
            raise SystemExit("CSV must contain a 'slide_id' column.")
        shared_model = None
        shared_name: str | None = None
        shared_fd: int | None = None
        if args.checkpoint is not None:
            shared_model, shared_name, shared_fd = _init_model_from_checkpoint(args.checkpoint)
            shared_model.to(dev)
        for _, row in df.iterrows():
            sid = str(row["slide_id"]).strip()
            out_png = (args.output_dir / f"{sid}.png") if args.output_dir else None
            rast = (args.raster_dir / f"{sid}.tif") if args.raster_dir else None
            try:
                _run_one_slide(
                    slide_id=sid,
                    feat_dir=args.feat_dir,
                    coords_dir=args.coords_dir,
                    checkpoint=args.checkpoint,
                    attention_dir=args.attention_dir,
                    wsi_dir=args.wsi_dir,
                    output=out_png,
                    raster_out=rast,
                    thumb_size=args.thumb_size,
                    device=dev,
                    cmap=args.cmap,
                    alpha=args.alpha,
                    blockmap_dir=args.blockmap_dir,
                    print_probs=print_probs,
                    gaussian_sigma=g_sigma,
                    percentile_low=args.percentile_low,
                    percentile_high=args.percentile_high,
                    use_percentile_rank=args.percentile_rank,
                    max_side=args.max_side,
                    vis_level=args.vis_level,
                    mask_mode=args.mask_mode,
                    patch_mask_dilate=args.patch_mask_dilate,
                    model=shared_model,
                    model_name_cached=shared_name,
                    ckpt_feat_dim=shared_fd,
                )
            except Exception as e:
                print(f"ERROR {sid}: {e}", file=sys.stderr)
                raise
        return

    if not args.slide_id:
        raise SystemExit("Provide --slide-id or --slides-csv.")

    if args.blockmap_dir:
        args.blockmap_dir.mkdir(parents=True, exist_ok=True)
    if args.raster_out is not None:
        args.raster_out = Path(args.raster_out)
        args.raster_out.parent.mkdir(parents=True, exist_ok=True)

    ax = _run_one_slide(
        slide_id=args.slide_id,
        feat_dir=args.feat_dir,
        coords_dir=args.coords_dir,
        checkpoint=args.checkpoint,
        attention_dir=args.attention_dir,
        wsi_dir=args.wsi_dir,
        output=args.output,
        raster_out=args.raster_out,
        thumb_size=args.thumb_size,
        device=dev,
        cmap=args.cmap,
        alpha=args.alpha,
        blockmap_dir=args.blockmap_dir,
        print_probs=print_probs,
        gaussian_sigma=g_sigma,
        percentile_low=args.percentile_low,
        percentile_high=args.percentile_high,
        use_percentile_rank=args.percentile_rank,
        max_side=args.max_side,
        vis_level=args.vis_level,
        mask_mode=args.mask_mode,
        patch_mask_dilate=args.patch_mask_dilate,
        model=None,
        model_name_cached=None,
        ckpt_feat_dim=None,
    )

    if ax is not None and args.output is None:
        import matplotlib.pyplot as plt

        plt.show()


if __name__ == "__main__":
    main()
