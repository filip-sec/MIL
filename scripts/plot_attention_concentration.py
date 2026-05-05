#!/usr/bin/env python3
"""Plot per-slide attention concentration (sorted weights + cumulative mass).

This script quantifies how concentrated MIL attention is on each slide:
- what fraction of patches carries 50% attention mass
- what fraction of patches carries 90% attention mass

It supports two input modes:
1) Compute attention from a checkpoint + feature .h5 files
2) Load pre-extracted attention vectors from --attention-dir (slide_id.pt)
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

# Repo root so `import mil` works when run from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mil.attention_map import get_attention_and_logits_for_slide
from mil.model import build_model
from mil.utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Plot MIL attention concentration per slide.")
    ap.add_argument("--feat-dir", type=Path, required=True, help="Directory with per-slide feature .h5 files.")
    ap.add_argument("--slide-id", type=str, default=None, help="Single slide ID.")
    ap.add_argument("--slides-csv", type=Path, default=None, help="CSV with a 'slide_id' column.")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint .pt to compute attention (exclusive with --attention-dir).",
    )
    ap.add_argument(
        "--attention-dir",
        type=Path,
        default=None,
        help="Directory with pre-extracted attention files <slide_id>.pt (exclusive with --checkpoint).",
    )
    ap.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figures/attention_concentration"),
        help="Directory to save per-slide concentration plots.",
    )
    ap.add_argument(
        "--output-tag",
        type=str,
        default=None,
        help=(
            "Optional suffix to include in output filenames/summary CSV "
            "(default: inferred from checkpoint dir or attention dir)."
        ),
    )
    ap.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional summary CSV path (default: <output-dir>/attention_concentration_summary.csv).",
    )
    ap.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Optional CSV with columns: slide_id,isup_grade for group-level concentration summaries.",
    )
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    return ap.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; using CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def init_model_from_checkpoint(ckpt_path: Path) -> tuple[torch.nn.Module, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    feat_dim = int(ckpt["feat_dim"])
    model_name = ckpt.get("model_name", "attention")
    hidden = ckpt.get("hidden", 256 if model_name != "attention" else 128)
    top_k = ckpt.get("top_k", 8)
    sd = ckpt.get("model_state_dict", {})

    if model_name == "clam":
        mid_dim = int(sd["fc.0.weight"].shape[0]) if "fc.0.weight" in sd else 512
        model_kwargs: dict[str, int] = {
            "feat_dim": feat_dim,
            "num_classes": 6,
            "mid_dim": mid_dim,
            "k_sample": int(top_k),
        }
        if "hidden" in ckpt:
            model_kwargs["attn_dim"] = int(ckpt["hidden"])
    else:
        model_kwargs = {"feat_dim": feat_dim, "num_classes": 6, "hidden": hidden}
        if model_name == "ordinal":
            model_kwargs["top_k"] = int(top_k)

    model = build_model(model_name, **model_kwargs)
    load_checkpoint(ckpt_path, model)
    model.eval()
    return model, feat_dim


def load_features_for_slide(h5_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(h5_path, "r") as f:
        feats = torch.from_numpy(f["features"][:]).float().unsqueeze(0)
        n = feats.size(1)
        mask = torch.ones(1, n, dtype=torch.bool)
    return feats, mask


def load_attention_vector(
    *,
    slide_id: str,
    feat_dir: Path,
    attention_dir: Path | None,
    model: torch.nn.Module | None,
    checkpoint_feat_dim: int | None,
    device: torch.device,
) -> np.ndarray:
    h5_path = feat_dir / f"{slide_id}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Features not found: {h5_path}")

    if attention_dir is not None:
        att_path = attention_dir / f"{slide_id}.pt"
        if not att_path.exists():
            raise FileNotFoundError(f"Attention file not found: {att_path}")
        att = torch.load(att_path, map_location="cpu", weights_only=True)
        att = torch.as_tensor(att).float().view(-1)
        with h5py.File(h5_path, "r") as f:
            n = int(f["features"].shape[0])
        if att.numel() != n:
            raise ValueError(f"Attention length mismatch for {slide_id}: expected {n}, got {att.numel()}")
        arr = att.numpy()
    else:
        if model is None or checkpoint_feat_dim is None:
            raise ValueError("Model/checkpoint feature dimension not initialized.")
        feats, mask = load_features_for_slide(h5_path)
        if feats.size(-1) != checkpoint_feat_dim:
            raise ValueError(
                f"Feature dim mismatch for {slide_id}: .h5 has D={feats.size(-1)}, "
                f"checkpoint expects {checkpoint_feat_dim}"
            )
        att, _ = get_attention_and_logits_for_slide(model, feats, mask, device=device)
        arr = att.detach().cpu().numpy()

    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    if np.any(arr < 0):
        raise ValueError(f"Negative attention value detected for {slide_id}.")
    total = float(arr.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Non-positive or invalid attention sum for {slide_id}.")
    return (arr / total).astype(np.float64)


def _top_mass_stats(attn_sorted: np.ndarray, mass: float) -> tuple[int, float]:
    if mass <= 0 or mass >= 1:
        raise ValueError("mass must be in (0, 1)")
    cum = np.cumsum(attn_sorted)
    k = int(np.searchsorted(cum, mass, side="left") + 1)
    frac = float(k / len(attn_sorted))
    return k, frac


def make_concentration_plot(
    attn: np.ndarray,
    slide_id: str,
    out_path: Path,
    dpi: int,
) -> dict[str, float | int | str]:
    attn_sorted = np.sort(attn)[::-1]
    cum = np.cumsum(attn_sorted)
    cum = cum / (cum[-1] + 1e-12)

    k50, frac50 = _top_mass_stats(attn_sorted, 0.5)
    k90, frac90 = _top_mass_stats(attn_sorted, 0.9)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # Left: sorted attention magnitudes (log scale)
    y_plot = np.clip(attn_sorted, 1e-12, None)
    axes[0].plot(y_plot, lw=1.5)
    axes[0].fill_between(np.arange(len(y_plot)), y_plot, alpha=0.25)
    axes[0].set_xlabel("Patch rank (highest attention first)")
    axes[0].set_ylabel("Attention weight")
    axes[0].set_yscale("log")
    axes[0].set_title(f"Sorted attention · {slide_id}")

    # Right: cumulative attention mass
    x_frac = np.arange(1, len(cum) + 1) / len(cum)
    axes[1].plot(x_frac, cum, lw=1.5)
    axes[1].axhline(0.5, ls=":", c="grey")
    axes[1].axhline(0.9, ls=":", c="grey")
    axes[1].set_xlabel("Fraction of patches (sorted)")
    axes[1].set_ylabel("Cumulative attention mass")
    axes[1].set_title("Attention concentration (CDF)")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1.01)
    axes[1].text(
        0.03,
        0.06,
        f"Top-{k50}/{len(attn_sorted)} patches = 50% mass ({100.0 * frac50:.2f}%)\n"
        f"Top-{k90}/{len(attn_sorted)} patches = 90% mass ({100.0 * frac90:.2f}%)",
        transform=axes[1].transAxes,
        fontsize=9,
        va="bottom",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "#cccccc"},
    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return {
        "slide_id": slide_id,
        "n_patches": int(len(attn_sorted)),
        "top50_patches": int(k50),
        "top50_frac": float(frac50),
        "top90_patches": int(k90),
        "top90_frac": float(frac90),
    }


def resolve_slide_ids(args: argparse.Namespace) -> list[str]:
    if args.slide_id and args.slides_csv:
        raise SystemExit("Use either --slide-id or --slides-csv, not both.")
    if not args.slide_id and not args.slides_csv:
        raise SystemExit("Provide --slide-id or --slides-csv.")
    if args.slide_id:
        return [args.slide_id.strip()]

    assert args.slides_csv is not None
    df = pd.read_csv(args.slides_csv)
    if "slide_id" not in df.columns:
        raise SystemExit(f"{args.slides_csv}: missing 'slide_id' column.")
    return [str(x).strip() for x in df["slide_id"].tolist() if str(x).strip()]


def _sanitize_tag(tag: str) -> str:
    tag = tag.strip()
    tag = re.sub(r"[^A-Za-z0-9._-]+", "-", tag)
    tag = re.sub(r"-{2,}", "-", tag).strip("-")
    return tag or "run"


def resolve_output_tag(args: argparse.Namespace) -> str:
    if args.output_tag:
        return _sanitize_tag(args.output_tag)
    if args.checkpoint is not None:
        return _sanitize_tag(args.checkpoint.parent.name)
    if args.attention_dir is not None:
        return _sanitize_tag(args.attention_dir.name)
    return "run"


def main() -> None:
    args = parse_args()
    dev = resolve_device(args.device)

    if (args.checkpoint is None) == (args.attention_dir is None):
        raise SystemExit("Provide exactly one of --checkpoint or --attention-dir.")

    slide_ids = resolve_slide_ids(args)
    output_tag = resolve_output_tag(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.summary_csv or (args.output_dir / f"attention_concentration_summary_{output_tag}.csv")

    model = None
    ckpt_feat_dim = None
    if args.checkpoint is not None:
        model, ckpt_feat_dim = init_model_from_checkpoint(args.checkpoint)
        model = model.to(dev)

    rows: list[dict[str, float | int | str]] = []
    for slide_id in slide_ids:
        attn = load_attention_vector(
            slide_id=slide_id,
            feat_dir=args.feat_dir,
            attention_dir=args.attention_dir,
            model=model,
            checkpoint_feat_dim=ckpt_feat_dim,
            device=dev,
        )
        out_path = args.output_dir / f"{slide_id}_{output_tag}_attention_concentration.png"
        row = make_concentration_plot(attn, slide_id, out_path, args.dpi)
        rows.append(row)
        print(
            f"{slide_id}: top50={row['top50_frac']:.4f} ({row['top50_patches']}/{row['n_patches']}), "
            f"top90={row['top90_frac']:.4f} ({row['top90_patches']}/{row['n_patches']})"
        )

    out_df = pd.DataFrame(rows).sort_values("slide_id").reset_index(drop=True)

    if args.labels_csv is not None:
        labels_df = pd.read_csv(args.labels_csv)
        id_col = None
        for candidate in ("slide_id", "image_id"):
            if candidate in labels_df.columns:
                id_col = candidate
                break
        if id_col is None or "isup_grade" not in labels_df.columns:
            raise SystemExit(
                f"{args.labels_csv}: expected columns ['isup_grade' + one of slide_id/image_id], "
                f"got {list(labels_df.columns)}"
            )

        labels_df = labels_df[[id_col, "isup_grade"]].copy()
        labels_df = labels_df.rename(columns={id_col: "slide_id"})
        labels_df["slide_id"] = labels_df["slide_id"].astype(str)
        labels_df["isup_grade"] = labels_df["isup_grade"].astype(int)
        out_df = out_df.merge(labels_df, on="slide_id", how="left")
        out_df["cancer_group"] = np.where(out_df["isup_grade"].fillna(0) > 0, "cancer", "benign_or_unknown")

        grp = out_df.groupby("cancer_group", dropna=False)[["top50_frac", "top90_frac"]].median()
        print("\nMedian concentration by group:")
        print(grp.to_string())

    out_df.to_csv(summary_csv, index=False)
    print(f"\nSaved {len(out_df)} slide summaries to {summary_csv}")


if __name__ == "__main__":
    main()
