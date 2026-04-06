#!/usr/bin/env python3
"""Build a PANDA-style submission CSV from MIL checkpoints and pre-extracted .h5 features.

Expects the same layout as training: ``feat_dir/{image_id}.h5`` with dataset ``features`` shaped
``[N, feat_dim]``. Test / hold-out slides do not need real labels: dummy zeros are used only
to satisfy the dataset API.

Typical Kaggle columns: ``image_id``, ``isup_grade`` (integers 0–5).

Examples:

  # Single checkpoint (one fold or a dedicated final model)
  python scripts/make_submission.py \\
    --feat-dir /path/to/test_or_val_features \\
    --slides-csv /path/to/test.csv \\
    --checkpoint checkpoints/uni2_gated_final/fold0_best.pt \\
    --output submission.csv

  # Ensemble: average softmax probs over every ``fold*_best.pt`` in a directory
  python scripts/make_submission.py \\
    --feat-dir ... \\
    --slides-csv sample_submission.csv \\
    --checkpoint-dir checkpoints/uni2_gated_final \\
    --output submission.csv

Match ``--max-patches-val`` to training (default 512) for comparable behavior.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from mil.data import H5BagDataset, collate_bags
from mil.model import build_model
from mil.utils import seed_everything


def _load_ckpt(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _build_model_from_ckpt(ckpt: dict, device: torch.device) -> tuple[torch.nn.Module, bool]:
    model_name = ckpt.get("model_name", "gated")
    feat_dim = int(ckpt["feat_dim"])
    hidden = ckpt.get("hidden")
    if hidden is None:
        hidden = 128 if model_name == "attention" else 256
    else:
        hidden = int(hidden)
    top_k = int(ckpt.get("top_k", 8))
    kw: dict = {"feat_dim": feat_dim, "num_classes": 6, "hidden": hidden}
    if model_name == "ordinal":
        kw["top_k"] = top_k
    model = build_model(model_name, **kw)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    model = model.to(device)
    is_ordinal = model_name == "ordinal"
    return model, is_ordinal


@torch.no_grad()
def _run_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    is_ordinal: bool,
) -> tuple[torch.Tensor, list[str]]:
    """Return (probs [S, 6], slide_ids in row order)."""
    chunks: list[torch.Tensor] = []
    ids_order: list[str] = []
    for feats, mask, _lab, ids, _coords in loader:
        feats = feats.to(device)
        mask = mask.to(device) if mask is not None else None
        if is_ordinal:
            out = model(feats, mask)
            logits = out["ce_logits"]
        else:
            logits = model(feats, mask)
        chunks.append(torch.softmax(logits.float(), dim=-1).cpu())
        ids_order.extend(ids)
    return torch.cat(chunks, dim=0), ids_order


def main():
    ap = argparse.ArgumentParser(description="MIL inference → submission CSV (PANDA-style)")
    ap.add_argument("--feat-dir", type=Path, required=True)
    ap.add_argument(
        "--slides-csv",
        type=Path,
        required=True,
        help="CSV with ``image_id`` column listing slides (e.g. test.csv or sample_submission.csv)",
    )
    ap.add_argument("--checkpoint", type=Path, default=None, help="Single .pt checkpoint")
    ap.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        help="If set, ensemble all ``fold*_best.pt`` in this directory (same architecture)",
    )
    ap.add_argument("--output", type=Path, required=True, help="submission.csv path")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-patches-val", type=int, default=512)
    ap.add_argument("--coords-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument(
        "--fill-missing",
        type=int,
        default=None,
        metavar="ISUP",
        help="If some slide_ids lack .h5, fill their isup_grade with this int (e.g. 0). "
        "Default: exit with error if any missing.",
    )
    args = ap.parse_args()

    if (args.checkpoint is None) == (args.checkpoint_dir is None):
        ap.error("Provide exactly one of --checkpoint or --checkpoint-dir")

    root = Path(__file__).resolve().parent.parent
    for attr in ("feat_dir", "slides_csv", "checkpoint", "checkpoint_dir", "output", "coords_dir"):
        v = getattr(args, attr)
        if v is not None and not v.is_absolute():
            setattr(args, attr, root / v)

    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}", flush=True)

    slides = pd.read_csv(args.slides_csv)
    if "image_id" not in slides.columns:
        raise SystemExit("slides-csv must contain column 'image_id'")
    slide_ids = slides["image_id"].astype(str).str.strip().tolist()

    feat_dir = args.feat_dir
    have = {p.stem for p in feat_dir.glob("*.h5")}
    use_ids = [s for s in slide_ids if s in have]
    missing = [s for s in slide_ids if s not in have]
    if missing:
        print(f"Note: {len(missing)} slide ids have no .h5 under feat-dir", flush=True)
        if args.fill_missing is None:
            raise SystemExit(
                "Use --fill-missing ISUP (e.g. 0) for a full-length CSV, or extract .h5 for every id."
            )
    if not use_ids:
        raise SystemExit("No slides with features — nothing to score")

    dummy = pd.DataFrame({"image_id": use_ids, "isup_grade": 0}).set_index("image_id")
    val_ds = H5BagDataset(
        feat_dir,
        use_ids,
        dummy,
        coords_dir=args.coords_dir,
        training=False,
        max_patches_val=args.max_patches_val,
    )
    loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_bags,
        persistent_workers=args.num_workers > 0,
    )

    ckpt_paths: list[Path]
    if args.checkpoint is not None:
        ckpt_paths = [args.checkpoint]
    else:
        ckpt_paths = sorted(args.checkpoint_dir.glob("fold*_best.pt"))
        if not ckpt_paths:
            raise SystemExit(f"No fold*_best.pt in {args.checkpoint_dir}")

    first = _load_ckpt(ckpt_paths[0])
    is_ordinal = first.get("model_name", "") == "ordinal"
    # All ensemble members must match (light check)
    for p in ckpt_paths[1:]:
        c = _load_ckpt(p)
        if c.get("model_name") != first.get("model_name") or int(c["feat_dim"]) != int(first["feat_dim"]):
            raise SystemExit(f"Checkpoint mismatch: {ckpt_paths[0].name} vs {p.name}")

    prob_sum: torch.Tensor | None = None
    ids_ref: list[str] | None = None
    for ckpt_path in ckpt_paths:
        ckpt = _load_ckpt(ckpt_path)
        model, mo = _build_model_from_ckpt(ckpt, device)
        if mo != is_ordinal:
            raise SystemExit("Inconsistent ordinal flag across checkpoints")
        probs, ids_order = _run_probs(model, loader, device, is_ordinal=is_ordinal)
        if ids_ref is None:
            ids_ref = ids_order
        elif ids_ref != ids_order:
            raise SystemExit("Slide order changed between checkpoints (bug)")
        prob_sum = probs if prob_sum is None else prob_sum + probs
        print(f"  scored with {ckpt_path.name}", flush=True)

    assert prob_sum is not None and ids_ref is not None
    prob_mean = prob_sum / len(ckpt_paths)
    preds = prob_mean.argmax(dim=-1).numpy().astype(int)

    scored = pd.DataFrame({"image_id": ids_ref, "isup_grade": preds})
    template = pd.DataFrame({"image_id": slide_ids})
    out = template.merge(scored, on="image_id", how="left")
    if out["isup_grade"].isna().any():
        assert args.fill_missing is not None
        n_fill = int(out["isup_grade"].isna().sum())
        out["isup_grade"] = out["isup_grade"].fillna(float(args.fill_missing)).astype(int)
        print(f"Filled {n_fill} rows with --fill-missing {args.fill_missing}", flush=True)
    else:
        out["isup_grade"] = out["isup_grade"].astype(int)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(out)} rows)", flush=True)


if __name__ == "__main__":
    main()
