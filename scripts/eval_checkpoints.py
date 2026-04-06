#!/usr/bin/env python3
"""Evaluate trained MIL checkpoints on SICAPv2 WSI test split.

Loads one or many checkpoints, runs inference on WSI-level .h5 features, and writes:
- per-checkpoint metrics CSV
- per-slide predictions CSV(s)

SICAPv2 specifics:
- labels file: ``wsi_labels.xlsx`` with columns:
  ``slide_id``, ``Gleason_primary``, ``Gleason_secondary``
- test partition file: ``partition/Test/Test.xlsx`` with patch-level ``image_name``
  where slide id is prefix before ``_Block_``.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
from torch.utils.data import DataLoader

from mil.data import H5BagDataset, collate_bags
from mil.model import build_model
from mil.utils import compute_metrics, seed_everything


def _read_excel(path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(path)
    except ImportError as exc:
        raise SystemExit(
            "Reading .xlsx requires openpyxl. Install it in your env, e.g.:\n"
            "  pip install openpyxl"
        ) from exc


def _gleason_to_isup(primary: int, secondary: int) -> int:
    """Map Gleason primary/secondary to ISUP grade group (0..5)."""
    p, s = int(primary), int(secondary)
    if p <= 0 and s <= 0:
        return 0
    if (p, s) == (3, 3):
        return 1
    if (p, s) == (3, 4):
        return 2
    if (p, s) == (4, 3):
        return 3
    if (p, s) in {(4, 4), (3, 5), (5, 3)}:
        return 4
    if (p, s) in {(4, 5), (5, 4), (5, 5)}:
        return 5
    score = p + s
    if score <= 6:
        return 1
    if score == 7:
        return 2 if p == 3 else 3
    if score == 8:
        return 4
    return 5


def _extract_slide_id(image_name: str) -> str:
    stem = Path(str(image_name)).stem
    m = re.match(r"^([^_]+)_Block_", stem)
    if m:
        return m.group(1)
    return stem.split("_", 1)[0]


def _load_ckpt(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _build_model_from_ckpt(ckpt: dict, device: torch.device) -> tuple[torch.nn.Module, bool]:
    model_name = ckpt.get("model_name", "gated")
    feat_dim = int(ckpt["feat_dim"])
    hidden = ckpt.get("hidden")
    hidden = int(hidden) if hidden is not None else (128 if model_name == "attention" else 256)
    top_k = int(ckpt.get("top_k", 8))
    kw = {"feat_dim": feat_dim, "num_classes": 6, "hidden": hidden}
    if model_name == "ordinal":
        kw["top_k"] = top_k
    model = build_model(model_name, **kw)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model = model.to(device).eval()
    return model, model_name == "ordinal"


@torch.no_grad()
def _run_probs(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    is_ordinal: bool,
) -> tuple[torch.Tensor, list[str]]:
    chunks: list[torch.Tensor] = []
    ids_order: list[str] = []
    for feats, mask, _labels, ids, _coords in loader:
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


def _save_predictions(path: Path, ids: list[str], y_true: list[int], probs: torch.Tensor) -> None:
    preds = probs.argmax(dim=-1).numpy().astype(int)
    df = pd.DataFrame(
        {
            "slide_id": ids,
            "y_true": y_true,
            "y_pred": preds,
        }
    )
    for i in range(6):
        df[f"prob_{i}"] = probs[:, i].numpy()
    df.to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate MIL checkpoints on SICAPv2 test WSI split")
    ap.add_argument("--feat-dir", type=Path, required=True, help="Directory with WSI .h5 features")
    ap.add_argument("--labels-xlsx", type=Path, required=True, help="SICAPv2 wsi_labels.xlsx")
    ap.add_argument("--test-xlsx", type=Path, required=True, help="SICAPv2 partition/Test/Test.xlsx")
    ap.add_argument("--checkpoint", type=Path, default=None, help="Single checkpoint .pt")
    ap.add_argument("--checkpoint-dir", type=Path, default=None, help="Directory with checkpoint .pt files")
    ap.add_argument(
        "--checkpoint-pattern",
        type=str,
        default="fold*_best.pt",
        help="Glob used when --checkpoint-dir is set",
    )
    ap.add_argument("--out-dir", type=Path, required=True, help="Output directory for metrics and predictions")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--max-patches-val", type=int, default=512)
    ap.add_argument("--coords-dir", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument(
        "--strict",
        action="store_true",
        help="Fail if test slide ids are missing labels or features",
    )
    ap.add_argument(
        "--ensemble",
        action="store_true",
        help="Also score mean-probability ensemble over all checkpoints",
    )
    args = ap.parse_args()

    if (args.checkpoint is None) == (args.checkpoint_dir is None):
        ap.error("Provide exactly one of --checkpoint or --checkpoint-dir")

    root = Path(__file__).resolve().parent.parent
    for attr in (
        "feat_dir",
        "labels_xlsx",
        "test_xlsx",
        "checkpoint",
        "checkpoint_dir",
        "out_dir",
        "coords_dir",
    ):
        v = getattr(args, attr)
        if v is not None and not v.is_absolute():
            setattr(args, attr, root / v)

    seed_everything(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}", flush=True)

    labels_df = _read_excel(args.labels_xlsx)
    required = {"slide_id", "Gleason_primary", "Gleason_secondary"}
    missing_cols = sorted(required - set(labels_df.columns))
    if missing_cols:
        raise SystemExit(f"labels-xlsx missing required columns: {missing_cols}")
    labels_df["slide_id"] = labels_df["slide_id"].astype(str).str.strip()
    labels_df["isup_grade"] = [
        _gleason_to_isup(p, s)
        for p, s in zip(labels_df["Gleason_primary"].tolist(), labels_df["Gleason_secondary"].tolist())
    ]
    labels_df = labels_df.drop_duplicates(subset=["slide_id"]).set_index("slide_id")

    test_df = _read_excel(args.test_xlsx)
    if "image_name" not in test_df.columns:
        raise SystemExit("test-xlsx must contain 'image_name' column")
    test_ids_all = [_extract_slide_id(v) for v in test_df["image_name"].astype(str).tolist()]
    test_slide_ids = list(dict.fromkeys(test_ids_all))

    have_h5 = {p.stem for p in args.feat_dir.glob("*.h5")}
    missing_feat = [sid for sid in test_slide_ids if sid not in have_h5]
    missing_lab = [sid for sid in test_slide_ids if sid not in labels_df.index]
    if missing_feat:
        print(f"Missing features for {len(missing_feat)} test slides", flush=True)
    if missing_lab:
        print(f"Missing labels for {len(missing_lab)} test slides", flush=True)
    if args.strict and (missing_feat or missing_lab):
        raise SystemExit("Strict mode failed: missing feature/label rows")

    eval_ids = [sid for sid in test_slide_ids if sid in have_h5 and sid in labels_df.index]
    if not eval_ids:
        raise SystemExit("No test slides have both feature .h5 and labels")
    print(f"Evaluating on {len(eval_ids)} test slides", flush=True)

    eval_labels = labels_df.loc[eval_ids, ["isup_grade"]]
    ds = H5BagDataset(
        args.feat_dir,
        eval_ids,
        eval_labels,
        coords_dir=args.coords_dir,
        training=False,
        max_patches_val=args.max_patches_val,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_bags,
        persistent_workers=args.num_workers > 0,
    )

    if args.checkpoint is not None:
        ckpt_paths = [args.checkpoint]
    else:
        ckpt_paths = sorted(args.checkpoint_dir.glob(args.checkpoint_pattern))
        if not ckpt_paths:
            raise SystemExit(f"No checkpoints matched: {args.checkpoint_dir / args.checkpoint_pattern}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_rows = []
    prob_list: list[torch.Tensor] = []
    ids_ref: list[str] | None = None
    y_true_ref: list[int] | None = None

    for ckpt_path in ckpt_paths:
        ckpt = _load_ckpt(ckpt_path)
        model, is_ordinal = _build_model_from_ckpt(ckpt, device)
        probs, ids_order = _run_probs(model, loader, device, is_ordinal=is_ordinal)
        y_true = labels_df.loc[ids_order, "isup_grade"].astype(int).tolist()
        preds = probs.argmax(dim=-1).numpy().astype(int)
        m = compute_metrics(preds, pd.Series(y_true).to_numpy())
        metrics_rows.append(
            {
                "checkpoint": str(ckpt_path),
                "model_name": ckpt.get("model_name", "unknown"),
                "feat_dim": int(ckpt.get("feat_dim", -1)),
                "n_test": len(ids_order),
                **m,
            }
        )
        pred_path = args.out_dir / f"predictions_{ckpt_path.stem}.csv"
        _save_predictions(pred_path, ids_order, y_true, probs)
        print(
            f"{ckpt_path.name}: qwk={m['qwk']:.4f} bal_acc={m['balanced_accuracy']:.4f} "
            f"acc={m['accuracy']:.4f} mae={m['mae']:.4f}",
            flush=True,
        )
        print(f"  wrote {pred_path}", flush=True)

        if ids_ref is None:
            ids_ref = ids_order
            y_true_ref = y_true
        elif ids_ref != ids_order:
            raise SystemExit("Slide order mismatch between checkpoints")
        prob_list.append(probs)

    if args.ensemble and len(prob_list) > 1:
        assert ids_ref is not None and y_true_ref is not None
        ens_probs = torch.stack(prob_list, dim=0).mean(dim=0)
        ens_preds = ens_probs.argmax(dim=-1).numpy().astype(int)
        em = compute_metrics(ens_preds, pd.Series(y_true_ref).to_numpy())
        metrics_rows.append(
            {
                "checkpoint": "ENSEMBLE_MEAN",
                "model_name": "ensemble",
                "feat_dim": -1,
                "n_test": len(ids_ref),
                **em,
            }
        )
        ens_path = args.out_dir / "predictions_ensemble_mean.csv"
        _save_predictions(ens_path, ids_ref, y_true_ref, ens_probs)
        print(
            f"ENSEMBLE_MEAN: qwk={em['qwk']:.4f} bal_acc={em['balanced_accuracy']:.4f} "
            f"acc={em['accuracy']:.4f} mae={em['mae']:.4f}",
            flush=True,
        )
        print(f"  wrote {ens_path}", flush=True)

    metrics_df = pd.DataFrame(metrics_rows).sort_values("qwk", ascending=False)
    metrics_csv = args.out_dir / "metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"\nWrote {metrics_csv}", flush=True)
    print(metrics_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
