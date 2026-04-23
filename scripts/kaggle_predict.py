#!/usr/bin/env python3
"""Build Kaggle PANDA submission.csv from test patch features (.h5) + trained checkpoint(s).

Requires one ``.h5`` per slide under ``--feat-dir`` (same layout as training), produced by thesame TRIDENT/encoder pipeline as the train set.

Typical flow------------
1. Download competition ``sample_submission.csv`` (defines ``image_id`` order and count).
2. Extract test WSI features to ``feat_dir`` (same key as train, default ``features``).
3. Run this script, then submit::

 kaggle competitions submit -c prostate-cancer-grade-assessment \\
       -f submission.csv -m "CLAM uni_v2 5-fold avg"

File submission uses ``-f`` and ``-m`` only. Options ``-k`` / ``-v`` belong to other Kaggle
commands (e.g. notebook output), not the usual ``competitions submit`` file upload.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import h5py
import pandas as pd
import torch

from mil.config import merge_runtime_config
from mil.encoders import infer_encoder_from_feature_dir
from mil.model import build_model


def _build_from_checkpoint(ckpt: dict) -> tuple[torch.nn.Module, str, int, int, str | None]:
    """Return (model, model_name, feat_dim, top_k, encoder_name) without loading weights."""
    model_name = ckpt.get("model_name", "attention")
    encoder_name = ckpt.get("encoder_name")
    hidden = ckpt.get("hidden", 256 if model_name != "attention" else 128)
    top_k = ckpt.get("top_k", 8)
    feat_dim = ckpt.get("feat_dim")
    if feat_dim is None:
        raise ValueError("Checkpoint missing feat_dim")
    if model_name == "clam":
        model = build_model(
            "clam",
            feat_dim=feat_dim,
            num_classes=6,
            k_sample=ckpt.get("clam_k_sample", top_k),
        )
    else:
        kw: dict = {"feat_dim": feat_dim, "num_classes": 6, "hidden": hidden}
        if model_name == "ordinal":
            kw["top_k"] = top_k
        model = build_model(model_name, **kw)
    return model, model_name, feat_dim, top_k, encoder_name


def _slide_predict_one(
    model: torch.nn.Module,
    model_name: str,
    feats: torch.Tensor,
    device: torch.device,
) -> int:
    """One slide, feats [N, D] with N>=1. Matches ``mil.train.evaluate`` (ordinal uses ``predict``)."""
    x = feats.unsqueeze(0).to(device)
    n = x.size(1)
    mask = torch.ones(1, n, dtype=torch.bool, device=device)
    with torch.no_grad():
        if model_name == "ordinal":
            return int(model.predict(x, mask).item())
        if model_name == "clam":
            return int(model(x, mask).argmax(dim=1).item())
        return int(model(x, mask).argmax(dim=1).item())


def _majority_vote(votes: list[int]) -> int:
    c = Counter(votes)
    best = max(c.values())
    tied = [k for k, v in c.items() if v == best]
    return min(tied)


@torch.no_grad()
def predict_ensemble_class(
    models: list[torch.nn.Module],
    model_names: list[str],
    feats: torch.Tensor,
    device: torch.device,
) -> int:
    """Each checkpoint votes; ties broken by lowest ISUP (conservative)."""
    votes = [_slide_predict_one(m, n, feats, device) for m, n in zip(models, model_names)]
    return _majority_vote(votes)


def main() -> None:
    p = argparse.ArgumentParser(description="Kaggle PANDA submission.csv from .h5 features + checkpoint(s)")
    p.add_argument("--config", type=Path, default=None, help="Optional YAML config (defaults to configs/pipelines/kaggle_predict.yaml)")
    p.add_argument("--feat-dir", type=Path, default=None, help="Directory with <image_id>.h5 test features")
    p.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        default=None,
        help="One or more fold*_best.pt (same feat_dim); probabilities are averaged",
    )
    p.add_argument(
        "--sample-submission",
        type=Path,
        default=None,
        help="Kaggle sample_submission.csv (sets row order and required image_ids)",
    )
    p.add_argument("-o", "--output", type=Path, default=None)
    p.add_argument("--key", type=str, default=None, help="H5 dataset name for patch features")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--max-patches",
        type=int,
        default=None,
        help="Cap patches per slide (same idea as val). Default: use all patches.",
    )
    p.add_argument("--seed", type=int, default=None, help="Subsampling seed when --max-patches is set")
    p.add_argument(
        "--default-grade",
        type=int,
        default=None,
        help="isup_grade if .h5 is missing or empty (must be 0..5)",
    )
    args = p.parse_args()

    overrides = {key: value for key, value in vars(args).items() if key != "config" and value is not None}
    runtime = merge_runtime_config("kaggle_predict", config_path=args.config, overrides=overrides)

    feat_dir = runtime.get("feat_dir")
    ckpt_paths = runtime.get("checkpoint")
    out_path = runtime.get("output")
    sample_path = runtime.get("sample_submission")
    if feat_dir is None or ckpt_paths is None:
        raise SystemExit("--feat-dir and --checkpoint are required (or set them in the config file)")
    if runtime.get("default_grade") < 0 or runtime.get("default_grade") > 5:
        raise SystemExit("--default-grade must be in 0..5")

    device_name = runtime.get("device")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    models: list[torch.nn.Module] = []
    names: list[str] = []
    ref_fd: int | None = None
    ref_name: str | None = None
    ref_encoder: str | None = None

    for ckpt_path in ckpt_paths:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
            raise ValueError(f"Invalid checkpoint: {ckpt_path}")
        model, model_name, feat_dim, _top_k, encoder_name = _build_from_checkpoint(ckpt)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.eval()
        model.to(device)
        models.append(model)
        names.append(model_name)
        if ref_fd is None:
            ref_fd, ref_name = feat_dim, model_name
            ref_encoder = encoder_name
        else:
            if feat_dim != ref_fd:
                raise ValueError(f"feat_dim mismatch: {ckpt_path} has {feat_dim}, expected {ref_fd}")
            if model_name != ref_name:
                raise ValueError(
                    f"model_name mismatch: {ckpt_path} is {model_name}, expected {ref_name} "
                    "(ensemble only supports identical architectures)."
                )
            if encoder_name != ref_encoder:
                raise ValueError(
                    f"encoder_name mismatch: {ckpt_path} is {encoder_name!r}, expected {ref_encoder!r}"
                )

    inferred_encoder = infer_encoder_from_feature_dir(feat_dir)
    if ref_encoder and inferred_encoder and ref_encoder != inferred_encoder:
        raise ValueError(
            f"Feature dir {feat_dir} looks like encoder '{inferred_encoder}', but checkpoints expect '{ref_encoder}'."
        )

    if sample_path is not None:
        if not sample_path.exists():
            raise SystemExit(f"Missing {sample_path}")
        sub = pd.read_csv(sample_path)
        if "image_id" not in sub.columns:
            raise SystemExit("sample_submission must contain column image_id")
        image_ids = sub["image_id"].astype(str).tolist()
    else:
        image_ids = sorted(p.stem for p in feat_dir.glob("*.h5"))

    if not image_ids:
        raise SystemExit("No image_ids (provide --sample-submission or ensure *.h5 in feat-dir)")

    g = torch.Generator(device="cpu")
    g.manual_seed(runtime.get("seed"))

    rows: list[tuple[str, int]] = []
    n_missing = 0
    n_empty = 0
    checked_dim = False

    for i, sid in enumerate(image_ids):
        h5_path = feat_dir / f"{sid}.h5"
        if not h5_path.exists():
            rows.append((sid, runtime.get("default_grade")))
            n_missing += 1
            continue
        with h5py.File(h5_path, "r") as f:
            if runtime.get("key") not in f:
                raise KeyError(f"{h5_path}: missing dataset {runtime.get('key')!r}")
            feats = torch.from_numpy(f[runtime.get("key")][:]).float()
        if not checked_dim:
            if feats.ndim != 2 or feats.size(1) != ref_fd:
                raise ValueError(
                    f"{h5_path} has feature shape {tuple(feats.shape)} but checkpoint expects feat_dim={ref_fd}"
                )
            checked_dim = True
        if feats.size(0) == 0:
            rows.append((sid, runtime.get("default_grade")))
            n_empty += 1
            continue
        if runtime.get("max_patches") is not None and feats.size(0) > runtime.get("max_patches"):
            idx = torch.randperm(feats.size(0), generator=g)[: runtime.get("max_patches")]
            feats = feats[idx]
        pred = predict_ensemble_class(models, names, feats, device)
        rows.append((sid, pred))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(image_ids)}", flush=True)

    out = pd.DataFrame(rows, columns=["image_id", "isup_grade"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"Wrote {out_path} ({len(out)} rows)", flush=True)
    if n_missing or n_empty:
        print(
            f"  Warning: default_grade={runtime.get('default_grade')} for missing={n_missing} empty_h5={n_empty}",
            flush=True,
        )


if __name__ == "__main__":
    main()
