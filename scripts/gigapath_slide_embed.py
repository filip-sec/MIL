#!/usr/bin/env python3
"""Generate per-slide embeddings with the experimental Prov-GigaPath slide encoder."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mil.config import merge_runtime_config
from mil.encoders import get_encoder_spec, resolve_hf_token


def _load_slide_ids(slides_csv: Path | None, feat_dir: Path) -> list[str]:
    if slides_csv is not None:
        df = pd.read_csv(slides_csv)
        if "image_id" not in df.columns:
            raise SystemExit("slides-csv must contain column 'image_id'")
        return df["image_id"].astype(str).tolist()
    return sorted(path.stem for path in feat_dir.glob("*.h5"))


def _load_coords_from_file(path: Path, n_patches: int) -> torch.Tensor:
    with h5py.File(path, "r") as cf:
        for key in ("coords", "coordinates", "patches"):
            if key in cf and cf[key].shape[0] == n_patches:
                return torch.from_numpy(cf[key][:]).float()
    raise ValueError(f"No coords-compatible dataset found in {path}")


def _load_slide_inputs(h5_path: Path, coords_dir: Path | None) -> tuple[torch.Tensor, torch.Tensor]:
    with h5py.File(h5_path, "r") as f:
        feats = torch.from_numpy(f["features"][:]).float()
        if "coords" in f and f["coords"].shape[0] == feats.shape[0]:
            coords = torch.from_numpy(f["coords"][:]).float()
        elif coords_dir is not None:
            coords = None
            for name in (f"{h5_path.stem}.h5", f"{h5_path.stem}_patches.h5"):
                candidate = coords_dir / name
                if candidate.exists():
                    coords = _load_coords_from_file(candidate, feats.shape[0])
                    break
            if coords is None:
                raise FileNotFoundError(f"No matching coords file found for {h5_path.stem} under {coords_dir}")
        else:
            raise ValueError(f"{h5_path} has no coords and no --coords-dir was provided")
    if feats.ndim != 2 or coords.ndim != 2:
        raise ValueError(f"Expected 2D features and coords for {h5_path}")
    return feats, coords[:, :2]


def _forward_slide_encoder(model, feats: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    last_exc: Exception | None = None
    candidates = [
        (feats, coords),
        (feats.unsqueeze(0), coords.unsqueeze(0)),
    ]
    for feat_arg, coord_arg in candidates:
        try:
            output = model(feat_arg, coord_arg)
            if isinstance(output, (tuple, list)):
                output = output[0]
            return output.squeeze().detach().cpu().float()
        except Exception as exc:  # noqa: BLE001 - try the next compatible calling convention
            last_exc = exc
    raise RuntimeError(f"Prov-GigaPath slide encoder forward failed for all supported input shapes: {last_exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Prov-GigaPath slide encoder over cached tile embeddings")
    ap.add_argument("--config", type=Path, default=None, help="Optional YAML config (defaults to configs/pipelines/gigapath_slide_embed.yaml)")
    ap.add_argument("--feat-dir", type=Path, default=None, help="Directory with features_gigapath/*.h5")
    ap.add_argument("--out-dir", type=Path, default=None, help="Output dir for slide_id.pt embeddings")
    ap.add_argument("--slides-csv", type=Path, default=None, help="Optional CSV with image_id column")
    ap.add_argument("--coords-dir", type=Path, default=None, help="Fallback coords dir if not present in .h5 files")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--repo-root", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true", default=None, help="Recompute embeddings even if slide_id.pt exists")
    ap.add_argument("--limit", type=int, default=None, help="Encode only the first N slides")
    args = ap.parse_args()

    overrides = {key: value for key, value in vars(args).items() if key != "config" and value is not None}
    runtime = merge_runtime_config("gigapath_slide_embed", config_path=args.config, overrides=overrides)

    feat_dir = runtime.get("feat_dir")
    out_dir = runtime.get("out_dir")
    coords_dir = runtime.get("coords_dir")
    slides_csv = runtime.get("slides_csv")
    repo_root = runtime.get("repo_root")
    if feat_dir is None or out_dir is None:
        raise SystemExit("--feat-dir and --out-dir are required (or set them in the config file)")

    spec = get_encoder_spec("gigapath")
    token, source = resolve_hf_token(repo_root, spec.key, env=os.environ)
    os.environ["HF_TOKEN"] = token

    try:
        from huggingface_hub import hf_hub_download, login
    except ImportError as exc:
        raise SystemExit("gigapath_slide_embed.py requires huggingface_hub") from exc

    try:
        hf_hub_download(repo_id=spec.repo_id, filename="config.json", token=token)
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            f"Unable to access {spec.repo_id} using token source '{source}'. "
            "Check that the token is valid and that Hugging Face access has been granted."
        ) from exc

    login(token=token)

    try:
        import gigapath
    except ImportError as exc:
        raise SystemExit("gigapath_slide_embed.py requires the prov-gigapath package in the current environment") from exc

    device_name = runtime.get("device")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    model = gigapath.slide_encoder.create_model(
        f"hf_hub:{spec.repo_id}",
        "gigapath_slide_enc12l768d",
        spec.feature_dim,
    )
    model = model.to(device).eval()

    slide_ids = _load_slide_ids(slides_csv, feat_dir)
    if runtime.get("limit") is not None:
        slide_ids = slide_ids[: runtime.get("limit")]

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "slide_encoder_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "encoder_name": spec.key,
                "repo_id": spec.repo_id,
                "source_feature_dir": str(feat_dir),
                "coords_dir": str(coords_dir) if coords_dir is not None else None,
                "token_source": source,
                "config_name": runtime.config_name,
                "config_path": str(runtime.config_path),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    produced = 0
    for idx, slide_id in enumerate(slide_ids, start=1):
        out_path = out_dir / f"{slide_id}.pt"
        if out_path.exists() and not runtime.get("overwrite"):
            continue
        h5_path = feat_dir / f"{slide_id}.h5"
        if not h5_path.exists():
            print(f"Skipping {slide_id}: missing {h5_path}", flush=True)
            continue
        feats, coords = _load_slide_inputs(h5_path, coords_dir)
        feats = feats.to(device=device)
        coords = coords.to(device=device, dtype=torch.long)
        with torch.inference_mode():
            emb = _forward_slide_encoder(model, feats, coords)
        torch.save(emb, out_path)
        produced += 1
        if idx % 100 == 0:
            print(f"  encoded {idx}/{len(slide_ids)} slides", flush=True)

    print(f"Saved {produced} slide embeddings to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
