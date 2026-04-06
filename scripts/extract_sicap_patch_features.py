#!/usr/bin/env python3
"""Extract SICAPv2 patch-image features and save WSI-level MIL bags.

SICAPv2 contains pre-cut patch images (not raw WSIs), so PANDA-style WSI segmentation
(`grandqc`/`hest`) is not applicable. This script groups patches by slide ID prefix
and writes one `.h5` per slide with datasets:

- `features`: [N_patches, feat_dim] float32
- `coords`: [N_patches, 2] int64 (xini, yini parsed from filename; -1 if missing)

Filename convention expected (from SICAPv2):
`<slide>_Block_Region_..._xini_<X>_yini_<Y>.jpg`
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import h5py
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


def _set_hf_cache_dir(path: Path) -> None:
    root = path.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    hub = root / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(hub)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub))


def _read_hf_token_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"--hf-token-file not found: {path}")
    token = path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    if not token:
        raise ValueError(f"Empty first line in {path}")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    os.environ["HF_TOKEN"] = token


def _build_backbone(name: str, *, pretrained: bool = True):
    import timm
    import timm.layers
    import torch.nn as nn
    from torchvision.models import ResNet18_Weights, resnet18

    if name == "virchow2":
        return timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=pretrained,
            num_classes=0,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=nn.SiLU,
            reg_tokens=4,
        )
    if name == "uni2_h":
        return timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h",
            pretrained=pretrained,
            img_size=224,
            patch_size=14,
            depth=24,
            num_heads=24,
            init_values=1e-5,
            embed_dim=1536,
            mlp_ratio=2.66667 * 2,
            num_classes=0,
            no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=nn.SiLU,
            reg_tokens=8,
            dynamic_img_size=True,
        )
    if name == "resnet18":
        m = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        m.fc = torch.nn.Identity()
        return m
    raise ValueError(f"Unsupported backbone: {name}")


def _extract_slide_id(image_name: str) -> str:
    stem = Path(str(image_name)).stem
    m = re.match(r"^([^_]+)_Block_", stem)
    if m:
        return m.group(1)
    return stem.split("_", 1)[0]


def _parse_xy(path: Path) -> tuple[int, int]:
    m = re.search(r"_xini_(\d+)_yini_(\d+)", path.stem)
    if not m:
        return -1, -1
    return int(m.group(1)), int(m.group(2))


class PatchDataset(Dataset):
    def __init__(self, paths: list[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        with Image.open(p) as img:
            img = img.convert("RGB")
            x = self.transform(img)
        xy = _parse_xy(p)
        return x, xy


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract SICAPv2 patch features into WSI-level .h5 bags")
    ap.add_argument("--images-dir", type=Path, required=True, help="SICAPv2 images directory")
    ap.add_argument("--split-xlsx", type=Path, required=True, help="SICAPv2 split file with image_name column")
    ap.add_argument("--out-dir", type=Path, required=True, help="Output dir for <slide_id>.h5 bags")
    ap.add_argument("--backbone", choices=["virchow2", "uni2_h", "resnet18"], default="virchow2")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit-slides", type=int, default=0, help="Debug: process only first N slides (0=all)")
    ap.add_argument("--hf-token-file", type=Path, default=None)
    ap.add_argument("--hf-cache-dir", type=Path, default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.hf_cache_dir is not None:
        _set_hf_cache_dir(args.hf_cache_dir)
    if args.hf_token_file is not None:
        _read_hf_token_file(args.hf_token_file.expanduser())

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}", flush=True)

    split_df = pd.read_excel(args.split_xlsx)
    if "image_name" not in split_df.columns:
        raise SystemExit("--split-xlsx must contain column 'image_name'")
    slide_ids = list(dict.fromkeys(_extract_slide_id(x) for x in split_df["image_name"].astype(str).tolist()))
    if args.limit_slides > 0:
        slide_ids = slide_ids[: args.limit_slides]
    print(f"Target slides from split: {len(slide_ids)}", flush=True)

    model = _build_backbone(args.backbone, pretrained=True).to(device).eval()
    import timm.data

    data_cfg = timm.data.resolve_data_config({}, model=model)
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total_written = 0

    for i, sid in enumerate(slide_ids, start=1):
        out_h5 = args.out_dir / f"{sid}.h5"
        if out_h5.exists() and not args.overwrite:
            print(f"[{i}/{len(slide_ids)}] skip existing {out_h5.name}", flush=True)
            continue

        patch_paths = sorted(args.images_dir.glob(f"{sid}_Block_*.jpg"))
        if not patch_paths:
            patch_paths = sorted(args.images_dir.glob(f"{sid}_Block_*.jpeg"))
        if not patch_paths:
            print(f"[{i}/{len(slide_ids)}] WARNING: no patches found for {sid}", flush=True)
            continue

        ds = PatchDataset(patch_paths, transform)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=args.num_workers > 0,
        )

        feat_chunks: list[torch.Tensor] = []
        coord_chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for imgs, coords in loader:
                imgs = imgs.to(device, non_blocking=True)
                feats = model(imgs)
                if isinstance(feats, (tuple, list)):
                    feats = feats[0]
                if isinstance(feats, dict):
                    raise RuntimeError("Backbone returned dict output; adjust extraction path")
                if feats.ndim == 3:
                    feats = feats[:, 0]
                feat_chunks.append(feats.float().cpu())
                coord_chunks.append(coords.long().cpu())

        features = torch.cat(feat_chunks, dim=0).numpy()
        coords_np = torch.cat(coord_chunks, dim=0).numpy()

        with h5py.File(out_h5, "w") as f:
            f.create_dataset("features", data=features, compression="gzip")
            f.create_dataset("coords", data=coords_np, compression="gzip")
            f.attrs["slide_id"] = sid
            f.attrs["backbone"] = args.backbone
            f.attrs["n_patches"] = int(features.shape[0])
            f.attrs["feat_dim"] = int(features.shape[1])

        total_written += 1
        print(
            f"[{i}/{len(slide_ids)}] wrote {out_h5.name} "
            f"(patches={features.shape[0]}, feat_dim={features.shape[1]})",
            flush=True,
        )

    print(f"Done. Wrote {total_written} slide bags into {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
