"""Dataset and collation for .h5 bag features."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset


def _subsample_patches(
    feats: torch.Tensor,
    coords: Optional[torch.Tensor],
    n_keep: int,
    attention_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Subsample patches. If attention_weights given, bias toward high-attention patches."""
    n = feats.size(0)
    if n_keep >= n:
        return feats, coords
    if attention_weights is not None and attention_weights.numel() == n:
        # Biased: sample proportionally to attention (add eps for stability)
        probs = attention_weights.float() + 1e-8
        probs = probs / probs.sum()
        idx = torch.multinomial(probs, num_samples=min(n_keep, n), replacement=False)
    else:
        idx = torch.randperm(n, device=feats.device)[:n_keep]
    feats = feats[idx]
    coords = coords[idx] if coords is not None else None
    return feats, coords


class H5BagDataset(Dataset):
    """Load pre-extracted patch features from .h5 files.

    Each item returns (feats, coords, label, slide_id) where
    feats: [N_patches, feat_dim], coords: [N_patches, 2+] or None,
    label: int (ISUP 0-5).

    TRIDENT-style outputs store ``coords`` in the **same** file as ``features`` (level-0 top-left
    ``(x, y)`` pixels, often int64 on disk; returned as float32 tensors). If absent, ``coords_dir``
    is tried as a fallback for separate coord files.

    Smart sampling (training only):
    - max_patches: cap bag size (random subsample if n > max_patches)
    - patch_dropout: further random subsample
    - attention_weights: optional dict slide_id -> [n] weights for biased sampling (keep high-attention)
    """

    def __init__(
        self,
        feat_dir: Path,
        slide_ids: list[str],
        train_labels: pd.DataFrame,
        key: str = "features",
        coords_key: str = "coords",
        coords_dir: Optional[Path] = None,
        patch_dropout: float = 0.0,
        max_patches: Optional[int] = None,
        max_patches_val: Optional[int] = None,
        attention_weights: Optional[dict[str, torch.Tensor]] = None,
        training: bool = False,
    ):
        self.feat_dir = Path(feat_dir)
        self.slide_ids = list(slide_ids)
        self.labels = train_labels
        self.key = key
        self.coords_key = coords_key
        self.coords_dir = Path(coords_dir) if coords_dir else None
        self.patch_dropout = patch_dropout
        self.max_patches = max_patches
        self.max_patches_val = max_patches_val
        self.attention_weights = attention_weights or {}
        self.training = training

    def __len__(self) -> int:
        return len(self.slide_ids)

    def _load_coords(self, slide_id: str, h5_file: h5py.File, n_patches: int) -> Optional[torch.Tensor]:
        """Load coords from same .h5 or from coords_dir."""
        if self.coords_key in h5_file:
            c = h5_file[self.coords_key][:]
            return torch.from_numpy(c).float()
        if self.coords_dir:
            for name in (f"{slide_id}.h5", f"{slide_id}_patches.h5"):
                path = self.coords_dir / name
                if path.exists():
                    with h5py.File(path, "r") as cf:
                        for k in ("coords", "coordinates", "patches"):
                            if k in cf and cf[k].shape[0] == n_patches:
                                return torch.from_numpy(cf[k][:]).float()
        return None

    def __getitem__(self, idx: int):
        slide_id = self.slide_ids[idx]
        h5_path = self.feat_dir / f"{slide_id}.h5"
        with h5py.File(h5_path, "r") as f:
            feats = torch.from_numpy(f[self.key][:]).float()
            coords = self._load_coords(slide_id, f, feats.size(0))

        cap = self.max_patches if self.training else self.max_patches_val
        if cap is not None and feats.size(0) > cap:
            n = feats.size(0)
            attn = self.attention_weights.get(slide_id) if self.training else None
            feats, coords = _subsample_patches(feats, coords, cap, attn)
            n = feats.size(0)
        else:
            n = feats.size(0)

        if self.training and n > 1 and self.patch_dropout > 0.0:
            attn = self.attention_weights.get(slide_id)
            keep = max(1, int(n * (1.0 - self.patch_dropout)))
            feats, coords = _subsample_patches(feats, coords, keep, attn)

        label = int(self.labels.loc[slide_id, "isup_grade"])
        return feats, coords, label, slide_id


def collate_bags(batch):
    """Pad variable-length bags to [B, N_max, D] with boolean mask.
    Returns (feats, mask, labels, ids, coords) where coords is [B,N_max,C] or None.
    """
    if not batch:
        raise ValueError("collate_bags received empty batch")

    feats_list = [b[0] for b in batch]
    coords_list = [b[1] for b in batch]
    labels = torch.tensor([b[2] for b in batch], dtype=torch.long)
    ids = [b[3] for b in batch]

    N_max = max(f.size(0) for f in feats_list)
    D = feats_list[0].size(1)
    feats = torch.zeros(len(batch), N_max, D, dtype=feats_list[0].dtype)
    mask = torch.zeros(len(batch), N_max, dtype=torch.bool)

    has_coords = all(c is not None for c in coords_list)
    coords = None
    if has_coords:
        C = coords_list[0].size(1)
        coords = torch.zeros(len(batch), N_max, C, dtype=coords_list[0].dtype)

    for i, f in enumerate(feats_list):
        n = f.size(0)
        feats[i, :n] = f
        mask[i, :n] = True
        if has_coords:
            coords[i, :n] = coords_list[i]

    return feats, mask, labels, ids, coords


# ---------------------------------------------------------------------------
# In-memory cached dataset (preload all .h5 to RAM for fast training)
# ---------------------------------------------------------------------------


def preload_features(
    feat_dir: Path,
    slide_ids: list[str],
    key: str = "features",
    coords_key: str | None = "coords",
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
    """Load all .h5 feature files into RAM.

    Returns ``(feat_cache, coord_cache)``. If ``coords_key`` is set and present in a file,
    ``coord_cache[sid]`` is ``[N, C]`` float32; otherwise ``None`` for that slide.
    Pass ``coords_key=None`` to skip coords (saves RAM).
    """
    feat_dir = Path(feat_dir)
    feat_cache: dict[str, torch.Tensor] = {}
    coord_cache: dict[str, torch.Tensor | None] = {}
    for i, sid in enumerate(slide_ids):
        h5_path = feat_dir / f"{sid}.h5"
        if h5_path.exists():
            with h5py.File(h5_path, "r") as f:
                feat_cache[sid] = torch.from_numpy(f[key][:]).float()
                if coords_key and coords_key in f:
                    coord_cache[sid] = torch.from_numpy(f[coords_key][:]).float()
                else:
                    coord_cache[sid] = None
        if (i + 1) % 1000 == 0:
            gb_feat = sum(v.nelement() * 4 for v in feat_cache.values()) / 1e9
            gb_c = sum(c.nelement() * 4 for c in coord_cache.values() if c is not None) / 1e9 if coord_cache else 0.0
            print(
                f"  Loaded {i + 1}/{len(slide_ids)} slides (feats ~{gb_feat:.1f} GB, coords ~{gb_c:.2f} GB)", flush=True
            )
    gb_feat = sum(v.nelement() * 4 for v in feat_cache.values()) / 1e9
    gb_c = sum(c.nelement() * 4 for c in coord_cache.values() if c is not None) / 1e9
    n_c = sum(1 for c in coord_cache.values() if c is not None)
    print(
        f"  Preloaded {len(feat_cache)} slides, feats ~{gb_feat:.1f} GB; "
        f"coords ~{gb_c:.2f} GB ({n_c} slides with coords)",
        flush=True,
    )
    return feat_cache, coord_cache


class CachedBagDataset(Dataset):
    """In-memory dataset — no disk IO during training.

    If ``coord_cache`` is provided (same keys as ``cache``), patch coordinates are subsampled
    with features so spatial layouts stay aligned (e.g. for visualization or future spatial MIL).

    Smart sampling (training only):
    - max_patches: cap bag size
    - patch_dropout: further random subsample
    - attention_weights: optional dict for biased sampling (keep high-attention)
    """

    def __init__(
        self,
        cache: dict[str, torch.Tensor],
        slide_ids: list[str],
        train_labels: pd.DataFrame,
        coord_cache: dict[str, torch.Tensor | None] | None = None,
        patch_dropout: float = 0.0,
        max_patches: Optional[int] = None,
        max_patches_val: Optional[int] = None,
        attention_weights: Optional[dict[str, torch.Tensor]] = None,
        training: bool = False,
    ):
        self.cache = cache
        self.coord_cache = coord_cache
        self.slide_ids = [sid for sid in slide_ids if sid in cache]
        self.labels = train_labels
        self.patch_dropout = patch_dropout
        self.max_patches = max_patches
        self.max_patches_val = max_patches_val
        self.attention_weights = attention_weights or {}
        self.training = training

    def __len__(self) -> int:
        return len(self.slide_ids)

    def __getitem__(self, idx: int):
        sid = self.slide_ids[idx]
        feats = self.cache[sid]
        coords = None
        if self.coord_cache is not None:
            coords = self.coord_cache.get(sid)

        cap = self.max_patches if self.training else self.max_patches_val
        if cap is not None and feats.size(0) > cap:
            n = feats.size(0)
            attn = self.attention_weights.get(sid) if self.training else None
            feats, coords = _subsample_patches(feats, coords, cap, attn)
            n = feats.size(0)
        else:
            n = feats.size(0)

        if self.training and n > 1 and self.patch_dropout > 0.0:
            attn = self.attention_weights.get(sid)
            keep = max(1, int(n * (1.0 - self.patch_dropout)))
            feats, coords = _subsample_patches(feats, coords, keep, attn)

        label = int(self.labels.loc[sid, "isup_grade"])
        return feats, coords, label, sid
