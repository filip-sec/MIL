"""Attention map visualization: overlay MIL attention weights on WSI thumbnail."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.gridspec import GridSpec

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import openslide

    HAS_OPENSLIDE = True
except ImportError:
    HAS_OPENSLIDE = False

try:
    from scipy.ndimage import gaussian_filter
    from scipy.stats import rankdata

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    rankdata = None  # type: ignore[misc, assignment]


def accumulate_attention_on_thumbnail_grid(
    x: np.ndarray,
    y: np.ndarray,
    att: np.ndarray,
    *,
    patch_size_l0: int,
    w0: int,
    h0: int,
    tw: int,
    th: int,
    overlay_agg: str = "max",
) -> np.ndarray:
    """Rasterize patch attention onto a thumbnail-sized grid (H, W) = (th, tw).

    Each patch is a **level-0 axis-aligned square** with top-left ``(x[i], y[i])`` and side
    ``patch_size_l0``. The rectangle is mapped to thumbnail pixels with ``floor``/``ceil`` so
    the full patch **area** contributes (not a single pixel).

    Args:
        x, y: top-left coordinates in level-0 slide pixels
        att: per-patch weights (same length as x)
        patch_size_l0: patch width/height in level-0 pixels
        w0, h0: slide level-0 dimensions
        tw, th: thumbnail width and height in pixels
        overlay_agg: ``\"max\"`` or ``\"sum\"`` where patches overlap on the grid

    Returns:
        ``(th, tw)`` float32 grid (same orientation as ``imshow`` with origin upper).
    """
    if overlay_agg not in ("max", "sum"):
        raise ValueError(f"overlay_agg must be 'max' or 'sum', got {overlay_agg!r}")
    psz = int(patch_size_l0)
    if psz < 1:
        raise ValueError("patch_size_l0 must be >= 1")

    grid = np.zeros((th, tw), dtype=np.float32)
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    att = np.asarray(att, dtype=np.float32).ravel()
    if len(x) != len(y) or len(x) != len(att):
        raise ValueError("x, y, att must have the same length")

    w0f, h0f = float(w0), float(h0)
    twf, thf = float(tw), float(th)

    for xi, yi, ai in zip(x, y, att):
        x1, y1 = xi + psz, yi + psz
        px0 = int(np.floor(xi / w0f * twf))
        px1 = int(np.ceil(x1 / w0f * twf))
        py0 = int(np.floor(yi / h0f * thf))
        py1 = int(np.ceil(y1 / h0f * thf))
        px0 = int(np.clip(px0, 0, tw))
        px1 = int(np.clip(px1, 0, tw))
        py0 = int(np.clip(py0, 0, th))
        py1 = int(np.clip(py1, 0, th))
        if px1 <= px0:
            px1 = min(tw, px0 + 1)
        if py1 <= py0:
            py1 = min(th, py0 + 1)
        if px1 <= px0 or py1 <= py0:
            continue
        block = grid[py0:py1, px0:px1]
        if overlay_agg == "max":
            grid[py0:py1, px0:px1] = np.maximum(block, ai)
        else:
            grid[py0:py1, px0:px1] = block + ai

    return grid


def patch_union_mask_on_thumbnail_grid(
    x: np.ndarray,
    y: np.ndarray,
    patch_size_l0: int,
    w0: int,
    h0: int,
    tw: int,
    th: int,
    *,
    dilate_iters: int = 0,
) -> np.ndarray:
    """Binary mask (float 0/1) covering all patch **squares** in level-0 space, rasterized like ``accumulate_attention_on_thumbnail_grid``.

    Multiplying a blurred heatmap by this mask **clips halo** outside the patch union (sharper edges).
    """
    psz = int(patch_size_l0)
    if psz < 1:
        raise ValueError("patch_size_l0 must be >= 1")
    mask = np.zeros((th, tw), dtype=np.float32)
    w0f, h0f = float(w0), float(h0)
    twf, thf = float(tw), float(th)
    xa = np.asarray(x, dtype=np.float64).ravel()
    ya = np.asarray(y, dtype=np.float64).ravel()
    for xi, yi in zip(xa, ya):
        x1, y1 = xi + psz, yi + psz
        px0 = int(np.floor(xi / w0f * twf))
        px1 = int(np.ceil(x1 / w0f * twf))
        py0 = int(np.floor(yi / h0f * thf))
        py1 = int(np.ceil(y1 / h0f * thf))
        px0 = int(np.clip(px0, 0, tw))
        px1 = int(np.clip(px1, 0, tw))
        py0 = int(np.clip(py0, 0, th))
        py1 = int(np.clip(py1, 0, th))
        if px1 <= px0:
            px1 = min(tw, px0 + 1)
        if py1 <= py0:
            py1 = min(th, py0 + 1)
        if px1 <= px0 or py1 <= py0:
            continue
        mask[py0:py1, px0:px1] = 1.0
    if dilate_iters > 0 and HAS_SCIPY:
        from scipy.ndimage import binary_dilation

        m = mask > 0.5
        foot = np.ones((3, 3), dtype=bool)
        for _ in range(int(dilate_iters)):
            m = binary_dilation(m, footprint=foot)
        mask = m.astype(np.float32)
    return mask


def _otsu_threshold_from_hist(hist: np.ndarray) -> int:
    h = hist.astype(np.float64)
    total = h.sum()
    if total < 1:
        return 127
    best_t, best_var = 0, -1.0
    idx = np.arange(256, dtype=np.float64)
    for t in range(256):
        w0 = h[: t + 1].sum()
        w1 = h[t + 1 :].sum()
        if w0 < 1.0 or w1 < 1.0:
            continue
        m0 = (h[: t + 1] * idx[: t + 1]).sum() / w0
        m1 = (h[t + 1 :] * idx[t + 1 :]).sum() / w1
        var = w0 * w1 * (m0 - m1) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return int(best_t)


def tissue_mask_from_brightness_rgb(
    rgb: np.ndarray,
    *,
    close_iters: int = 4,
    open_iters: int = 1,
) -> np.ndarray:
    """Coarse tissue foreground from H&E: Otsu on luminance + morphology (bright-glass background).

    Heuristic only; fails on uneven illumination or unusual staining. Prefer ``mask_mode='patches'`` for strict footprint clipping.
    """
    if not HAS_SCIPY:
        raise ImportError("tissue mask requires scipy (ndimage)")
    from scipy.ndimage import binary_closing, binary_opening

    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("rgb must be HxWx3")
    g = (
        0.299 * rgb[..., 0].astype(np.float64)
        + 0.587 * rgb[..., 1].astype(np.float64)
        + 0.114 * rgb[..., 2].astype(np.float64)
    )
    gray = np.clip(g, 0, 255).astype(np.uint8)
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    t = _otsu_threshold_from_hist(hist)
    mu = float(gray.mean())
    if mu > 160.0:
        fg = gray.astype(np.float32) < float(t)
    else:
        fg = gray.astype(np.float32) > float(t)
    m = fg.astype(bool)
    if open_iters > 0:
        m = binary_opening(m, iterations=int(open_iters))
    if close_iters > 0:
        m = binary_closing(m, iterations=int(close_iters))
    return m.astype(np.float32)


def infer_patch_size_l0_from_coords(coord: np.ndarray) -> int | None:
    """Heuristic square patch size in L0 px from a coordinate grid (fallback if attrs missing)."""
    if coord is None or len(coord) < 2:
        return None
    xs = np.sort(np.unique(coord[:, 0].astype(np.float64)))
    ys = np.sort(np.unique(coord[:, 1].astype(np.float64)))
    dx = float(np.median(np.diff(xs))) if len(xs) >= 2 else None
    dy = float(np.median(np.diff(ys))) if len(ys) >= 2 else None
    if dx is None and dy is None:
        return None
    if dx is None:
        step = dy
    elif dy is None:
        step = dx
    else:
        step = min(dx, dy)
    if step <= 0 or not np.isfinite(step):
        return None
    return max(1, int(round(step)))


def postprocess_attention_grid(
    grid: np.ndarray,
    tw: int,
    th: int,
    *,
    gaussian_sigma: float | None = None,
    percentile_low: float = 5.0,
    percentile_high: float = 99.0,
    use_percentile_rank: bool = False,
) -> np.ndarray:
    """Blur + contrast scaling for rasterized attention (CLAM-style ``drawHeatmap`` conventions).

    Args:
        grid: 2D non-negative scores (e.g. max attention per thumbnail pixel).
        tw, th: grid width/height (used for default Gaussian sigma).
        gaussian_sigma: If ``None``, ``sigma = max(1, min(tw,th) / 128)``. Use ``0`` to skip blur.
        percentile_low, percentile_high: Inliers for robust min/max when ``use_percentile_rank`` is False.
        use_percentile_rank: If True, map positive pixels to normalized ranks (spread colormap).
    """
    grid = np.asarray(grid, dtype=np.float32)
    if HAS_SCIPY and np.any(grid > 0) and gaussian_sigma != 0.0:
        sigma = float(gaussian_sigma) if gaussian_sigma is not None else max(1.0, min(tw, th) / 128.0)
        if sigma > 0:
            grid = gaussian_filter(grid, sigma=sigma)

    if use_percentile_rank:
        if not HAS_SCIPY or rankdata is None:
            raise ImportError("percentile-rank smoothing requires scipy")
        flat = grid.ravel().astype(np.float64)
        pos = flat > 0
        out = np.zeros_like(flat, dtype=np.float32)
        if np.any(pos):
            r = rankdata(flat[pos], method="average").astype(np.float64)
            out[pos] = ((r - r.min()) / (r.max() - r.min() + 1e-9)).astype(np.float32)
        return out.reshape(grid.shape)

    grid_vis = grid.astype(np.float32)
    if np.any(grid_vis > 0):
        lo, hi = np.percentile(grid_vis[grid_vis > 0], [percentile_low, percentile_high])
        grid_vis = np.clip((grid_vis - lo) / (hi - lo + 1e-9), 0.0, 1.0)
    return grid_vis


def choose_vis_level_for_max_side(slide: object, max_side: int) -> int:
    """Pick a pyramid level whose width/height does not exceed ``max_side`` too much."""
    w0, h0 = slide.level_dimensions[0]
    need = max(w0, h0) / float(max(512, max_side))
    if need <= 1.0:
        return 0
    return int(slide.get_best_level_for_downsample(need))


def blend_heatmap_on_rgb(
    background_rgb: np.ndarray,
    heat_norm: np.ndarray,
    cmap_name: str = "turbo",
    alpha: float = 0.45,
) -> np.ndarray:
    """Alpha-blend colormap(heat) onto ``background_rgb`` (H,W,3 uint8) → uint8 RGB."""
    if not HAS_MPL:
        raise ImportError("matplotlib required")
    bg = np.clip(background_rgb.astype(np.float32), 0, 255)
    hmap = np.clip(heat_norm.astype(np.float32), 0.0, 1.0)
    if hasattr(matplotlib, "colormaps"):
        cmap = matplotlib.colormaps[cmap_name]
    else:
        cmap = matplotlib.cm.get_cmap(cmap_name)
    colored = cmap(hmap)[..., :3].astype(np.float32) * 255.0
    a = float(np.clip(alpha, 0.0, 1.0))
    out = bg * (1.0 - a) + colored * a
    return np.clip(out, 0, 255).astype(np.uint8)


def save_fullslide_attention_raster(
    wsi_path: Path | str,
    attention: torch.Tensor | np.ndarray,
    coords: np.ndarray,
    patch_size_level0: int,
    out_path: Path | str,
    *,
    vis_level: int | None = None,
    max_side: int = 4096,
    overlay_agg: str = "max",
    cmap: str = "turbo",
    alpha: float = 0.45,
    gaussian_sigma: float | None = None,
    percentile_low: float = 5.0,
    percentile_high: float = 99.5,
    use_percentile_rank: bool = False,
    normalize_attention: bool = True,
    mask_mode: str = "none",
    patch_mask_dilate: int = 0,
) -> tuple[int, int, int]:
    """CLAM-style full-slide heatmap: read WSI at a pyramid level, rasterize attention, blur, save TIFF/PNG.

    Coordinates are **level-0** top-left patch corners; ``patch_size_level0`` is the L0 square side.
    The accumulation grid matches the chosen level dimensions (same aspect as ``read_region(..., level)``).

    ``mask_mode``: ``none`` | ``patches`` (clip to patch footprint union) | ``tissue`` (Otsu luminance + morphology).

    Returns:
        ``(vis_level, height, width)`` of the written image.
    """
    if not HAS_OPENSLIDE:
        raise ImportError("openslide required for full-slide raster")
    try:
        from PIL import Image
    except ImportError as e:
        raise ImportError("Pillow required to save raster; pip install pillow") from e

    wsi_path = Path(wsi_path)
    if not wsi_path.exists():
        alt = wsi_path.with_suffix(".tif") if wsi_path.suffix == ".tiff" else wsi_path.with_suffix(".tiff")
        if alt.exists():
            wsi_path = alt

    att = np.asarray(attention, dtype=np.float32).ravel()
    if normalize_attention:
        att = att / (att.sum() + 1e-9)
    coord = np.asarray(coords)
    x, y = coord[:, 0], coord[:, 1]
    psz = int(patch_size_level0)
    if psz < 1:
        raise ValueError("patch_size_level0 must be >= 1")

    slide = openslide.OpenSlide(str(wsi_path))
    try:
        w0, h0 = slide.level_dimensions[0]
        level = int(vis_level) if vis_level is not None else choose_vis_level_for_max_side(slide, max_side)
        level = max(0, min(level, len(slide.level_dimensions) - 1))
        w_l, h_l = slide.level_dimensions[level]

        thumb = slide.read_region((0, 0), level, (w_l, h_l))
        bg = np.array(thumb.convert("RGB"), dtype=np.uint8)

        grid = accumulate_attention_on_thumbnail_grid(
            x,
            y,
            att,
            patch_size_l0=psz,
            w0=int(w0),
            h0=int(h0),
            tw=int(w_l),
            th=int(h_l),
            overlay_agg=overlay_agg,
        )
        grid_vis = postprocess_attention_grid(
            grid,
            int(w_l),
            int(h_l),
            gaussian_sigma=gaussian_sigma,
            percentile_low=percentile_low,
            percentile_high=percentile_high,
            use_percentile_rank=use_percentile_rank,
        )
        mm = (mask_mode or "none").strip().lower()
        if mm == "patches":
            pm = patch_union_mask_on_thumbnail_grid(
                x,
                y,
                psz,
                int(w0),
                int(h0),
                int(w_l),
                int(h_l),
                dilate_iters=patch_mask_dilate,
            )
            grid_vis = grid_vis * pm
        elif mm == "tissue":
            tm = tissue_mask_from_brightness_rgb(bg)
            grid_vis = grid_vis * tm
        elif mm != "none":
            raise ValueError("mask_mode must be none, patches, or tissue")
        composite = blend_heatmap_on_rgb(bg, grid_vis, cmap_name=cmap, alpha=alpha)

        out_p = Path(out_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        if out_p.suffix.lower() not in (".png", ".tif", ".tiff"):
            out_p = out_p.with_suffix(".png")
        img = Image.fromarray(composite)
        if out_p.suffix.lower() in (".tif", ".tiff"):
            img.save(out_p, compression="tiff_deflate")
        else:
            img.save(out_p)

        return level, int(h_l), int(w_l)
    finally:
        slide.close()


def plot_attention_map(
    attention: torch.Tensor | np.ndarray,
    coords: Optional[torch.Tensor | np.ndarray] = None,
    wsi_path: Optional[Path | str] = None,
    slide_dimensions: Optional[tuple[int, int]] = None,
    alpha: float = 0.5,
    cmap: str = "turbo",
    thumbnail_size: int = 512,
    ax=None,
    topk_idx: Optional[torch.Tensor | np.ndarray] = None,
    patch_size_level0: Optional[int] = None,
    overlay_agg: str = "max",
    gaussian_sigma: float | None = None,
    percentile_low: float = 5.0,
    percentile_high: float = 99.0,
    use_percentile_rank: bool = False,
    mask_mode: str = "none",
    patch_mask_dilate: int = 0,
) -> "matplotlib.axes.Axes":
    """Overlay attention weights on WSI thumbnail as heatmap.

    Args:
        attention: [N] attention weights per patch (will be normalized for display)
        coords: [N, 2+] patch coordinates (x, y) in level-0 pixels; if None, uses 1D bar plot
        wsi_path: Path to .tiff WSI for thumbnail background
        slide_dimensions: (width, height) of WSI at level 0 if no wsi_path
        alpha: Transparency of attention overlay (0=invisible, 1=opaque)
        cmap: Matplotlib colormap name
        thumbnail_size: Max size for thumbnail
        ax: Matplotlib axes to draw on (creates new if None)
        topk_idx: [k] indices (OrdinalMIL); optional, not overlaid on the map
        patch_size_level0: L0 patch footprint in pixels for **area** heatmap (TRIDENT top-left
            + square side). If None with coords, a median grid step is inferred from coords.
        overlay_agg: ``\"max\"`` or ``\"sum\"`` for overlapping patches on the thumbnail grid
        gaussian_sigma: Blur sigma after rasterizing (``None`` = auto). ``0`` skips blur.
        percentile_low, percentile_high: Robust contrast stretch (ignored if ``use_percentile_rank``).
        use_percentile_rank: Rank-normalize positive pixels (spread colormap; needs scipy).
        mask_mode: ``none`` | ``patches`` | ``tissue`` — clip heatmap outside patch union or coarse tissue.
        patch_mask_dilate: Binary dilation iterations on patch-union mask (small halo inside mask).

    Returns:
        Matplotlib axes
    """
    if not HAS_MPL:
        raise ImportError("matplotlib required for plot_attention_map")

    att = np.asarray(attention).ravel().astype(np.float32)
    att = att / (att.sum() + 1e-9)  # normalize for display

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if coords is not None:
        coord = np.asarray(coords)
        x, y = coord[:, 0], coord[:, 1]

        thumb_arr = None
        w = h = None
        if wsi_path and HAS_OPENSLIDE:
            path = Path(wsi_path)
            if path.exists():
                slide = openslide.OpenSlide(str(path))
                try:
                    thumb = slide.get_thumbnail((thumbnail_size, thumbnail_size))
                    w, h = slide.level_dimensions[0]
                    thumb_arr = np.array(thumb)
                    ax.imshow(thumb_arr, extent=[0, w, h, 0], origin="upper")
                finally:
                    slide.close()
        elif slide_dimensions:
            w, h = slide_dimensions
            ax.set_xlim(0, w)
            ax.set_ylim(h, 0)
            ax.set_aspect("equal")

        psz = patch_size_level0
        if psz is None:
            psz = infer_patch_size_l0_from_coords(coord)

        use_area = thumb_arr is not None and w is not None and h is not None and psz is not None

        if use_area:
            th, tw = thumb_arr.shape[0], thumb_arr.shape[1]
            grid = accumulate_attention_on_thumbnail_grid(
                x,
                y,
                att,
                patch_size_l0=int(psz),
                w0=int(w),
                h0=int(h),
                tw=int(tw),
                th=int(th),
                overlay_agg=overlay_agg,
            )
            grid_vis = postprocess_attention_grid(
                grid,
                int(tw),
                int(th),
                gaussian_sigma=gaussian_sigma,
                percentile_low=percentile_low,
                percentile_high=percentile_high,
                use_percentile_rank=use_percentile_rank,
            )
            mm = (mask_mode or "none").strip().lower()
            if mm == "patches" and psz is not None:
                pm = patch_union_mask_on_thumbnail_grid(
                    x, y, int(psz), int(w), int(h), int(tw), int(th), dilate_iters=patch_mask_dilate
                )
                grid_vis = grid_vis * pm
            elif mm == "tissue" and thumb_arr is not None:
                tm = tissue_mask_from_brightness_rgb(thumb_arr[:, :, :3])
                grid_vis = grid_vis * tm
            elif mm not in ("none", "patches", "tissue"):
                raise ValueError("mask_mode must be none, patches, or tissue")
            im = ax.imshow(
                grid_vis,
                extent=[0, w, h, 0],
                origin="upper",
                cmap=cmap,
                alpha=alpha,
                vmin=0,
                vmax=1,
            )
            plt.colorbar(im, ax=ax, label="attention (normalized)")
        else:
            scatter = ax.scatter(x, y, c=att, s=15, cmap=cmap, alpha=alpha)
            plt.colorbar(scatter, ax=ax, label="attention")
    else:
        # No coords: simple bar plot
        cm = matplotlib.cm.get_cmap(cmap)
        colors = cm(np.clip(att / (att.max() + 1e-9), 0, 1))
        ax.bar(np.arange(len(att)), att, color=colors)
        ax.set_xlabel("patch index")
        ax.set_ylabel("attention")

    return ax


def get_attention_for_slide(
    model: torch.nn.Module,
    feats: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    return_topk: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run model and return attention weights for one slide.

    Args:
        model: MIL model (AttentionMIL, GatedAttentionMIL, or OrdinalMIL)
        feats: [1, N, D] or [N, D] patch features
        mask: [1, N] boolean mask (optional)
        device: Device to run on
        return_topk: If True and model returns topk_idx (OrdinalMIL), return (att, topk_idx)

    Returns:
        [N] attention weights (sum to 1), or (att, topk_idx) if return_topk and available
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    if feats.dim() == 2:
        feats = feats.unsqueeze(0)
    if mask is not None and mask.dim() == 1:
        mask = mask.unsqueeze(0)

    feats = feats.to(device)
    if mask is not None:
        mask = mask.to(device)

    with torch.no_grad():
        out = model(feats, mask, return_attention=True)
        if isinstance(out, tuple):
            _, att = out
            topk_idx = None
        elif isinstance(out, dict):
            att = out["attention"]
            topk_idx = out.get("topk_idx")  # [1, k_max] for OrdinalMIL
            if topk_idx is not None:
                topk_idx = topk_idx[0].cpu()  # [k_max]
        else:
            raise ValueError("Model must support return_attention=True")

    att = att[0].cpu()  # first (only) slide
    if mask is not None:
        att = att.masked_fill(~mask[0].cpu(), 0.0)

    if return_topk and topk_idx is not None:
        return att, topk_idx
    return att


def get_attention_and_logits_for_slide(
    model: torch.nn.Module,
    feats: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run MIL/CLAM on one slide; return (attention [N], bag_logits [C]) on CPU.

    ``bag_logits`` are raw classifier outputs before softmax (same as training CE head).
    """
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    if feats.dim() == 2:
        feats = feats.unsqueeze(0)
    if mask is not None and mask.dim() == 1:
        mask = mask.unsqueeze(0)

    feats = feats.to(device)
    if mask is not None:
        mask = mask.to(device)

    with torch.no_grad():
        out = model(feats, mask, return_attention=True)
        if isinstance(out, tuple):
            logits_bag, att = out[0], out[1]
            logits_vec = logits_bag[0].detach().float().cpu()
        elif isinstance(out, dict):
            att = out["attention"]
            if "ce_logits" in out:
                logits_vec = out["ce_logits"][0].detach().float().cpu()
            elif "ordinal_logits" in out:
                logits_vec = out["ordinal_logits"][0].detach().float().cpu()
            else:
                raise ValueError("Model dict output missing ce_logits / ordinal_logits")
        else:
            raise ValueError("Model must support return_attention=True")

    att = att[0].cpu().float()
    if mask is not None:
        att = att.masked_fill(~mask[0].cpu(), 0.0)
    return att, logits_vec


def save_attention_blockmap_h5(
    path: Path | str,
    attention: torch.Tensor | np.ndarray,
    coords: np.ndarray,
) -> None:
    """Write CLAM-style blockmap HDF5: ``attention_scores`` [N,1], ``coords`` [N, 2+]."""
    import h5py

    att = np.asarray(attention, dtype=np.float32).reshape(-1, 1)
    xy = np.asarray(coords)
    if xy.shape[0] != att.shape[0]:
        raise ValueError(f"attention length {att.shape[0]} != coords rows {xy.shape[0]}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("attention_scores", data=att)
        f.create_dataset("coords", data=xy)


def read_trident_coord_attrs_from_h5(h5_path: Path | str) -> dict:
    """Read TRIDENT-style attributes from the `coords` dataset in a feature .h5 file.

    TRIDENT (mahmoodlab/TRIDENT) stores patch coords as **top-left (x, y) in level-0 pixels**
    (OpenSlide reference). See ``WSIPatcher._colrow_to_xy`` and ``get_tile_xy``.

    Relevant attrs often include:
    - ``patch_size_level0``: patch width/height in level-0 pixels (use for ``read_region`` at level 0)
    - ``patch_size``: patch size at target magnification (e.g. 224)
    - ``level0_magnification``, ``target_magnification``
    """
    import h5py

    out: dict = {}
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            return out
        d = f["coords"]
        for k in d.attrs:
            v = d.attrs[k]
            if isinstance(v, bytes):
                v = v.decode("utf-8", errors="replace")
            out[str(k)] = v
    return out


def infer_patch_size_level0_trusted(attrs: dict, fallback_encoder_patch: int) -> tuple[int, bool]:
    """Return (level0_patch_px, trusted).

    ``trusted`` is False only when falling back to encoder patch size (224/256) — not a true
    level-0 footprint; do not use for final figures without ``--allow-l0-fallback``.
    """
    if "patch_size_level0" in attrs:
        return int(np.asarray(attrs["patch_size_level0"]).item()), True
    ps = attrs.get("patch_size")
    l0 = attrs.get("level0_magnification")
    tgt = attrs.get("target_magnification")
    if ps is not None and l0 is not None and tgt is not None:
        try:
            return int(int(ps) * int(l0) // int(tgt)), True
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return int(fallback_encoder_patch), False


def infer_patch_size_level0(attrs: dict, fallback_encoder_patch: int) -> int:
    """Level-0 patch extent in pixels for OpenSlide read_region at level 0."""
    s, _ = infer_patch_size_level0_trusted(attrs, fallback_encoder_patch)
    return s


def plot_attention_figure_publishable(
    attention: torch.Tensor | np.ndarray,
    coords: np.ndarray,
    topk_idx: Optional[torch.Tensor | np.ndarray],
    wsi_path: Optional[Path | str],
    slide_id: str,
    true_label: int,
    pred_label: int,
    correct: bool,
    thumbnail_size: int = 512,
    n_patch_crops: int = 8,
    patch_size: int = 224,
    patch_size_level0: Optional[int] = None,
    overlay_agg: str = "max",
    allow_l0_size_fallback: bool = False,
    figsize: tuple[float, float] = (14, 5),
    dpi: int = 150,
    colorbar_fraction: float = 0.035,
    colorbar_pad: float = 0.02,
    wsi_geometry_seg_level: int = -1,
    annotate_wsi_geometry: bool = True,
    wsi_geometry_lang: str = "en",
) -> "matplotlib.figure.Figure":
    """Create 3-panel publication figure: A=thumbnail, B=attention overlay, C=top-k patch crops.

    **Coordinate semantics (TRIDENT / OpenSlide):** each row of ``coords`` is the **top-left**
    corner of the patch in **level-0** slide pixels, not the center. Crops use
    ``read_region((x, y), 0, (patch_size_level0, patch_size_level0))``.

    **``patch_size_level0``** must be the true L0 footprint (HDF5 ``patch_size_level0`` or trusted
    formula). If ``None`` and ``allow_l0_size_fallback`` is False, raises; if True, falls back to
    ``patch_size`` (debug only).

    **Attention (panel B):** each patch paints a **rectangle** on the thumbnail grid by mapping the
    level-0 square ``[x, x+psz] × [y, y+psz]`` with ``floor``/``ceil`` (not a single pixel).

    **Attention:** overlay uses raw softmax weights; contrast via percentile norm on the grid.
    Panel C titles show the same raw scores. ``overlay_agg`` is ``"max"`` or ``"sum"`` for overlaps.

    **WSI reporting:** if ``annotate_wsi_geometry`` is True, the figure footer includes the standard
    four lines (level 0 size, segmentation level ``L_s`` and size, ``level_downsamples[L_s]``, MPP)
    from :mod:`mil.wsi_metadata`. Set ``wsi_geometry_seg_level`` to the pyramid index your
    segmentation pipeline uses (``-1`` = coarsest level).
    """
    if not HAS_MPL:
        raise ImportError("matplotlib required")
    if not HAS_OPENSLIDE or wsi_path is None or not Path(wsi_path).exists():
        raise ValueError("OpenSlide and WSI path required for publishable figure")

    if overlay_agg not in ("max", "sum"):
        raise ValueError(f"overlay_agg must be 'max' or 'sum', got {overlay_agg!r}")

    att_raw = np.asarray(attention).ravel().astype(np.float32)
    coord = np.asarray(coords)
    x, y = coord[:, 0], coord[:, 1]
    wsi_path = Path(wsi_path)

    # Resolve .tiff / .tif
    if not wsi_path.exists():
        alt = wsi_path.with_suffix(".tif") if wsi_path.suffix == ".tiff" else wsi_path.with_suffix(".tiff")
        if alt.exists():
            wsi_path = alt

    slide = openslide.OpenSlide(str(wsi_path))
    try:
        w0, h0 = slide.level_dimensions[0]
        if patch_size_level0 is None:
            if not allow_l0_size_fallback:
                raise ValueError(
                    "patch_size_level0 must be set (from HDF5 coords attrs) for publishable figures; "
                    "or pass allow_l0_size_fallback=True for debug-only encoder-size fallback."
                )
            psz = int(patch_size)
        else:
            psz = int(patch_size_level0)

        thumb = slide.get_thumbnail((thumbnail_size, thumbnail_size))
        thumb_arr = np.array(thumb)
        tw, th = thumb_arr.shape[1], thumb_arr.shape[0]

        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = GridSpec(1, 3, width_ratios=[1, 1, 1.2], figure=fig)

        # Panel A: thumbnail only
        ax_a = fig.add_subplot(gs[0])
        ax_a.imshow(thumb_arr, extent=[0, w0, h0, 0], origin="upper")
        ax_a.set_title("(A) WSI thumbnail")
        ax_a.axis("off")

        # Panel B: thumbnail + patch-area heatmap (top-left coords)
        ax_b = fig.add_subplot(gs[1])
        ax_b.imshow(thumb_arr, extent=[0, w0, h0, 0], origin="upper")

        grid = np.zeros((th, tw), dtype=np.float32)
        if len(x) > 0:
            grid = accumulate_attention_on_thumbnail_grid(
                x,
                y,
                att_raw,
                patch_size_l0=psz,
                w0=int(w0),
                h0=int(h0),
                tw=int(tw),
                th=int(th),
                overlay_agg=overlay_agg,
            )

            if HAS_SCIPY:
                grid = gaussian_filter(grid, sigma=max(1.0, min(tw, th) / 128.0))

            if np.any(grid > 0):
                lo, hi = np.percentile(grid[grid > 0], [5, 99])
                grid = np.clip((grid - lo) / (hi - lo + 1e-9), 0, 1)

            im = ax_b.imshow(
                grid,
                extent=[0, w0, h0, 0],
                origin="upper",
                cmap="turbo",
                alpha=0.45,
                vmin=0,
                vmax=1,
            )
            cbar = fig.colorbar(im, ax=ax_b, fraction=colorbar_fraction, pad=colorbar_pad)
            cbar.set_label("Normalized attention")

        ax_b.set_title("(B) Attention overlay")
        ax_b.axis("off")

        # Panel C: top-k patch crops (grid)
        if topk_idx is not None and n_patch_crops > 0:
            idx = np.asarray(topk_idx).ravel()
            idx = idx[idx < len(x)][:n_patch_crops]
            n_show = len(idx)
            if n_show > 0:
                # Prefer 3 columns when showing ≤9 crops (paper layout); wider grid if more
                ncols = min(3, n_show) if n_show <= 9 else min(4, n_show)
                nrows = (n_show + ncols - 1) // ncols
                gs_c = gs[2].subgridspec(nrows, ncols)
                for i, patch_i in enumerate(idx):
                    row, col = i // ncols, i % ncols
                    sub = fig.add_subplot(gs_c[row, col])
                    px, py = int(x[patch_i]), int(y[patch_i])
                    loc_x = max(0, min(px, w0 - 1))
                    loc_y = max(0, min(py, h0 - 1))
                    rw = min(psz, w0 - loc_x)
                    rh = min(psz, h0 - loc_y)
                    patch = slide.read_region((loc_x, loc_y), 0, (rw, rh))
                    patch_arr = np.array(patch)[:, :, :3]
                    sub.imshow(patch_arr)
                    ar = float(att_raw[patch_i])
                    sub.set_title(f"#{i + 1}  a={ar:.4g}", fontsize=8)
                    sub.axis("off")
                fig.text(0.83, 0.96, "(C) Top-k patches", ha="center", fontsize=10)
            else:
                ax_c = fig.add_subplot(gs[2])
                ax_c.text(0.5, 0.5, "No top-k patches", ha="center", va="center", transform=ax_c.transAxes)
                ax_c.axis("off")
        else:
            ax_c = fig.add_subplot(gs[2])
            ax_c.text(0.5, 0.5, "No top-k patches", ha="center", va="center", transform=ax_c.transAxes)
            ax_c.axis("off")

        if correct:
            if true_label == 0:
                title = "Correctly classified low-grade slide (ISUP 0)"
            else:
                title = f"Correctly classified slide (ISUP {true_label})"
        else:
            if pred_label < true_label:
                title = f"Misclassified slide: underprediction (true ISUP {true_label}, predicted {pred_label})"
            elif pred_label > true_label:
                title = f"Misclassified slide: overprediction (true ISUP {true_label}, predicted {pred_label})"
            else:
                title = f"Misclassified slide (true ISUP {true_label}, predicted {pred_label})"

        fig.suptitle(title, fontsize=11, y=0.99)

        footer_lines: list[str] = []
        if annotate_wsi_geometry:
            try:
                from mil.wsi_metadata import Language, format_wsi_geometry_lines, wsi_geometry_from_slide

                lang: Language = "sk" if wsi_geometry_lang == "sk" else "en"
                g = wsi_geometry_from_slide(slide, seg_level=wsi_geometry_seg_level)
                footer_lines.append(format_wsi_geometry_lines(g, lang=lang))
            except Exception:
                pass
        footer_lines.append(f"Slide ID: {slide_id}")
        fig.text(
            0.5,
            0.004,
            "\n".join(footer_lines),
            ha="center",
            fontsize=6,
            family="monospace",
            transform=fig.transFigure,
            va="bottom",
            linespacing=1.12,
            style="normal",
        )
        fig.subplots_adjust(bottom=0.18, top=0.90)
    finally:
        slide.close()

    return fig
