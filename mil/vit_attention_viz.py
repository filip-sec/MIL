"""Plot ViT-style attention matrices (CLS → patches) per layer and head.

Expects tensors shaped **(B, num_heads, seq, seq)** (attention probabilities or pre-dropout
weights). With timm's default fused SDPA, ``attn_drop`` is **not executed**; disable fused attention
(``TIMM_FUSED_ATTN=0`` / ``timm.layers.set_fused_attn(False)``) before building the model
so ``attn_drop`` receives **(B, num_heads, N, N)** attention probabilities.

``out`` keys are typically ``named_modules`` paths (e.g. ``blocks.0.attn...``); they are sorted
by block index when possible.
"""
from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np

StyleName = Literal["publication", "compact"]
LayoutName = Literal["all_layers", "slide_heads"]


def _layer_index_from_key(k: str) -> int | None:
    m = re.search(r"blocks\.(\d+)", k)
    return int(m.group(1)) if m else None


def _publication_rc() -> dict[str, Any]:
    """Matplotlib rcParams for print / slides (sans-serif, neutral background)."""
    return {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica", "Liberation Sans"],
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#333333",
        "axes.titlepad": 6,
        "axes.labelpad": 4,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "figure.constrained_layout.use": False,
    }


def tensor_chw_to_display_hwc(chw: Any) -> np.ndarray:
    """(3,H,W) tensor/array, ImageNet-normalized → HWC RGB float in ``[0, 1]`` for ``imshow``."""
    chw = _to_numpy_any(chw)
    if chw.ndim != 3 or chw.shape[0] != 3:
        raise ValueError(f"expected (3,H,W), got {chw.shape}")
    return imagenet_denormalize(chw).transpose(1, 2, 0)


def _to_numpy_any(x: Any) -> np.ndarray:
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def imagenet_denormalize(chw: np.ndarray) -> np.ndarray:
    """CHW float image; reverse default ImageNet normalization, clip to [0,1]."""
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    x = chw * std + mean
    return np.clip(x, 0.0, 1.0)


def _sort_layer_keys(keys: list[str]) -> list[str]:
    def layer_index(k: str) -> tuple:
        m = re.search(r"blocks\.(\d+)", k)
        return (int(m.group(1)) if m else 9999, k)

    return sorted(keys, key=layer_index)


def _to_numpy(x: Any) -> np.ndarray:
    return _to_numpy_any(x)


def _default_patch_slice(seq_len: int, cls_index: int) -> tuple[slice, bool]:
    """Infer key-token slice for CLS→patch heatmaps.

    If all tokens after ``cls_index`` form an ``n×n`` grid, use ``slice(cls_index + 1, seq_len)``
    and return ``(slice, False)``.
    Otherwise assume **timm-style** layout: a short prefix after CLS (register / extra tokens),
    then ``n×n`` patch tokens occupying the **last** ``n²`` positions (``n = ⌊√L⌋``, ``L`` =
    number of keys after CLS). E.g. Virchow2: 261 = 1 CLS + 4 reg + 256 patches. Returns
    ``(slice, True)`` for that fallback.
    """
    if not (0 <= cls_index < seq_len):
        raise ValueError(f"cls_index={cls_index} invalid for seq_len={seq_len}")
    n_after_cls = seq_len - cls_index - 1
    if n_after_cls <= 0:
        raise ValueError(f"no key positions after cls_index={cls_index} (seq_len={seq_len})")
    side = math.isqrt(n_after_cls)
    patch_len = side * side
    if patch_len == n_after_cls:
        return slice(cls_index + 1, seq_len), False
    if patch_len < 1:
        raise ValueError(f"cannot infer patch grid from seq_len={seq_len}, cls_index={cls_index}")
    return slice(seq_len - patch_len, seq_len), True


def _find_key_for_block_index(keys_sorted: list[str], block_index: int) -> str:
    for k in keys_sorted:
        if _layer_index_from_key(k) == block_index:
            return k
    avail = sorted(
        {li for k in keys_sorted for li in [_layer_index_from_key(k)] if li is not None}
    )
    raise ValueError(
        f"No layer blocks.{block_index} in attention dict. Available block indices: {avail[:32]}"
        + (f" … (+{len(avail) - 32} more)" if len(avail) > 32 else "")
    )


def _plot_attention_slide_heads_column(
    out: dict[str, Any],
    img_chw: Any,
    save_path: str | Path | None,
    *,
    paired_layer: int,
    cls_index: int = 0,
    patch_slice: slice | None = None,
    style: StyleName = "publication",
    suptitle: str | None = None,
    cmap: str = "cividis",
    dpi: int = 300,
    show: bool = False,
    percentile_clip: tuple[float, float] | None = None,
    show_colorbar: bool | None = None,
    interpolation: str = "nearest",
) -> None:
    """One column: reference slide (full height) | stacked head heatmaps for ``blocks.{paired_layer}``."""
    import matplotlib.pyplot as plt
    import torch
    from matplotlib.gridspec import GridSpec

    if save_path is None and not show:
        raise ValueError("Set save_path and/or show=True")
    if img_chw is None:
        raise ValueError("slide_heads layout requires img_chw (e.g. --image tile)")

    if show_colorbar is None:
        show_colorbar = style == "publication"

    keys = _sort_layer_keys(list(out.keys()))
    k = _find_key_for_block_index(keys, int(paired_layer))
    v = out[k]
    if isinstance(v, torch.Tensor):
        t = v.detach().float().cpu()
    else:
        t = torch.as_tensor(v).float()
    if t.dim() == 4:
        t = t.squeeze(0)
    if t.dim() != 3:
        raise ValueError(f"{k}: expected (H,S,S), got {tuple(t.shape)}")
    n_heads, s, s2 = t.shape
    if s != s2:
        raise ValueError(f"{k}: expected square attention, got {(s, s2)}")

    seq_len = t.shape[-1]
    if patch_slice is not None:
        sl = patch_slice
    else:
        sl, used_last_n2 = _default_patch_slice(seq_len, cls_index)
        if used_last_n2:
            warnings.warn(
                f"Inferred patch key slice {sl} for seq_len={seq_len}; override patch_slice= if wrong.",
                UserWarning,
                stacklevel=3,
            )

    mats: list[np.ndarray] = []
    for head in range(n_heads):
        vec = t[head][cls_index, sl].numpy()
        n = vec.size
        side = int(round(n**0.5))
        if side * side != n:
            raise ValueError(
                f"{k} head {head}: {n} tokens in slice {sl} is not square (sqrt={side}). "
                "Set patch_slice (e.g. --patch-start / --patch-len)."
            )
        mats.append(vec.reshape(side, side).astype(np.float32))

    vmin, vmax = 0.0, 1.0
    if percentile_clip is not None:
        lo_p, hi_p = percentile_clip
        flat = np.concatenate([m.ravel() for m in mats])
        vmin, vmax = float(np.percentile(flat, lo_p)), float(np.percentile(flat, hi_p))
        if vmin >= vmax:
            vmax = vmin + 1e-6

    chw = _to_numpy(img_chw)
    if chw.ndim != 3 or chw.shape[0] != 3:
        raise ValueError(f"img_chw must be (3,H,W), got {chw.shape}")
    img_np = imagenet_denormalize(chw).transpose(1, 2, 0)

    row_h = max(1.1, 1.6 * min(n_heads, 8) / max(n_heads, 1))
    fig_w = 10.0 if style == "publication" else 8.0
    fig_h = row_h * n_heads
    rc = _publication_rc() if style == "publication" else {}

    with plt.rc_context(rc=rc):
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        gs = GridSpec(n_heads, 2, figure=fig, width_ratios=[1.2, 1.0], wspace=0.12, hspace=0.25)

        ax_slide = fig.add_subplot(gs[:, 0])
        ax_slide.imshow(np.clip(img_np, 0, 1), interpolation=interpolation)
        ax_slide.set_title(
            "Slide / tile" if style == "publication" else "input",
            fontsize=11 if style == "publication" else 9,
            fontweight="medium",
            color="#222222",
        )
        ax_slide.axis("off")
        lyr = _layer_index_from_key(k)
        ax_slide.text(
            0.02,
            0.98,
            f"Layer {lyr}" if lyr is not None else k[:40],
            transform=ax_slide.transAxes,
            fontsize=9,
            fontweight="medium",
            color="#333333",
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc", alpha=0.92),
        )

        im_last = None
        head_axes: list = []
        for h in range(n_heads):
            ax = fig.add_subplot(gs[h, 1])
            head_axes.append(ax)
            im = ax.imshow(
                mats[h],
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation=interpolation,
                aspect="equal",
            )
            im_last = im
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_edgecolor("#dddddd")
            ax.set_ylabel(f"Head {h}", fontsize=9, fontweight="medium", color="#333333", rotation=90, va="center")
            if h == 0 and style == "publication":
                ax.set_title("CLS → patch", fontsize=10, fontweight="medium", color="#222222")
            ax.set(xticks=[], yticks=[])

        if suptitle:
            fig.suptitle(suptitle, fontsize=12, fontweight="semibold", color="#1a1a1a", y=0.98)

        plt.subplots_adjust(left=0.06, right=0.88 if show_colorbar else 0.96, top=0.90, bottom=0.05)

        if show_colorbar and im_last is not None:
            cbar = fig.colorbar(im_last, ax=head_axes, shrink=0.85, pad=0.02, aspect=25)
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label(
                "Attention weight" if percentile_clip is None else "Attention (clipped)",
                fontsize=9,
                labelpad=6,
            )
            cbar.outline.set_linewidth(0.6)
            cbar.outline.set_edgecolor("#333333")

        if save_path is not None:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                p,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.08,
                facecolor=fig.get_facecolor(),
                edgecolor="none",
            )
        if show:
            plt.show()
        plt.close(fig)


def plot_attention_scores_grid(
    out: dict[str, Any],
    img_chw: Any | None,
    save_path: str | Path | None = None,
    *,
    cls_index: int = 0,
    patch_slice: slice | None = None,
    figscale: float = 2.0,
    cmap: str = "cividis",
    dpi: int = 300,
    show: bool = False,
    style: StyleName = "publication",
    suptitle: str | None = None,
    percentile_clip: tuple[float, float] | None = None,
    layer_stride: int = 1,
    show_colorbar: bool | None = None,
    max_fig_width_in: float = 14.0,
    max_fig_height_in: float = 22.0,
    interpolation: str = "nearest",
    layout: LayoutName = "all_layers",
    paired_layer: int | None = None,
) -> None:
    """Attention heatmaps: full grid or slide + heads column.

    **layout="all_layers"** (default): one row per block, columns = optional image + each head.

    **layout="slide_heads"**: one **reference image** (full height, left) and a **column** of head
    heatmaps (right) for a **single** block ``blocks.{paired_layer}``. Requires ``img_chw`` and
    ``paired_layer``. Ignores ``layer_stride``.

    For each tensor ``v`` in ``out`` with shape (B, H, S, S), uses
    ``v[0, head, cls_index, patch_slice]`` as a 1D patch attention vector, reshaped to a square.

    If ``patch_slice`` is None, infers a slice: all keys after CLS if that count is a perfect
    square; else **last** ``n²`` keys (register tokens before patches; e.g. Virchow2). Pass
    ``patch_slice`` explicitly if your token order differs.

    ``style="publication"`` (default): sans-serif typography, row labels ``Layer k``, column
    titles ``Head h`` on the first row, shared colorbar, fixed [0,1] scale unless
    ``percentile_clip`` is set, PDF-friendly white background.

    ``style="compact"``: smaller per-cell titles (legacy layout), lower default visual density.

    ``img_chw``: (3, H, W) tensor or array for the first column; skipped if None.

    Pass ``save_path`` to write PNG/PDF/SVG, ``show=True`` for notebooks. If both are false,
    raises ``ValueError``.
    """
    import matplotlib.pyplot as plt
    import torch

    if save_path is None and not show:
        raise ValueError("Set save_path and/or show=True")

    if not out:
        raise ValueError("out dict is empty")

    if layout == "slide_heads":
        if paired_layer is None:
            raise ValueError("layout='slide_heads' requires paired_layer (e.g. 15 for blocks.15)")
        if img_chw is None:
            raise ValueError("layout='slide_heads' requires img_chw (e.g. pass --image PATH to the script)")
        if show_colorbar is None:
            show_colorbar = style == "publication"
        _plot_attention_slide_heads_column(
            out,
            img_chw,
            save_path,
            paired_layer=int(paired_layer),
            cls_index=cls_index,
            patch_slice=patch_slice,
            style=style,
            suptitle=suptitle,
            cmap=cmap,
            dpi=dpi,
            show=show,
            percentile_clip=percentile_clip,
            show_colorbar=show_colorbar,
            interpolation=interpolation,
        )
        return

    if layer_stride < 1:
        raise ValueError("layer_stride must be >= 1")

    if show_colorbar is None:
        show_colorbar = style == "publication"

    keys = _sort_layer_keys(list(out.keys()))
    rows: list[tuple[str, Any, int]] = []
    for k in keys:
        v = out[k]
        if isinstance(v, torch.Tensor):
            t = v.detach().float().cpu()
        else:
            t = torch.as_tensor(v).float()
        if t.dim() == 4:
            t = t.squeeze(0)
        if t.dim() != 3:
            raise ValueError(
                f"{k}: expected attention (H,S,S) or (B,H,S,S), got shape {tuple(t.shape)}. "
                "You may be hooking a module that outputs (B,N,C) features, not attention weights."
            )
        n_heads, s, s2 = t.shape
        if s != s2:
            raise ValueError(f"{k}: expected square last dims, got {(s, s2)}")
        rows.append((k, t, n_heads))

    rows = rows[::layer_stride]
    if not rows:
        raise ValueError("no layers left after layer_stride")

    max_heads = max(h for _, _, h in rows)
    n_layers = len(rows)
    n_data_cols = max_heads + (1 if img_chw is not None else 0)

    # --- resolve patch slices & collect heatmaps for optional global percentile scaling ---
    _warned_patch_infer = False
    heatmaps: list[list[np.ndarray]] = []  # [layer][head] = (side, side) float
    grid_side: int | None = None

    for k, t, n_heads in rows:
        seq_len = t.shape[-1]
        if patch_slice is not None:
            sl = patch_slice
        else:
            sl, used_last_n2 = _default_patch_slice(seq_len, cls_index)
            if used_last_n2 and not _warned_patch_infer:
                warnings.warn(
                    f"Inferred patch key slice {sl} for seq_len={seq_len} (prefix after CLS "
                    f"is not a square grid; assuming last {sl.stop - sl.start} keys are patches). "
                    "Override with patch_slice= if wrong.",
                    UserWarning,
                    stacklevel=2,
                )
                _warned_patch_infer = True

        row_mats: list[np.ndarray] = []
        for head in range(n_heads):
            attn_head = t[head]
            vec = attn_head[cls_index, sl].numpy()
            n = vec.size
            side = int(round(n**0.5))
            if side * side != n:
                raise ValueError(
                    f"{k} head {head}: {n} patch tokens in slice {sl} is not a square (sqrt={side}). "
                    "Set patch_slice explicitly (e.g. exclude register tokens)."
                )
            row_mats.append(vec.reshape(side, side).astype(np.float32))
            if grid_side is None:
                grid_side = side
            elif grid_side != side:
                raise ValueError(
                    f"Inconsistent patch grid side {side} vs {grid_side} for {k}; "
                    "all layers must use the same patch layout."
                )
        heatmaps.append(row_mats)

    vmin, vmax = 0.0, 1.0
    if percentile_clip is not None:
        lo_p, hi_p = percentile_clip
        flat = np.concatenate([m.ravel() for row in heatmaps for m in row])
        vmin, vmax = float(np.percentile(flat, lo_p)), float(np.percentile(flat, hi_p))
        if vmin >= vmax:
            vmax = vmin + 1e-6

    # --- figure size: cap width/height for huge grids ---
    if style == "publication":
        cell = min(max_fig_width_in / max(n_data_cols, 1), max_fig_height_in / max(n_layers, 1))
        cell = max(cell, 0.18)
        fig_w = cell * n_data_cols
        fig_h = cell * n_layers
        if show_colorbar:
            fig_w += cell * 0.45
    else:
        fig_w = figscale * n_data_cols
        fig_h = figscale * n_layers

    rc = _publication_rc() if style == "publication" else {}

    with plt.rc_context(rc=rc):
        fig, axs = plt.subplots(n_layers, n_data_cols, figsize=(fig_w, fig_h), squeeze=False)

        img_np = None
        if img_chw is not None:
            chw = _to_numpy(img_chw)
            if chw.ndim == 3 and chw.shape[0] == 3:
                img_np = imagenet_denormalize(chw).transpose(1, 2, 0)
            else:
                raise ValueError(f"img_chw must be (3,H,W), got {chw.shape}")

        col0 = 0
        if img_np is not None:
            for i in range(n_layers):
                axs[i, 0].imshow(np.clip(img_np, 0, 1), interpolation=interpolation)
                for spine in axs[i, 0].spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.6)
                    spine.set_edgecolor("#cccccc")
                if style == "publication" and i == 0:
                    axs[0, 0].set_title("Input", fontsize=10, fontweight="medium", color="#222222")
                elif style == "compact":
                    axs[i, 0].set_title("input", fontsize=8)
            col0 = 1

        im_last = None
        for i, (k, _t, n_heads) in enumerate(rows):
            lyr = _layer_index_from_key(k)
            ylab = f"Layer {lyr}" if lyr is not None else (k[:40] + "..." if len(k) > 40 else k)
            for head in range(max_heads):
                c = col0 + head
                ax = axs[i, c]
                if head >= n_heads:
                    ax.axis("off")
                    continue
                mat = heatmaps[i][head]
                im = ax.imshow(
                    mat,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    interpolation=interpolation,
                    aspect="equal",
                )
                im_last = im
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.5)
                    spine.set_edgecolor("#dddddd")

                if style == "publication":
                    if i == 0:
                        ax.set_title(
                            f"Head {head}",
                            fontsize=10,
                            fontweight="medium",
                            color="#222222",
                        )
                else:
                    ax.set_title(f"{k.split('.')[:3]}\nh{head}", fontsize=7)

            if style == "publication":
                axs[i, col0].set_ylabel(
                    ylab,
                    fontsize=9,
                    fontweight="medium",
                    rotation=90,
                    va="center",
                    ha="center",
                    labelpad=14,
                    color="#333333",
                )
            else:
                axs[i, col0].set_ylabel(k[:48], fontsize=6, rotation=0, ha="right", va="center")

            for c in range(n_data_cols):
                axs[i, c].set(xticks=[], yticks=[])

        if suptitle:
            fig.suptitle(
                suptitle,
                fontsize=12,
                fontweight="semibold",
                color="#1a1a1a",
                y=0.995,
            )

        hspace = min(0.55, 0.22 + 0.012 * n_layers)
        plt.subplots_adjust(
            left=0.09,
            right=0.84 if show_colorbar else 0.98,
            top=0.88 if suptitle else 0.95,
            bottom=0.04,
            wspace=0.18 if style == "publication" else 0.12,
            hspace=hspace,
        )

        if show_colorbar and im_last is not None:
            cbar = fig.colorbar(
                im_last,
                ax=axs.ravel().tolist(),
                shrink=0.78,
                pad=0.02,
                aspect=32,
            )
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label(
                "Attention weight" if percentile_clip is None else "Attention (clipped scale)",
                fontsize=9,
                labelpad=8,
            )
            cbar.outline.set_linewidth(0.6)
            cbar.outline.set_edgecolor("#333333")

        if save_path is not None:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                p,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.08,
                facecolor=fig.get_facecolor(),
                edgecolor="none",
            )
        if show:
            plt.show()
        plt.close(fig)
