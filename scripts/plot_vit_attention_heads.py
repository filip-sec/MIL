#!/usr/bin/env python3
"""Plot a grid: optional input image + CLS→patch attention per head, per layer.

``--dict`` is a ``torch.save`` file containing a ``dict[str, Tensor]`` of attention matrices
``(B, num_heads, seq, seq)``. Build it with ``extract_encoder_intermediates.py`` using
``--no-fused-attn --contains attn_drop`` (fused SDPA skips ``attn_drop``, so hooks capture
nothing without that flag).

Example::

    python scripts/extract_encoder_intermediates.py ... --save-dict attn_out.pt
    python scripts/plot_vit_attention_heads.py --dict attn_out.pt --out figures/attn_grid.pdf \\
        --title 'Virchow2 — CLS to patch attention'

Use **single quotes** around ``--title`` in bash if the string has spaces or em dashes.

Publication defaults: ``--style publication``, 300 DPI, ``cividis`` colormap, shared colorbar,
row labels ``Layer k``, column titles ``Head h``. Use ``--style compact`` for the older dense
per-cell labels. PDF/SVG: set ``--out`` extension accordingly.

With a reference tile, use ``--image``. Virchow2 register tokens: use ``--patch-start`` /
``--patch-len`` if needed (otherwise inferred).

**Slide + heads (one layer):** use ``--layout slide_heads --paired-layer N --image tile.png`` —
left = one slide/tile (full height), right = stacked CLS→patch maps for each head at ``blocks.N``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot ViT CLS→patch attention (grid or slide+heads column).",
        epilog=(
            "If you get 'unrecognized arguments: --layout' on the cluster, copy the latest files "
            "from your laptop, e.g.  bash scripts/sync_to_cluster.sh  "
            "(needs scripts/plot_vit_attention_heads.py and mil/vit_attention_viz.py)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dict", dest="dict_path", type=Path, required=True, help="torch.save dict of tensors")
    ap.add_argument("--out", type=Path, required=True, help="Output path (.png / .pdf / .svg)")
    ap.add_argument("--image", type=Path, default=None, help="Optional image (RGB file) resized inside script")
    ap.add_argument("--patch-start", type=int, default=None, help="Start index of patch tokens in seq (after CLS/regs)")
    ap.add_argument("--patch-len", type=int, default=None, help="Number of patch tokens (e.g. 256 for 16×16)")
    ap.add_argument("--cls-index", type=int, default=0)
    ap.add_argument(
        "--style",
        choices=("publication", "compact"),
        default="publication",
        help="publication: layer/head labels + colorbar; compact: small per-cell titles",
    )
    ap.add_argument("--title", default=None, help="Figure suptitle (quote if it contains spaces)")
    ap.add_argument("--dpi", type=int, default=300, help="Figure DPI (raster backends)")
    ap.add_argument("--cmap", default="cividis", help="Colormap name (e.g. cividis, viridis, plasma)")
    ap.add_argument(
        "--layer-stride",
        type=int,
        default=1,
        metavar="N",
        help="Plot every N-th transformer block (e.g. 2 for half the rows)",
    )
    ap.add_argument(
        "--clip-percentiles",
        nargs=2,
        type=float,
        metavar=("LO", "HI"),
        default=None,
        help="Global percentile scaling for contrast, e.g. 1 99",
    )
    ap.add_argument("--no-colorbar", action="store_true", help="Hide shared colorbar")
    ap.add_argument(
        "--interpolation",
        default="nearest",
        choices=("nearest", "bilinear", "bicubic"),
        help="imshow interpolation",
    )
    ap.add_argument(
        "--layout",
        choices=("all_layers", "slide_heads"),
        default="all_layers",
        help="all_layers: grid of layers×heads; slide_heads: one image column + heads for --paired-layer",
    )
    ap.add_argument(
        "--paired-layer",
        type=int,
        default=None,
        metavar="N",
        help="With --layout slide_heads: transformer block index (e.g. 12 for blocks.12.attn...)",
    )
    args = ap.parse_args()

    if args.layout == "slide_heads":
        if args.paired_layer is None:
            ap.error("--layout slide_heads requires --paired-layer N")
        if args.image is None:
            ap.error("--layout slide_heads requires --image (reference slide or patch tile)")

    import torch

    dict_path = args.dict_path.expanduser()
    if not dict_path.is_file():
        print(
            f"File not found: {dict_path.resolve()}\n"
            "  Create it first, e.g.:\n"
            "    python scripts/extract_encoder_intermediates.py ... --save-dict attn_out.pt\n"
            "  or in Python: torch.save(out, 'attn_out.pt')",
            file=sys.stderr,
        )
        raise SystemExit(1)

    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    from mil.vit_attention_viz import plot_attention_scores_grid

    blob = torch.load(dict_path, map_location="cpu", weights_only=False)
    if not isinstance(blob, dict):
        raise TypeError(f"Expected dict in {dict_path}, got {type(blob)}")
    if not blob:
        raise ValueError(
            f"{dict_path} is an empty dict. For timm ViTs, re-run extract with "
            "--no-fused-attn when using --contains attn_drop (fused SDPA never calls attn_drop)."
        )
    out = {k: v for k, v in blob.items() if hasattr(v, "shape")}
    if not out:
        ks = list(blob.keys())
        preview = ks[:12]
        extra = f" (+{len(ks) - 12} more)" if len(ks) > 12 else ""
        raise ValueError(
            f"No tensor-like entries (with .shape) in {dict_path}. Keys: {preview}{extra}"
        )

    img = None
    if args.image is not None:
        from PIL import Image
        import torchvision.transforms as T

        pil = Image.open(args.image).convert("RGB")
        t = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(
            pil
        )
        img = t

    patch_slice = None
    if args.patch_start is not None:
        pl = args.patch_len
        if pl is None:
            ap.error("--patch-len required when --patch-start is set")
        patch_slice = slice(args.patch_start, args.patch_start + pl)

    clip = None
    if args.clip_percentiles is not None:
        clip = (float(args.clip_percentiles[0]), float(args.clip_percentiles[1]))

    plot_attention_scores_grid(
        out,
        img,
        args.out,
        cls_index=args.cls_index,
        patch_slice=patch_slice,
        style=args.style,
        suptitle=args.title,
        dpi=args.dpi,
        cmap=args.cmap,
        layer_stride=args.layer_stride,
        percentile_clip=clip,
        show_colorbar=not args.no_colorbar,
        interpolation=args.interpolation,
        layout=args.layout,
        paired_layer=args.paired_layer,
    )
    print(f"Saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
