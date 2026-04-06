#!/usr/bin/env python3
"""Run a dummy forward and capture tensors at submodules whose names match ``--contains``.

For **Virchow2 / UNI2-h**, ``torchvision.models.feature_extraction.create_feature_extractor``
usually **fails** (same FX / control-flow issue as ``get_graph_node_names``). This script defaults
to **forward hooks** on ``named_modules()``, which matches the node names printed by
``print_torch_graph_nodes.py``.

``attn_drop`` sits on the *manual* attention path. With timm's default **fused SDPA**
(``F.scaled_dot_product_attention``), that submodule is **never called**, so hooks match
many ``attn_drop`` layers but capture **nothing**. Use ``--no-fused-attn`` (or
``TIMM_FUSED_ATTN=0`` before building the model) when hooking ``attn_drop`` for attention
probabilities shaped ``(B, num_heads, N, N)``.

Examples:
  cd ~/MIL && python scripts/extract_encoder_intermediates.py \\
      --hf-cache-dir /storage/brno2/home/$USER/hf_cache \\
      --hf-token-file .hf_token --backbone virchow2 \\
      --no-fused-attn --contains attn_drop --save-dict attn_out.pt

  python scripts/plot_vit_attention_heads.py --dict attn_out.pt --out figures/attn_grid.png

  # Try FX first (often works for ResNet; ViT may fall back to hooks)
  python scripts/extract_encoder_intermediates.py --backbone resnet18 --contains layer1 --method auto
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def _load_print_torch_graph_nodes():
    """Load sibling module without package import."""
    p = Path(__file__).resolve().parent / "print_torch_graph_nodes.py"
    spec = importlib.util.spec_from_file_location("print_torch_graph_nodes", p)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _interesting_names(model, substrs: list[str]) -> list[str]:
    names = [n for n, _ in model.named_modules() if n]
    return [n for n in names if all(s in n for s in substrs)]


def _capture_with_hooks(model, x, names: list[str]):
    """Register hooks only on listed module paths (must match named_modules keys)."""
    import torch

    by_name = dict(model.named_modules())
    captured: dict = {}
    handles = []

    for name in names:
        m = by_name.get(name)
        if m is None:
            continue

        def make_hook(n: str):
            def hook(_module, _inp, out):
                if torch.is_tensor(out):
                    captured[n] = out.detach()
                else:
                    captured[n] = out

            return hook

        handles.append(m.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return captured


def _try_create_feature_extractor(model, x, names: list[str]):
    """Returns (dict of tensors, None) on success, or (None, error_message)."""
    import torch
    from torchvision.models.feature_extraction import create_feature_extractor

    # Output dict keys: safe identifiers; values: graph node names (here = module paths).
    return_nodes = {n.replace(".", "__"): n for n in names}
    try:
        ex = create_feature_extractor(model, return_nodes=return_nodes)
        ex.eval()
        with torch.no_grad():
            out = ex(x)
        return out, None
    except Exception as e:
        return None, str(e)


def main() -> None:
    ap = argparse.ArgumentParser(description="Capture intermediate encoder outputs (hooks or FX).")
    ap.add_argument("--backbone", choices=("resnet18", "virchow2", "uni2_h"), required=True)
    ap.add_argument(
        "--contains",
        action="append",
        default=[],
        metavar="SUBSTR",
        required=True,
        help="Submodule name must contain this string (repeat = all must match).",
    )
    ap.add_argument(
        "--method",
        choices=("auto", "fx", "hooks"),
        default="auto",
        help="auto: try FX then hooks; fx: FX only; hooks: forward hooks only.",
    )
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--channels", type=int, default=3)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--width", type=int, default=224)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--hf-token-file", type=Path, default=None)
    ap.add_argument("--hf-cache-dir", type=Path, default=None)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--strict-checkpoint", action="store_true")
    ap.add_argument(
        "--save-dict",
        type=Path,
        default=None,
        help="Save captured tensors with torch.save (e.g. attn_out.pt for plot_vit_attention_heads.py).",
    )
    ap.add_argument(
        "--no-fused-attn",
        action="store_true",
        help=(
            "Disable timm fused SDPA before building ViT backbones. "
            "Otherwise modules like attn_drop are unused in forward and hooks capture nothing."
        ),
    )
    args = ap.parse_args()

    import torch

    if args.no_fused_attn:
        import timm.layers

        timm.layers.set_fused_attn(False)

    ptn = _load_print_torch_graph_nodes()
    if args.hf_cache_dir is not None:
        ptn._set_hf_cache_dir(args.hf_cache_dir)
    if args.hf_token_file is not None:
        ptn._read_hf_token_file(args.hf_token_file.expanduser())
    ptn._ensure_hf_token()

    use_pretrained = not args.no_pretrained and args.checkpoint is None
    if args.checkpoint is not None:
        use_pretrained = False

    if args.backbone == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        if args.checkpoint:
            ckpt = ptn._resolve_checkpoint_path(args.checkpoint)
            model = resnet18(weights=None)
            ptn._load_checkpoint_into_model(model, ckpt, strict=args.strict_checkpoint)
        else:
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif args.backbone == "virchow2":
        model = ptn._build_virchow2(pretrained=use_pretrained)
        if args.checkpoint is not None:
            ckpt = ptn._resolve_checkpoint_path(args.checkpoint)
            ptn._load_checkpoint_into_model(model, ckpt, strict=args.strict_checkpoint)
    else:
        model = ptn._build_uni2_h(pretrained=use_pretrained)
        if args.checkpoint is not None:
            ckpt = ptn._resolve_checkpoint_path(args.checkpoint)
            ptn._load_checkpoint_into_model(model, ckpt, strict=args.strict_checkpoint)

    device = torch.device(args.device)
    model = model.to(device)

    substrs = args.contains
    names = _interesting_names(model, substrs)
    if not names:
        print(f"No modules matched --contains {substrs!r}", file=sys.stderr)
        raise SystemExit(1)
    print(f"Matched {len(names)} modules (showing shapes after one forward)", file=sys.stderr)

    x = torch.randn(args.batch, args.channels, args.height, args.width, device=device)

    out = None
    used = None
    if args.method in ("auto", "fx"):
        out, err = _try_create_feature_extractor(model, x, names)
        if out is not None:
            used = "create_feature_extractor"
        elif args.method == "fx":
            print(f"create_feature_extractor failed: {err}", file=sys.stderr)
            raise SystemExit(1)

    if out is None:
        out = _capture_with_hooks(model, x, names)
        used = "forward_hooks"

    if not out:
        print(
            "Captured dict is empty: no forward hook ran (submodules unused in this forward). "
            "For timm ViTs with default fused attention, hook targets like `attn_drop` are skipped — "
            "re-run with --no-fused-attn, or use --contains proj_drop / qkv, etc.",
            file=sys.stderr,
        )
        if args.save_dict is not None:
            raise SystemExit(1)

    print(f"method={used}")
    for k in sorted(out.keys()):
        v = out[k]
        if torch.is_tensor(v):
            print(k, tuple(v.shape), v.dtype, v.device)
        else:
            print(k, type(v))

    if args.save_dict is not None:
        to_save = {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in out.items()}
        args.save_dict.parent.mkdir(parents=True, exist_ok=True)
        torch.save(to_save, args.save_dict)
        print(f"Saved dict: {args.save_dict.resolve()}", flush=True)


if __name__ == "__main__":
    main()
