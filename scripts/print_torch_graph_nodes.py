#!/usr/bin/env python3
"""Print FX graph node names (or ``named_modules`` fallback) for inspection / hooks.

Run from MIL repo root (where ``scripts/`` lives), e.g. ``cd ~/MIL`` or
``cd /storage/brno2/home/$USER/MIL``.

- **virchow2** / **uni2_h**: timm models from Hugging Face (set token via env or ``--hf-token-file``).
- **mil**: your trained bag classifier from ``fold*_best.pt`` (not the tile encoder).

**Do not commit HF tokens.** Put the token in ``.hf_token`` (one line), add to env, or use
``--hf-token-file`` — that file is gitignored in this repo.

Examples:
  cd ~/MIL && python scripts/print_torch_graph_nodes.py --backbone resnet18 --max-print 50
  python scripts/print_torch_graph_nodes.py --hf-token-file .hf_token --backbone virchow2 --contains attn_drop

MetaCentrum: if ``/scratch`` is full (Errno 28), put the Hugging Face cache on ``/storage``::

  python scripts/print_torch_graph_nodes.py --hf-cache-dir /storage/brno2/home/$USER/hf_cache \\
      --hf-token-file .hf_token --backbone virchow2 --max-print 50

Token file name is ``.hf_token`` (underscore), not ``.hf-token``. MIL ``.pt`` files: use a path
where they actually exist (e.g. sync from laptop) or ``--checkpoint /storage/.../fold1_best.pt``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _set_hf_cache_dir(path: Path) -> None:
    """Avoid full ``/scratch`` on clusters: HF downloads go under this directory."""
    root = path.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    hub = root / "hub"
    hub.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(hub)
    # timm may consult these too
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub))
    print(f"HF cache: HF_HOME={root}", file=sys.stderr)


def _read_hf_token_file(path: Path) -> None:
    if not path.is_file():
        hint = ""
        if path.name == ".hf-token":
            hint = " (did you mean .hf_token with an underscore?)"
        raise FileNotFoundError(f"--hf-token-file not found: {path}{hint}")
    line = path.read_text(encoding="utf-8").strip().splitlines()[0].strip()
    if not line:
        raise ValueError(f"Empty first line in {path}")
    os.environ["HUGGING_FACE_HUB_TOKEN"] = line
    os.environ["HF_TOKEN"] = line


def _ensure_hf_token() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ.setdefault("HUGGING_FACE_HUB_TOKEN", token)


def _load_checkpoint_into_model(model, path: Path, *, strict: bool) -> None:
    import torch

    blob = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(blob, dict):
        if "state_dict" in blob and isinstance(blob["state_dict"], dict):
            sd = blob["state_dict"]
        elif "model" in blob and isinstance(blob["model"], dict):
            sd = blob["model"]
        elif "model_state_dict" in blob and isinstance(blob["model_state_dict"], dict):
            sd = blob["model_state_dict"]
        else:
            sd = blob
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(blob)}")

    if not sd:
        raise ValueError("Empty state dict in checkpoint")

    keys = list(sd.keys())
    if keys and isinstance(keys[0], str) and keys[0].startswith("module."):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}

    ret = model.load_state_dict(sd, strict=strict)
    if ret is not None:
        print(
            f"checkpoint: {path}  strict={strict}  "
            f"missing_keys={len(ret.missing_keys)}  unexpected_keys={len(ret.unexpected_keys)}",
            file=sys.stderr,
        )
    else:
        print(f"checkpoint: {path}  strict={strict}  (loaded)", file=sys.stderr)


def _build_virchow2(*, pretrained: bool):
    """Match Paige Virchow2 checkpoint (SwiGLU MLP + register tokens).

    Without ``SwiGLUPacked`` / ``SiLU``, timm builds a vanilla ViT-H MLP and weights fail
    (e.g. ``fc2.weight`` [1280, 6832] vs ckpt [1280, 3416]).
    """
    import timm
    import timm.layers
    import torch.nn as nn

    return timm.create_model(
        "hf-hub:paige-ai/Virchow2",
        pretrained=pretrained,
        num_classes=0,
        mlp_layer=timm.layers.SwiGLUPacked,
        act_layer=nn.SiLU,
        reg_tokens=4,
    )


def _build_uni2_h(*, pretrained: bool):
    """MahmoodLab UNI2-h (1536-d ViT); same family as TRIDENT ``uni_v2`` features."""
    import timm
    import timm.layers
    import torch.nn as nn

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


def _build_mil_from_fold_ckpt(ckpt_path: Path):
    """Load AttentionMIL / GatedAttentionMIL / OrdinalMIL from ``fold*_best.pt``."""
    import torch

    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))
    from mil.model import build_model

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" not in ckpt:
        raise KeyError(f"{ckpt_path} has no model_state_dict (not a MIL training checkpoint?)")
    name = ckpt["model_name"]
    feat_dim = ckpt["feat_dim"]
    hidden = ckpt.get("hidden", 256 if name != "attention" else 128)
    top_k = ckpt.get("top_k", 8)
    kw: dict = {"feat_dim": feat_dim, "num_classes": 6, "hidden": hidden}
    if name == "ordinal":
        kw["top_k"] = top_k
    model = build_model(name, **kw)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    return model, {"model_name": name, "feat_dim": feat_dim, "hidden": hidden, "top_k": top_k}


def _resolve_checkpoint_path(p: Path) -> Path:
    """Absolute path, cwd-relative, or under MIL repo root."""
    p = p.expanduser()
    if p.is_file():
        return p.resolve()
    repo = Path(__file__).resolve().parent.parent
    cand = repo / p
    if cand.is_file():
        return cand.resolve()
    return p


def main() -> None:
    ap = argparse.ArgumentParser(description="List FX graph nodes (or module names) for inspection.")
    ap.add_argument(
        "--backbone",
        choices=("resnet18", "virchow2", "uni2_h", "mil"),
        default="resnet18",
        help="resnet18 | timm Virchow2 | timm UNI2-h | mil (fold*_best.pt bag model)",
    )
    ap.add_argument("--max-print", type=int, default=0, help="Print only first N names after filters (0 = all)")
    ap.add_argument(
        "--contains",
        action="append",
        default=[],
        metavar="SUBSTR",
        help="Only list module names that contain this substring. Use multiple times: all substrings must match.",
    )
    ap.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Hub models only: do not download weights. Ignored if --checkpoint is set.",
    )
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="virchow2/uni2_h/resnet18: weight file after building skeleton. mil: required (fold*_best.pt).",
    )
    ap.add_argument(
        "--strict-checkpoint",
        action="store_true",
        help="strict=True when loading foundation --checkpoint (default: strict=False). MIL ckpt is always strict.",
    )
    ap.add_argument(
        "--hf-token-file",
        type=Path,
        default=None,
        help="Read Hugging Face token from first line of this file (e.g. MIL/.hf_token). Not logged.",
    )
    ap.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=None,
        help="Store Hugging Face / timm hub cache here (use /storage/... on MetaCentrum if /scratch is full).",
    )
    args = ap.parse_args()

    from torchvision.models.feature_extraction import get_graph_node_names

    if args.hf_cache_dir is not None:
        _set_hf_cache_dir(args.hf_cache_dir)

    if args.hf_token_file is not None:
        _read_hf_token_file(args.hf_token_file.expanduser())
    _ensure_hf_token()

    meta = {}
    if args.backbone == "mil":
        if args.checkpoint is None:
            ap.error("--backbone mil requires --checkpoint fold*_best.pt")
        ckpt_path = _resolve_checkpoint_path(args.checkpoint)
        if not ckpt_path.is_file():
            repo = Path(__file__).resolve().parent.parent
            print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
            print(f"  Tried: {ckpt_path.resolve()}", file=sys.stderr)
            print(f"  Put fold*_best.pt in cwd or under MIL repo: {repo}", file=sys.stderr)
            raise SystemExit(1)
        model, meta = _build_mil_from_fold_ckpt(ckpt_path)
    elif args.backbone == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        if args.checkpoint:
            ckpt_path = _resolve_checkpoint_path(args.checkpoint)
            if not ckpt_path.is_file():
                print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
                raise SystemExit(1)
            model = resnet18(weights=None)
            _load_checkpoint_into_model(model, ckpt_path, strict=args.strict_checkpoint)
        else:
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        use_pretrained = not args.no_pretrained and args.checkpoint is None
        if args.checkpoint is not None:
            use_pretrained = False

        try:
            if args.backbone == "virchow2":
                model = _build_virchow2(pretrained=use_pretrained)
            else:
                model = _build_uni2_h(pretrained=use_pretrained)
        except OSError as e:
            if getattr(e, "errno", None) == 28:
                print(
                    "Disk full while downloading HF cache (often /scratch on MetaCentrum).\n"
                    "  Re-run with e.g.  --hf-cache-dir /storage/brno2/home/$USER/hf_cache",
                    file=sys.stderr,
                )
            raise

        if args.checkpoint is not None:
            ckpt_path = _resolve_checkpoint_path(args.checkpoint)
            if not ckpt_path.is_file():
                print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
                raise SystemExit(1)
            _load_checkpoint_into_model(model, ckpt_path, strict=args.strict_checkpoint)

    model.eval()
    name_source = "fx_graph"
    try:
        nodes, _ = get_graph_node_names(model)
        names = list(nodes)
    except Exception as e:
        print(f"get_graph_node_names: {e}", file=sys.stderr)
        if "control flow" in str(e).lower() or "symbolically traced" in str(e).lower():
            print(
                "  (ViT/Virchow2/UNI2 often use dynamic control flow; FX tracing is not supported. "
                "Use the ``named_modules()`` list below for ``register_forward_hook`` — that is normal.)",
                file=sys.stderr,
            )
        else:
            print("  Falling back to model.named_modules()", file=sys.stderr)
        names = [n for n, _ in model.named_modules() if n]
        name_source = "named_modules"

    if args.contains:
        before = len(names)
        names = [n for n in names if all(sub in n for sub in args.contains)]
        print(
            f"filter --contains {args.contains!r}: {before} -> {len(names)} names",
            file=sys.stderr,
        )

    n = len(names)
    ckpt_note = f"  checkpoint={args.checkpoint}" if args.checkpoint else ""
    meta_note = f"  {meta}" if meta else ""
    print(f"backbone={args.backbone}  source={name_source}  num_names={n}{ckpt_note}{meta_note}")
    out = names if args.max_print <= 0 else names[: args.max_print]
    for name in out:
        print(name)
    if args.max_print > 0 and n > args.max_print:
        print(f"... ({n - args.max_print} more)")


if __name__ == "__main__":
    main()
