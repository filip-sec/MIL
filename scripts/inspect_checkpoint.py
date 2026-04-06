#!/usr/bin/env python3
"""Inspect a MIL checkpoint (.pt) – keys, metrics, model config.

Large checkpoints (hundreds of MB) can take **tens of seconds to a few minutes** to load
from network storage (e.g. MetaCentrum home). PyTorch 2.1+ uses ``mmap=True`` when available
to reduce peak RAM and often speed first read. Copy the file to ``$SCRATCHDIR`` or local SSD
if it is consistently slow.
"""

import argparse
import inspect
from pathlib import Path

import torch


def _torch_load_ckpt(path: Path):
    base = dict(map_location="cpu", weights_only=False)
    sig = inspect.signature(torch.load)
    if "mmap" in sig.parameters:
        try:
            return torch.load(path, **base, mmap=True)
        except (OSError, RuntimeError, TypeError):
            pass
    return torch.load(path, **base)


def main():
    ap = argparse.ArgumentParser(description="Inspect MIL checkpoint")
    ap.add_argument("ckpt", type=Path, help="Path to fold*_best.pt")
    args = ap.parse_args()

    sz_mb = args.ckpt.stat().st_size / (1024 * 1024)
    print(f"Loading {args.ckpt.name} ({sz_mb:.1f} MiB) — large files can take 30s–2min on NFS…", flush=True)
    ckpt = _torch_load_ckpt(args.ckpt)
    print(f"Checkpoint: {args.ckpt}")
    print("Keys:", list(ckpt.keys()))

    for k in ["epoch", "feat_dim", "model_name", "hidden", "top_k"]:
        if k in ckpt:
            print(f"  {k}: {ckpt[k]}")

    if "metrics" in ckpt:
        print("  metrics:", ckpt["metrics"])

    if "model_state_dict" in ckpt:
        sd = ckpt["model_state_dict"]
        print(f"  model_state_dict: {len(sd)} params")
        for name, t in list(sd.items())[:8]:
            print(f"    {name}: {tuple(t.shape)}")
        if len(sd) > 8:
            print(f"    ... and {len(sd) - 8} more")

    if "optimizer_state_dict" in ckpt:
        print("  optimizer_state_dict: present")
    if "scheduler_state_dict" in ckpt:
        print("  scheduler_state_dict: present")


if __name__ == "__main__":
    main()
