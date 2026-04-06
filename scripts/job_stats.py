#!/usr/bin/env python3
"""Summarize metrics from all training checkpoint runs."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main():
    p = argparse.ArgumentParser(description="Summarize job/run stats from checkpoints")
    p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Base checkpoint directory",
    )
    p.add_argument(
        "--run",
        type=str,
        default=None,
        help="Only show this run name (default: all)",
    )
    args = p.parse_args()

    ckpt_base = Path(args.checkpoint_dir)
    if not ckpt_base.exists():
        print(f"Checkpoint dir not found: {ckpt_base}")
        return

    runs = sorted(ckpt_base.iterdir()) if ckpt_base.is_dir() else []
    if args.run:
        runs = [ckpt_base / args.run] if (ckpt_base / args.run).exists() else []

    print(f"\n{'='*70}")
    print("TRAINING RUNS SUMMARY")
    print(f"{'='*70}\n")

    best_fold_counts = {}  # fold -> count of runs where this fold was best

    for run_dir in runs:
        if not run_dir.is_dir():
            continue
        run_name = run_dir.name
        meta_path = run_dir / "run_meta.txt"
        fold_ckpts = sorted(run_dir.glob("fold*_best.pt"))

        # Load run meta
        meta = {}
        if meta_path.exists():
            for line in meta_path.read_text().strip().split("\n"):
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.split("=", 1)
                    meta[k.strip()] = v.strip()

        model = meta.get("MODEL", "?")
        feat_dim = meta.get("FEATURE_DIM", "?")
        epochs = meta.get("EPOCHS", "?")
        status = meta.get("EXIT_STATUS", "running")
        start = meta.get("START_TIME", "")
        end = meta.get("END_TIME", "")

        print(f"RUN: {run_name}")
        print(f"  Model: {model} | feat_dim: {feat_dim} | epochs: {epochs}")
        print(f"  Status: {status} | Start: {start} | End: {end}")
        print(f"  Folds: {len(fold_ckpts)}/5")

        if not fold_ckpts:
            print("  No checkpoint files yet.\n")
            continue

        # Load metrics per fold
        rows = []
        for ckpt_path in fold_ckpts:
            try:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                m = ckpt.get("metrics", {})
                fold_num = int(ckpt_path.stem.replace("fold", "").replace("_best", ""))
                rows.append(
                    (fold_num, m.get("qwk"), m.get("balanced_accuracy"), m.get("accuracy"), m.get("mae"), ckpt.get("epoch"))
                )
            except Exception as e:
                rows.append((-1, None, None, None, None, str(e)))

        rows.sort(key=lambda r: r[0] if isinstance(r[0], int) else -1)
        valid_qwks = [(r[0], r[1]) for r in rows if r[1] is not None]
        best_fold = max(valid_qwks, key=lambda x: x[1])[0] if valid_qwks else None

        print(f"  {'Fold':>6} {'qwk':>8} {'bal_acc':>8} {'acc':>8} {'mae':>8} {'best_ep':>8}")
        print(f"  {'-'*50}")
        for r in rows:
            fold, qwk, bal, acc, mae, ep = r
            fold_s = str(fold) if isinstance(fold, int) and fold >= 0 else "?"
            mark = " *" if fold == best_fold and best_fold is not None else ""
            qwk_s = f"{qwk:.4f}" if qwk is not None else "-"
            bal_s = f"{bal:.4f}" if bal is not None else "-"
            acc_s = f"{acc:.4f}" if acc is not None else "-"
            mae_s = f"{mae:.3f}" if mae is not None else "-"
            ep_s = str(ep) if ep is not None else "-"
            print(f"  {fold_s:>6} {qwk_s:>8} {bal_s:>8} {acc_s:>8} {mae_s:>8} {ep_s:>8}{mark}")

        if len(rows) >= 2 and all(r[1] is not None for r in rows):
            qwks = [r[1] for r in rows]
            mean_qwk = sum(qwks) / len(qwks)
            std_qwk = (sum((x - mean_qwk) ** 2 for x in qwks) / len(qwks)) ** 0.5
            print(f"  {'mean':>6} {mean_qwk:.4f}  (std: {std_qwk:.4f})")
        if best_fold is not None:
            print(f"  Best fold (by qwk): {best_fold}")
            best_fold_counts[best_fold] = best_fold_counts.get(best_fold, 0) + 1
        print()

    if best_fold_counts:
        total = sum(best_fold_counts.values())
        print("Best fold per run:")
        for fold in sorted(best_fold_counts.keys()):
            n = best_fold_counts[fold]
            pct = 100 * n / total
            print(f"  fold {fold}: {n}/{total} runs ({pct:.0f}%)")
        print()

    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
