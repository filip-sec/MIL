"""Training and evaluation loops for MIL."""

from __future__ import annotations

import re
import time
import warnings
from pathlib import Path

import h5py
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data import CachedBagDataset, collate_bags, preload_features
from .loss import OrdinalWithCELoss
from .model import build_model
from .utils import compute_metrics, load_checkpoint, save_checkpoint, seed_everything


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    is_ordinal: bool = False,
) -> tuple[float, int, int]:
    """Returns (avg_loss, total_patches, n_fully_masked)."""
    model.train()
    total_loss = 0.0
    total_patches = 0
    n_fully_masked = 0
    for feats, mask, labels, _ids, _coords in loader:
        feats, mask, labels = feats.to(device), mask.to(device), labels.to(device)
        if mask is not None:
            n_valid = mask.sum(dim=1)
            total_patches += int(n_valid.sum().item())
            n_fully_masked += int((n_valid == 0).sum().item())
        optimizer.zero_grad()
        output = model(feats, mask)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    n_batches = len(loader)
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    return avg_loss, total_patches, n_fully_masked


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_ordinal: bool = False,
    criterion=None,
) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches = 0
    total_patches = 0
    n_fully_masked = 0
    for feats, mask, labels, _ids, _coords in loader:
        feats, mask, labels = feats.to(device), mask.to(device), labels.to(device)
        if mask is not None:
            n_valid = mask.sum(dim=1)
            total_patches += int(n_valid.sum().item())
            n_fully_masked += int((n_valid == 0).sum().item())
        output = model(feats, mask)
        if criterion is not None:
            loss = criterion(output, labels)
            total_loss += loss.item()
            n_batches += 1
        if is_ordinal:
            preds = model.predict(feats, mask)
        else:
            preds = output.argmax(dim=1)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())
    preds_np = torch.cat(all_preds).numpy()
    labels_np = torch.cat(all_labels).numpy()
    out = compute_metrics(preds_np, labels_np)
    if criterion is not None and n_batches > 0:
        out["loss"] = total_loss / n_batches
    out["n_patches"] = total_patches
    out["n_fully_masked"] = n_fully_masked
    return out


def run_fold(
    fold: int,
    train_ids: list[str],
    val_ids: list[str],
    feat_dir: Path,
    train_labels: pd.DataFrame,
    feat_dim: int,
    cfg: dict,
    device: torch.device,
    cache: dict[str, torch.Tensor] | None = None,
    coord_cache: dict[str, torch.Tensor | None] | None = None,
) -> dict:
    model_name = cfg.get("model_name", "attention")
    hidden = cfg.get("hidden", 256 if model_name != "attention" else 128)
    top_k = cfg.get("top_k", 8)
    epochs = cfg.get("epochs", 10)
    lr = cfg.get("lr", 1e-4)
    batch_size = cfg.get("batch_size", 8)
    num_workers = cfg.get("num_workers", 4)
    patience = cfg.get("patience", 0)
    patch_dropout = cfg.get("patch_dropout", 0.0)
    max_patches_train = cfg.get("max_patches_train", cfg.get("max_patches"))
    max_patches_val = cfg.get("max_patches_val", cfg.get("max_patches"))
    attention_weights = cfg.get("attention_weights")
    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))

    is_ordinal = model_name == "ordinal"

    # Datasets: cap reduces RAM, collation time, forward time (biggest perf win)
    train_ds_kw = dict(
        patch_dropout=patch_dropout,
        max_patches=max_patches_train,
        max_patches_val=max_patches_val,
        attention_weights=attention_weights,
    )
    # Validation only uses max_patches_val (training=False ignores max_patches).
    val_ds_kw = dict(max_patches_val=max_patches_val)
    if cache is not None:
        train_ds = CachedBagDataset(
            cache, train_ids, train_labels, coord_cache=coord_cache, training=True, **train_ds_kw
        )
        val_ds = CachedBagDataset(cache, val_ids, train_labels, coord_cache=coord_cache, **val_ds_kw)
    else:
        from .data import H5BagDataset

        coords_dir = cfg.get("coords_dir")
        train_ds = H5BagDataset(feat_dir, train_ids, train_labels, coords_dir=coords_dir, training=True, **train_ds_kw)
        val_ds = H5BagDataset(feat_dir, val_ids, train_labels, coords_dir=coords_dir, **val_ds_kw)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_bags,
        persistent_workers=num_workers > 0,  # avoid respawn per epoch
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_bags,
        persistent_workers=num_workers > 0,
    )
    print(
        f"  train: {len(train_ids)} slides, {len(train_loader)} batches | val: {len(val_ids)} slides, {len(val_loader)} batches",
        flush=True,
    )

    model_kw = {"feat_dim": feat_dim, "num_classes": 6, "hidden": hidden}
    if model_name == "ordinal":
        model_kw["top_k"] = top_k
    model = build_model(model_name, **model_kw).to(device)

    momentum = cfg.get("momentum", 0.9)
    weight_decay = cfg.get("weight_decay", 1e-4)
    warmup_epochs = cfg.get("warmup_epochs", 5)
    min_lr = cfg.get("min_lr", 1e-6)

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs - warmup_epochs), eta_min=min_lr)
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )

    if is_ordinal:
        criterion = OrdinalWithCELoss(num_classes=6, ce_weight=cfg.get("ce_weight", 0.3))
    else:
        criterion = nn.CrossEntropyLoss()

    best_qwk = -1.0
    best_metrics = {}
    wait = 0
    start_epoch = 1

    resume_from = cfg.get("resume_from")
    if resume_from is not None:
        resume_path = Path(resume_from)
        if resume_path.exists():
            ckpt = load_checkpoint(resume_path, model, optimizer=optimizer, scheduler=scheduler)
            start_epoch = ckpt["epoch"] + 1
            best_qwk = ckpt.get("metrics", {}).get("qwk", -1.0)
            best_metrics = ckpt.get("metrics", {})
            best_metrics["epoch"] = ckpt["epoch"]
            print(f"  Resumed from {resume_path} (epoch {ckpt['epoch']})", flush=True)
        else:
            print(f"  Resume path {resume_path} not found, starting fresh", flush=True)

    n_train, n_val = len(train_ids), len(val_ids)

    for epoch in range(start_epoch, epochs + 1):
        epoch_start = time.perf_counter()
        t0 = time.perf_counter()
        train_loss, train_patches, n_fm_train = train_one_epoch(
            model, train_loader, criterion, optimizer, device, is_ordinal
        )
        train_time = time.perf_counter() - t0
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*epoch parameter.*deprecated.*", module="torch.optim.lr_scheduler"
            )
            scheduler.step()
        t0 = time.perf_counter()
        metrics = evaluate(model, val_loader, device, is_ordinal, criterion=criterion)
        val_time = time.perf_counter() - t0
        epoch_time = time.perf_counter() - epoch_start
        qwk = metrics["qwk"]
        val_loss = metrics.get("loss", float("nan"))
        val_patches = metrics.get("n_patches", 0)
        n_fm_val = metrics.get("n_fully_masked", 0)
        train_sps = n_train / train_time if train_time > 0 else 0
        val_sps = n_val / val_time if val_time > 0 else 0
        train_pps = train_patches / train_time if train_time > 0 else 0
        val_pps = val_patches / val_time if val_time > 0 else 0
        fm_str = f" | fm_train={n_fm_train} fm_val={n_fm_val}" if (n_fm_train or n_fm_val) else ""
        print(
            f"  Fold {fold} | Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"qwk={qwk:.4f} | bal_acc={metrics['balanced_accuracy']:.4f} | mae={metrics['mae']:.3f} | "
            f"train={train_time:.1f}s ({train_sps:.0f} slides/s, {train_pps:.0f} patches/s) | "
            f"val={val_time:.1f}s ({val_sps:.0f} slides/s, {val_pps:.0f} patches/s) | epoch={epoch_time:.1f}s{fm_str}",
            flush=True,
        )
        if qwk > best_qwk:
            best_qwk = qwk
            best_metrics = {**metrics, "epoch": epoch}
            ckpt_path = checkpoint_dir / f"fold{fold}_best.pt"
            save_checkpoint(
                ckpt_path,
                model,
                epoch,
                metrics,
                feat_dim=feat_dim,
                model_name=model_name,
                hidden=hidden,
                top_k=top_k,
                optimizer=optimizer,
                scheduler=scheduler,
            )
            wait = 0
        else:
            wait += 1
            if patience > 0 and wait >= patience:
                print(f"  Early stop at epoch {epoch}", flush=True)
                break

    return best_metrics


def run_cv(
    feat_dir: Path,
    train_csv: Path,
    splits_csv: Path,
    cfg: dict,
    folds: list[int] | None = None,
) -> pd.DataFrame:
    seed_everything(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    train_labels = pd.read_csv(train_csv).set_index("image_id")
    feat_dir = Path(feat_dir)
    available_h5 = {p.stem for p in feat_dir.glob("*.h5")}
    print(f"Available .h5 files: {len(available_h5)}", flush=True)

    overlap = available_h5 & set(train_labels.index)
    if not overlap:
        raise ValueError("No slides found in both feat_dir and train_labels. Check feat_dir and train_csv.")
    sample_id = next(iter(overlap))
    with h5py.File(feat_dir / f"{sample_id}.h5", "r") as f:
        feat_dim = f["features"].shape[1]
    print(f"feat_dim: {feat_dim}", flush=True)

    splits = pd.read_csv(splits_csv)
    if folds is None:
        folds = sorted(splits["fold"].unique())

    checkpoint_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    cv_start = time.perf_counter()
    preload = cfg.get("preload", False)
    if preload:
        print("Preload: fold-local (per-fold cache, discarded after each fold)", flush=True)
    else:
        print("On-the-fly loading (H5BagDataset). Use --preload for fold-local cached mode.", flush=True)

    results = []
    for fold in folds:
        fold_start = time.perf_counter()
        print(f"\n{'=' * 60}", flush=True)
        print(f"FOLD {fold} START", flush=True)
        print(f"{'=' * 60}", flush=True)
        fold_train = splits[splits["fold"] != fold]["image_id"].tolist()
        fold_val = splits[splits["fold"] == fold]["image_id"].tolist()
        train_ids = [i for i in fold_train if i in available_h5 and i in train_labels.index]
        val_ids = [i for i in fold_val if i in available_h5 and i in train_labels.index]
        if not train_ids or not val_ids:
            print(f"  Skipping fold {fold}: empty train ({len(train_ids)}) or val ({len(val_ids)})", flush=True)
            continue
        print(f"  train: {len(train_ids)}, val: {len(val_ids)}", flush=True)

        # Fold-local preload: only this fold's slides, discard after run_fold
        cache: dict[str, torch.Tensor] | None = None
        coord_cache: dict[str, torch.Tensor | None] | None = None
        if preload:
            fold_slide_ids = list(dict.fromkeys(train_ids + val_ids))
            print(f"  Preloading {len(fold_slide_ids)} slides for this fold...", flush=True)
            cache, coord_cache = preload_features(feat_dir, fold_slide_ids)

        fold_cfg = {**cfg, "checkpoint_dir": checkpoint_dir}
        resume_path = cfg.get("resume_from")
        if resume_path is not None:
            # Only resume for the fold matching the checkpoint (e.g. fold0_best.pt -> fold 0)
            m = re.search(r"fold(\d+)[_\w]*\.pt", str(resume_path))
            if m and int(m.group(1)) == fold:
                fold_cfg["resume_from"] = Path(resume_path)
            else:
                fold_cfg["resume_from"] = None
        best = run_fold(
            fold=fold,
            train_ids=train_ids,
            val_ids=val_ids,
            feat_dir=feat_dir,
            train_labels=train_labels,
            feat_dim=feat_dim,
            cfg=fold_cfg,
            device=device,
            cache=cache,
            coord_cache=coord_cache,
        )
        best["fold"] = fold
        results.append(best)
        fold_time = time.perf_counter() - fold_start
        print(f"\n  FOLD {fold} END | walltime={fold_time:.1f}s ({fold_time / 60:.1f} min)", flush=True)
        del cache, coord_cache

    cv_time = time.perf_counter() - cv_start
    df = pd.DataFrame(results)
    print(f"\n{'=' * 60}", flush=True)
    print("CV SUMMARY", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"  Total walltime: {cv_time:.1f}s ({cv_time / 60:.1f} min)", flush=True)
    for col in ["qwk", "balanced_accuracy", "accuracy", "mae"]:
        if col in df.columns:
            print(f"  {col:20s}: {df[col].mean():.4f} ± {df[col].std():.4f}", flush=True)
    return df
