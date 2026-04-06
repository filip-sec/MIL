#!/usr/bin/env python3
"""
Create 5-fold stratified split from PANDA train.csv (by isup_grade).
Run before full-dataset TRIDENT extraction so the split is fixed and reproducible.

Usage:
  python scripts/make_panda_5fold_splits.py
  python scripts/make_panda_5fold_splits.py --csv path/to/train.csv --out path/to/splits/panda_5fold.csv
"""

import argparse
import os

import pandas as pd
from sklearn.model_selection import StratifiedKFold


def main():
    p = argparse.ArgumentParser(description="5-fold stratified split for PANDA (isup_grade)")
    p.add_argument(
        "--csv",
        default=os.path.expanduser("/storage/brno2/home/filipsec/MIL/data/raw/train.csv"),
        help="Path to train.csv",
    )
    p.add_argument(
        "--out",
        default=os.path.expanduser("/storage/brno2/home/filipsec/MIL/data/splits/panda_5fold_stratified.csv"),
        help="Output CSV path",
    )
    p.add_argument("--n_splits", type=int, default=5, help="Number of folds")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    df = pd.read_csv(args.csv).copy()
    if "fold" in df.columns:
        df = df.drop(columns=["fold"])
    df["fold"] = -1

    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    for fold, (_, val_idx) in enumerate(skf.split(df, df["isup_grade"])):
        df.loc[val_idx, "fold"] = fold

    df.to_csv(args.out, index=False)

    print("saved:", args.out)
    print(df["fold"].value_counts().sort_index())
    print(pd.crosstab(df["fold"], df["isup_grade"]))


if __name__ == "__main__":
    main()
