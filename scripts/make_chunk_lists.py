#!/usr/bin/env python3
"""
Create chunk CSV files from train.csv for TRIDENT batch extraction.
Each chunk has at most CHUNK_SIZE slide IDs; column "wsi" with values "<image_id>.tiff".

Usage (on cluster or locally, from MIL project root):
  python scripts/make_chunk_lists.py
  python scripts/make_chunk_lists.py --csv data/raw/train.csv --out-dir data/lists/chunks_500 --chunk-size 500

On cluster with absolute paths:
  python scripts/make_chunk_lists.py --csv /storage/brno2/home/filipsec/MIL/data/raw/train.csv --out-dir /storage/brno2/home/filipsec/MIL/data/lists/chunks_500
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    p = argparse.ArgumentParser(description="Create chunk CSVs from train.csv for TRIDENT")
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("data/raw/train.csv"),
        help="Path to train.csv (must have image_id column)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/lists/chunks_500"),
        help="Output directory for chunk CSVs",
    )
    p.add_argument("--chunk-size", type=int, default=500, help="Max slides per chunk")
    p.add_argument("--prefix", type=str, default="panda_chunk", help="Chunk file prefix")
    args = p.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"train.csv not found: {args.csv}")

    df = pd.read_csv(args.csv)
    if "image_id" not in df.columns:
        raise SystemExit("train.csv must have 'image_id' column")

    ids = df["image_id"].astype(str).tolist()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_chunks = (len(ids) + args.chunk_size - 1) // args.chunk_size
    for i in range(n_chunks):
        start = i * args.chunk_size
        end = min(start + args.chunk_size, len(ids))
        chunk_ids = ids[start:end]
        out_path = args.out_dir / f"{args.prefix}_{i:03d}.csv"
        chunk_df = pd.DataFrame({"wsi": [f"{sid}.tiff" for sid in chunk_ids]})
        chunk_df.to_csv(out_path, index=False)
        print(out_path, len(chunk_df))

    print(f"Created {n_chunks} chunk(s) in {args.out_dir}")


if __name__ == "__main__":
    main()
