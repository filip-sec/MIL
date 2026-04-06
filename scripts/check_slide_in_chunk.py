#!/usr/bin/env python3
"""
Find 0-based index and cache batch of a slide in a TRIDENT chunk CSV.
TRIDENT uses cache_batch_size=8, so batch_i = index // 8.

Usage on cluster:
  python scripts/check_slide_in_chunk.py \\
    --chunk /storage/brno2/home/filipsec/MIL/data/lists/chunks_500/panda_chunk_001.csv \\
    --slide 163fabe883fcd17de4f899ca8ede45a8
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def norm_id(raw: str) -> str:
    s = (raw or "").strip()
    for suf in (".tiff", ".tif"):
        if s.lower().endswith(suf):
            return s[: -len(suf)]
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description="Find slide index and batch in chunk CSV.")
    ap.add_argument("--chunk", required=True, type=Path, help="Path to chunk CSV (e.g. panda_chunk_001.csv)")
    ap.add_argument("--slide", required=True, help="Slide ID (with or without .tiff)")
    ap.add_argument("--batch-size", type=int, default=8, help="TRIDENT cache_batch_size (default 8)")
    args = ap.parse_args()

    target = norm_id(args.slide)
    path = args.chunk
    if not path.exists():
        raise SystemExit(f"Chunk CSV not found: {path}")

    ids: list[str] = []
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        col = "wsi" if r.fieldnames and "wsi" in r.fieldnames else (r.fieldnames[0] if r.fieldnames else None)
        if not col:
            raise SystemExit("No column in CSV")
        for row in r:
            val = row.get(col, "").strip()
            if not val:
                continue
            ids.append(norm_id(val))

    try:
        idx = ids.index(target)
    except ValueError:
        print(f"Slide {target!r} not found in {path}")
        print(f"Chunk has {len(ids)} slides (0-based indices 0..{len(ids)-1}).")
        return

    batch_size = args.batch_size
    batch_i = idx // batch_size
    print(f"Chunk: {path.name}")
    print(f"Slide: {target}")
    print(f"Index (0-based): {idx} of {len(ids)}")
    print(f"Batch (0-based, batch_size={batch_size}): {batch_i}")
    print(f"  -> Matches TRIDENT '[PROCESSOR] Found N valid slides in .../batch_{batch_i}' and any warning logged during that batch.")


if __name__ == "__main__":
    main()
