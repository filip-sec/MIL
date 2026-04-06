#!/usr/bin/env python3
"""
Inspect TRIDENT feature .h5 files: patch counts, small files, bad files.

Usage (on cluster):
  python scripts/inspect_trident_features.py
  python scripts/inspect_trident_features.py --features-dir /storage/brno2/home/filipsec/MIL/data/trident_out/panda_uni_v2_grandqc_20x_256_ov0/20x_256px_0px_overlap/features_uni_v2
  python scripts/inspect_trident_features.py --min-patches 10   # list slides with <= N patches

Shell equivalent for unique slide count only:
  find /storage/brno2/home/filipsec/MIL/data/trident_out -path "*/features_uni_v2/*.h5" -type f -exec basename {} .h5 \\; | sort -u | wc -l
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

try:
    import h5py
    import numpy as np
except ImportError:
    raise SystemExit("pip install h5py numpy")

def main() -> None:
    ap = argparse.ArgumentParser(description="Inspect TRIDENT feature .h5 files")
    ap.add_argument(
        "--features-dir",
        type=Path,
        default=Path("/storage/brno2/home/filipsec/MIL/data/trident_out"),
        help="Features dir or parent (searches **/features_uni_v2/*.h5)",
    )
    ap.add_argument(
        "--min-patches",
        type=int,
        default=None,
        help="Report slides with <= this many patches (default: no filter)",
    )
    ap.add_argument(
        "--small-size-kb",
        type=int,
        default=None,
        help="List files smaller than this many KB (default: no list)",
    )
    args = ap.parse_args()

    root = args.features_dir
    if root.is_dir() and (root / "features_uni_v2").exists():
        pattern = str(root / "features_uni_v2" / "*.h5")
    else:
        pattern = str(root / "**" / "features_uni_v2" / "*.h5")
    files = sorted(glob.glob(pattern))

    counts: list[tuple[str, int]] = []
    bad: list[tuple[str, str]] = []

    for f in files:
        try:
            with h5py.File(f, "r") as h:
                if "features" in h:
                    n = h["features"].shape[0]
                    counts.append((os.path.basename(f)[:-3], n))
                else:
                    bad.append((os.path.basename(f), "missing features key"))
        except Exception as e:
            bad.append((os.path.basename(f), str(e)))

    if not counts:
        print("No .h5 files with 'features' found.")
        if bad:
            print("Bad files:", bad)
        return

    unique_slides = len({c[0] for c in counts})
    arr = np.array([c[1] for c in counts])
    print("unique slide IDs (with .h5):", unique_slides)
    print("num .h5 files (can duplicate across run dirs):", len(counts))
    print("min:", arr.min())
    print("p1:", np.percentile(arr, 1))
    print("p5:", np.percentile(arr, 5))
    print("median:", np.median(arr))
    print("max:", arr.max())

    if args.min_patches is not None:
        low = [(sid, n) for sid, n in counts if n <= args.min_patches]
        low.sort(key=lambda x: x[1])
        print(f"\nSlides with <= {args.min_patches} patches:")
        for sid, n in low:
            print(" ", sid, n)

    if bad:
        print("\nBad files:")
        for name, msg in bad:
            print(" ", name, msg)

    if args.small_size_kb is not None:
        small = [(f, os.path.getsize(f)) for f in files if os.path.getsize(f) < args.small_size_kb * 1024]
        small.sort(key=lambda x: x[1])
        print(f"\nFiles < {args.small_size_kb} KB:")
        for path, size in small:
            print(" ", os.path.basename(path), size)


if __name__ == "__main__":
    main()
