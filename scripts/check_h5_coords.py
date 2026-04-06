#!/usr/bin/env python3
"""Quick check: do .h5 feature files contain coords? Required for attention map overlay."""
import argparse
from pathlib import Path

import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("feat_dir", type=Path, help="Features dir (e.g. .../features_virchow2)")
    ap.add_argument("-n", type=int, default=3, help="Check first N files")
    args = ap.parse_args()

    files = sorted(args.feat_dir.glob("*.h5"))[: args.n]
    if not files:
        print(f"No .h5 in {args.feat_dir}")
        return

    for p in files:
        with h5py.File(p, "r") as f:
            keys = list(f.keys())
            feats = f.get("features")
            n = feats.shape[0] if feats is not None else 0
            coords = f.get("coords")
            has_coords = coords is not None and coords.shape[0] == n
            print(f"{p.name}: keys={keys}, n_patches={n}, coords={'yes' if has_coords else 'NO'}")

    print("\nIf coords=NO: use --coords-dir pointing to 20x_*_overlap/ (patches in slide_id_patches.h5)")


if __name__ == "__main__":
    main()
