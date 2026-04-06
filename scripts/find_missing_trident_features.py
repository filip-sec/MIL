#!/usr/bin/env python3
"""
Find slide IDs that are in chunk CSV(s) but have no corresponding .h5 in the features dir.

List/count .h5 on cluster:
  FEAT_DIR="/storage/brno2/home/filipsec/MIL/data/trident_out/panda_uni_v2_grandqc_20x_256_ov0/20x_256px_0px_overlap/features_uni_v2"
  ls -1 "$FEAT_DIR"/*.h5 | wc -l
  ls -1 "$FEAT_DIR"/*.h5 | xargs -I{} basename {} .h5 | sort > feat_ids.txt

List .h5 IDs with this script (from MIL dir):
  python scripts/find_missing_trident_features.py --features-dir "$FEAT_DIR" --list-only

Find missing (run from MIL project dir on cluster):
  python scripts/find_missing_trident_features.py \\
    --chunks /storage/brno2/home/filipsec/MIL/data/lists/chunks_500/panda_chunk_000.csv \\
             /storage/brno2/home/filipsec/MIL/data/lists/chunks_500/panda_chunk_001.csv \\
    --features-dir /storage/brno2/home/filipsec/MIL/data/trident_out/panda_uni_v2_grandqc_20x_256_ov0/20x_256px_0px_overlap/features_uni_v2
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def norm_id(raw: str) -> str:
    """Normalize to slide ID (no .tiff/.tif)."""
    s = (raw or "").strip()
    for suf in (".tiff", ".tif"):
        if s.lower().endswith(suf):
            return s[: -len(suf)]
    return s


def ids_from_chunk_csv(path: Path) -> set[str]:
    out: set[str] = set()
    with path.open(newline="") as f:
        r = csv.DictReader(f)
        col = "wsi" if r.fieldnames and "wsi" in r.fieldnames else (r.fieldnames[0] if r.fieldnames else None)
        if not col:
            return out
        for row in r:
            val = row.get(col, "").strip()
            if not val:
                continue
            out.add(norm_id(val))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Find slides in chunk CSVs with no .h5 feature file.")
    ap.add_argument("--chunks", nargs="+", help="Paths to chunk CSV(s)")
    ap.add_argument("--features-dir", type=Path, required=True, help="Path to features_uni_v2 (dir of .h5)")
    ap.add_argument("--list-only", action="store_true", help="Only list .h5 basenames (one per line), then exit")
    args = ap.parse_args()

    feat_dir = args.features_dir
    if not feat_dir.is_dir():
        raise SystemExit(f"Features dir not found: {feat_dir}")

    if args.list_only:
        for p in sorted(feat_dir.glob("*.h5")):
            print(p.stem)
        return

    if not args.chunks:
        ap.error("--chunks required when not using --list-only")

    expected: set[str] = set()
    for p in args.chunks:
        path = Path(p)
        if not path.exists():
            raise SystemExit(f"Chunk CSV not found: {path}")
        expected |= ids_from_chunk_csv(path)

    have = {f.stem for f in feat_dir.glob("*.h5")}

    missing = expected - have
    extra = have - expected  # in case you care

    print(f"Expected (from {len(args.chunks)} chunk(s)): {len(expected)}")
    print(f"In dir (unique .h5): {len(have)}")
    print(f"Missing (in chunks, no .h5): {len(missing)}")
    if missing:
        for sid in sorted(missing):
            print(sid)
    if extra and len(extra) <= 20:
        print(f"Extra (in dir but not in chunks): {len(extra)}")
        for sid in sorted(extra):
            print(f"  {sid}")
    elif extra:
        print(f"Extra (in dir but not in chunks): {len(extra)}")


if __name__ == "__main__":
    main()
