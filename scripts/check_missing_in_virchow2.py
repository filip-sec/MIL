#!/usr/bin/env python3
"""
Check if slides from missing_21_slides.csv are also missing in Virchow2 extracted features.

Run on cluster (where Virchow2 features live):
  python scripts/check_missing_in_virchow2.py \\
    --missing-csv data/lists/missing_21_slides.csv \\
    --features-dir /storage/brno2/home/filipsec/MIL/data/trident_out/panda_virchow2_grandqc_20x_224_ov0/20x_224px_0px_overlap/features_virchow2
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


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Check if slides from missing CSV exist in Virchow2 features dir."
    )
    ap.add_argument("--missing-csv", type=Path, required=True, help="Path to missing_21_slides.csv")
    ap.add_argument(
        "--features-dir",
        type=Path,
        default=Path("/storage/brno2/home/filipsec/MIL/data/trident_out/panda_virchow2_grandqc_20x_224_ov0/20x_224px_0px_overlap/features_virchow2"),
        help="Path to Virchow2 features dir (dir of .h5)",
    )
    args = ap.parse_args()

    if not args.missing_csv.exists():
        raise SystemExit(f"Missing CSV not found: {args.missing_csv}")

    if not args.features_dir.is_dir():
        raise SystemExit(f"Features dir not found: {args.features_dir}")

    ids_from_csv: list[str] = []
    with args.missing_csv.open(newline="") as f:
        r = csv.DictReader(f)
        col = "wsi" if r.fieldnames and "wsi" in (r.fieldnames or []) else (r.fieldnames[0] if r.fieldnames else None)
        if not col:
            raise SystemExit("No column found in CSV")
        for row in r:
            val = row.get(col, "").strip()
            if val:
                ids_from_csv.append(norm_id(val))

    have = {f.stem for f in args.features_dir.glob("*.h5")}

    present = []
    missing = []
    for sid in ids_from_csv:
        if sid in have:
            present.append(sid)
        else:
            missing.append(sid)

    print(f"Checked {len(ids_from_csv)} slides from {args.missing_csv.name}")
    print(f"Virchow2 features dir: {args.features_dir}")
    print()
    print(f"PRESENT in Virchow2 ({len(present)}):")
    for sid in sorted(present):
        print(f"  {sid}")
    print()
    print(f"MISSING in Virchow2 ({len(missing)}):")
    for sid in sorted(missing):
        print(f"  {sid}")
    print()
    print(f"Summary: {len(present)} present, {len(missing)} missing in Virchow2")


if __name__ == "__main__":
    main()
