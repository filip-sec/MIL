#!/usr/bin/env python3
"""Print the four standard WSI reporting lines (level 0, seg level, downsample, MPP).

Examples::

    python scripts/print_wsi_geometry.py /path/to/slide.tiff
    python scripts/print_wsi_geometry.py slide.tiff --seg-level 2
    python scripts/print_wsi_geometry.py slide.tiff --seg-level -1 --lang sk --json

``--seg-level -1`` means the coarsest pyramid level. Set ``--seg-level`` to the index your
segmentation pipeline actually uses (TRIDENT / grandqc / etc.).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description="Print WSI level-0 / seg-level / MPP report")
    ap.add_argument("wsi", type=Path, help="Path to WSI (.tiff, .svs, ...)")
    ap.add_argument(
        "--seg-level",
        type=int,
        default=-1,
        help="Pyramid level L_s (-1 = coarsest). Must match your segmentation pipeline.",
    )
    ap.add_argument("--lang", choices=("en", "sk"), default="en")
    ap.add_argument("--json", action="store_true", help="Also print one JSON object to stdout (after text)")
    args = ap.parse_args()

    p = args.wsi.expanduser()
    if not p.is_file():
        print(f"Not found: {p}", file=sys.stderr)
        raise SystemExit(1)

    repo = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo))
    from mil.wsi_metadata import format_wsi_geometry_lines, wsi_geometry_from_path

    try:
        geom = wsi_geometry_from_path(p, seg_level=args.seg_level)
    except Exception as e:
        print(f"Failed to open or parse slide: {e}", file=sys.stderr)
        raise SystemExit(1) from e

    print(format_wsi_geometry_lines(geom, lang=args.lang))
    if args.json:
        print(json.dumps(geom.as_dict(), indent=2))


if __name__ == "__main__":
    main()
