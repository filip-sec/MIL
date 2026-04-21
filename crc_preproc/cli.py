"""Command-line interface for batch WSI processing."""

import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .labels import build_labels


def main():
    """Main CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--wsi_dir", required=True)
    ap.add_argument("--pattern", default="*.svs")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--labels_csv", default="")
    ap.add_argument("--downsample", type=int, default=32)
    ap.add_argument("--tile_px", type=int, default=512)
    ap.add_argument("--relax", action="store_true")
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    try:
        from .runner import process_one
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(
            "WSI preprocessing dependencies are missing. Install with `pip install -e \".[preproc]\"`."
        ) from e

    wsi_dir, out_root = Path(args.wsi_dir), Path(args.out_dir)
    files = sorted(wsi_dir.glob(args.pattern))
    if not files:
        print(f"No files found for {args.pattern} in {wsi_dir}")
        return

    labels_map = build_labels(Path(args.labels_csv)) if args.labels_csv else {}
    total_tiles = 0

    for wsi in tqdm(files, desc="Slides"):
        try:
            sid, saved, lbl = process_one(
                wsi,
                out_root,
                labels_map,
                args.downsample,
                args.tile_px,
                strict=not args.relax,
                skip=args.skip_existing,
            )
            total_tiles += saved
            print(f"[OK] {sid}: {saved} tiles | label={'' if lbl is None else str(lbl)}")
        except (ImportError, OSError, ValueError, RuntimeError) as e:
            print(f"[ERR] {wsi.name}: {e}")

    labels_dir = out_root / "data" / "labels"
    per_csvs = sorted(labels_dir.glob("*_tiles.csv"))
    if per_csvs:
        idx = pd.concat([pd.read_csv(p) for p in per_csvs], ignore_index=True)
        idx.to_csv(labels_dir / "tiles_index.csv", index=False)
        idx.groupby("slide_id").size().reset_index(name="n_tiles").to_csv(labels_dir / "slides_index.csv", index=False)
        print(f"[DONE] {len(idx)} tiles from {len(idx['slide_id'].unique())} slides")
        print(f"[DONE] Total tiles saved in this run: {total_tiles}")


if __name__ == "__main__":
    main()
