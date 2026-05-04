#!/usr/bin/env python3
"""Prepare SICAPv2 TRIDENT chunks and submit extraction jobs for four encoders.

Uses the existing PBS workflow in ``scripts/trident_chunk_extract.pbs`` with the
same locked PANDA extraction settings:
- segmenter=grandqc
- mag=20
- patch_size=256
- overlap=0
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path


DEFAULT_ENCODERS = ("uni_v2", "gigapath", "hoptimus0", "hoptimus1")
DEFAULT_MIL_ROOT = Path("/storage/brno2/home/filipsec/MIL")
DEFAULT_WSI_DIR = DEFAULT_MIL_ROOT / "data" / "SICAPv2" / "images"
DEFAULT_CHUNKS_ROOT = DEFAULT_MIL_ROOT / "data" / "lists" / "chunks" / "sicapv2"
DEFAULT_OUTPUT_ROOT = DEFAULT_MIL_ROOT / "data" / "trident_out"
DEFAULT_PBS = DEFAULT_MIL_ROOT / "scripts" / "trident_chunk_extract.pbs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract SICAPv2 TRIDENT features with PANDA-matched settings for four encoders."
    )
    p.add_argument("--wsi-dir", type=Path, default=DEFAULT_WSI_DIR, help="Directory with SICAPv2 WSIs.")
    p.add_argument(
        "--wsi-glob",
        action="append",
        default=["*.tiff", "*.tif"],
        help="Glob(s) used to discover WSI files (repeatable).",
    )
    p.add_argument("--wsi-ext", type=str, default=".tiff", help="TRIDENT --wsi_ext value (default: .tiff).")
    p.add_argument("--chunk-size", type=int, default=400, help="Slides per chunk CSV.")
    p.add_argument("--chunks-root", type=Path, default=DEFAULT_CHUNKS_ROOT, help="Where chunk CSVs are created.")
    p.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Parent dir for per-encoder TRIDENT output trees.",
    )
    p.add_argument(
        "--encoders",
        nargs="+",
        default=list(DEFAULT_ENCODERS),
        help="Encoder keys (e.g. uni_v2 gigapath hoptimus0 hoptimus1).",
    )
    p.add_argument(
        "--pbs-script",
        type=Path,
        default=DEFAULT_PBS,
        help="PBS script to submit (default: scripts/trident_chunk_extract.pbs).",
    )
    p.add_argument(
        "--submit",
        action="store_true",
        help="Submit qsub jobs (without this flag, only chunk CSVs are created and commands are printed).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing chunk CSVs under --chunks-root.",
    )
    return p.parse_args()


def discover_slides(wsi_dir: Path, patterns: list[str]) -> list[str]:
    slides: set[str] = set()
    for pat in patterns:
        for p in wsi_dir.glob(pat):
            if p.is_file():
                slides.add(p.name)
    return sorted(slides)


def write_chunks(slides: list[str], chunk_size: int, chunks_dir: Path, force: bool) -> list[Path]:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    existing = list(chunks_dir.glob("chunk_*.csv"))
    if existing and not force:
        raise SystemExit(f"{chunks_dir} already contains chunk_*.csv (use --force to overwrite).")
    for p in existing:
        p.unlink()

    chunk_paths: list[Path] = []
    for idx in range(0, len(slides), chunk_size):
        chunk_id = idx // chunk_size + 1
        chunk_path = chunks_dir / f"chunk_{chunk_id:03d}.csv"
        with chunk_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["wsi"])
            for name in slides[idx : idx + chunk_size]:
                w.writerow([name])
        chunk_paths.append(chunk_path)
    return chunk_paths


def encoder_job_dir(output_root: Path, encoder: str) -> Path:
    return output_root / f"sicapv2_{encoder}_grandqc_20x_256_ov0"


def build_qsub_command(
    *,
    pbs_script: Path,
    chunk_csv: Path,
    encoder: str,
    job_dir: Path,
    wsi_dir: Path,
    wsi_ext: str,
) -> list[str]:
    qsub_vars = ",".join(
        [
            f"CHUNK_CSV={chunk_csv}",
            f"ENCODER={encoder}",
            f"JOB_DIR={job_dir}",
            f"WSI_DIR={wsi_dir}",
            f"WSI_EXT={wsi_ext}",
        ]
    )
    return ["qsub", "-v", qsub_vars, str(pbs_script)]


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise SystemExit("--chunk-size must be > 0")
    if not args.wsi_dir.is_dir():
        raise SystemExit(f"WSI dir not found: {args.wsi_dir}")
    if not args.pbs_script.is_file():
        raise SystemExit(f"PBS script not found: {args.pbs_script}")

    slides = discover_slides(args.wsi_dir, args.wsi_glob)
    if not slides:
        raise SystemExit(f"No slides found in {args.wsi_dir} for patterns {args.wsi_glob}")

    chunks_dir = args.chunks_root.resolve()
    chunk_paths = write_chunks(slides, args.chunk_size, chunks_dir, args.force)
    print(f"Discovered {len(slides)} SICAPv2 slides")
    print(f"Wrote {len(chunk_paths)} chunk CSVs to {chunks_dir}")

    total_jobs = len(chunk_paths) * len(args.encoders)
    print(f"Planned jobs: {total_jobs} ({len(args.encoders)} encoders x {len(chunk_paths)} chunks)")

    for encoder in args.encoders:
        job_dir = encoder_job_dir(args.output_root.resolve(), encoder)
        print(f"\nEncoder: {encoder}")
        print(f"Output:  {job_dir}")
        for chunk_csv in chunk_paths:
            cmd = build_qsub_command(
                pbs_script=args.pbs_script.resolve(),
                chunk_csv=chunk_csv.resolve(),
                encoder=encoder,
                job_dir=job_dir,
                wsi_dir=args.wsi_dir.resolve(),
                wsi_ext=args.wsi_ext,
            )
            print(" ".join(cmd))
            if args.submit:
                subprocess.run(cmd, check=True)

    if not args.submit:
        print("\nDry run complete. Re-run with --submit to actually enqueue PBS jobs.")


if __name__ == "__main__":
    main()
