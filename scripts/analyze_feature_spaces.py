#!/usr/bin/env python3
"""CLI for cross-encoder feature-space analysis."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mil.analysis.feature_space import AnalysisConfig, run_feature_space_analysis
from mil.config import merge_runtime_config


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare patch-level feature spaces across encoders")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional YAML config (defaults to configs/pipelines/feature_space_analysis.yaml)",
    )
    parser.add_argument("--preset", type=str, default=None, help="Optional preset from the config YAML")
    parser.add_argument("--out-dir", type=Path, default=None, help="Override output directory")
    parser.add_argument("--labels-csv", type=Path, default=None, help="Optional labels CSV for probe analysis")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--viz-points", type=int, default=None)
    parser.add_argument("--per-slide-patch-cap", type=int, default=None)
    parser.add_argument("--max-total-pairs", type=int, default=None)
    parser.add_argument("--disable-probe", action="store_true", help="Disable probe analysis")
    parser.add_argument("--disable-alignment", action="store_true", help="Disable cross-space alignment analysis")
    parser.add_argument("--disable-spatial", action="store_true", help="Disable spatial smoothness analysis")
    args = parser.parse_args()

    overrides = {
        "out_dir": args.out_dir,
        "labels_csv": args.labels_csv,
        "seed": args.seed,
        "viz_points": args.viz_points,
        "per_slide_patch_cap": args.per_slide_patch_cap,
        "max_total_pairs": args.max_total_pairs,
    }
    if args.disable_probe:
        overrides["run_probe"] = False
    if args.disable_alignment:
        overrides["run_alignment"] = False
    if args.disable_spatial:
        overrides["run_spatial"] = False
    overrides = {k: v for k, v in overrides.items() if v is not None}

    runtime = merge_runtime_config(
        "feature_space_analysis",
        config_path=args.config,
        overrides=overrides,
        preset=args.preset,
    )
    cfg = AnalysisConfig.from_mapping(runtime.values, root_dir=ROOT)
    artifacts = run_feature_space_analysis(cfg)

    print("Feature-space analysis completed.")
    print(json.dumps({k: str(v) for k, v in artifacts.items()}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
