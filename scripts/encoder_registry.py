#!/usr/bin/env python3
"""Inspect encoder registry entries and validate Hugging Face access."""

from __future__ import annotations

import argparse
import os
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mil.config import load_encoder_registry_config
from mil.encoders import describe_encoder_table, get_encoder_spec, list_encoder_specs, resolve_hf_token


def _shell_quote(value: str) -> str:
    return shlex.quote(value)


def _cmd_exports(args: argparse.Namespace) -> int:
    spec = get_encoder_spec(args.encoder, config_path=args.config)
    lines = [
        f"ENCODER_KEY={_shell_quote(spec.key)}",
        f"ENCODER_DISPLAY_NAME={_shell_quote(spec.display_name)}",
        f"ENCODER_REPO_ID={_shell_quote(spec.repo_id)}",
        f"ENCODER_FEATURE_DIM={spec.feature_dim}",
        f"ENCODER_FEATURE_DIR_NAME={_shell_quote(spec.feature_dir_name)}",
        f"ENCODER_TILE_INPUT={_shell_quote(spec.tile_input)}",
        f"ENCODER_EXTRACTION_MODE={_shell_quote(spec.extraction_mode)}",
        f"ENCODER_DOWNSTREAM_ROUTE={_shell_quote(spec.downstream_route)}",
        f"ENCODER_GATED={1 if spec.gated else 0}",
        f"ENCODER_NONCOMMERCIAL={1 if spec.noncommercial else 0}",
        f"ENCODER_PATCH_SIZE={spec.patch_size}",
        f"ENCODER_FEAT_BATCH_SIZE={spec.feat_batch_size}",
        f"ENCODER_TOKEN_FILES={_shell_quote(' '.join(spec.token_files))}",
    ]
    print("\n".join(lines))
    return 0


def _cmd_resolve_token(args: argparse.Namespace) -> int:
    token, _source = resolve_hf_token(args.repo_root, args.encoder, env=os.environ, config_path=args.config)
    print(token)
    return 0


def _cmd_table(args: argparse.Namespace) -> int:
    rows = describe_encoder_table(config_path=args.config)
    print("| key | repo | dim | feature dir | token requirement | extraction | downstream |")
    print("|---|---|---:|---|---|---|---|")
    for row in rows:
        print(
            f"| {row['key']} | {row['repo_id']} | {row['feature_dim']} | {row['feature_dir_name']} | "
            f"{row['token_requirement']} | {row['extraction_mode']} | {row['downstream_route']} |"
        )
    return 0


def _cmd_check_access(args: argparse.Namespace) -> int:
    spec = get_encoder_spec(args.encoder, config_path=args.config)
    token, source = resolve_hf_token(args.repo_root, args.encoder, env=os.environ, config_path=args.config)
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("check-access requires huggingface_hub in the current environment") from exc

    probe_filenames = ["config.json"]
    if spec.key == "openmidnight":
        probe_filenames = ["teacher_checkpoint_load.pt", "config.json"]

    last_exc: Exception | None = None
    api = HfApi()
    for filename in probe_filenames:
        try:
            exists = api.file_exists(repo_id=spec.repo_id, filename=filename, token=token)
            if not exists:
                raise FileNotFoundError(f"{filename} not found in {spec.repo_id}")
            print(f"Access OK for {spec.repo_id} via {source}; found {filename}")
            return 0
        except Exception as exc:  # noqa: BLE001 - surface the HF message to shell users
            last_exc = exc

    assert last_exc is not None  # defensive: loop must set this on failure
    root_exc = last_exc.__cause__ or last_exc.__context__ or last_exc
    msg = str(root_exc)
    hint = ""
    if spec.gated:
        hint = (
            f" Confirm that the token source '{source}' is valid and that you accepted access for {spec.repo_id}."
        )
    raise SystemExit(f"Access check failed for {spec.repo_id}: {msg}.{hint}") from last_exc



def main() -> int:
    parser = argparse.ArgumentParser(description="MIL encoder registry helper")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional encoder registry YAML (defaults to configs/encoders.yaml)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("exports", parents=[common], help="Print shell-friendly exports for one encoder")
    p.add_argument("--encoder", required=True)
    p.set_defaults(func=_cmd_exports)

    p = sub.add_parser("resolve-token", parents=[common], help="Resolve a token using repo-local lookup rules")
    p.add_argument("--encoder", required=True)
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.set_defaults(func=_cmd_resolve_token)

    p = sub.add_parser("table", parents=[common], help="Print a markdown table for docs")
    p.set_defaults(func=_cmd_table)

    p = sub.add_parser("check-access", parents=[common], help="Verify Hugging Face access for one encoder")
    p.add_argument("--encoder", required=True)
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.set_defaults(func=_cmd_check_access)

    p = sub.add_parser("list", parents=[common], help="List canonical encoder keys")
    p.set_defaults(
        func=lambda args: print(
            "\n".join(spec.key for spec in list_encoder_specs(config_path=args.config))
        )
        or 0
    )

    args = parser.parse_args()
    load_encoder_registry_config(args.config)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
