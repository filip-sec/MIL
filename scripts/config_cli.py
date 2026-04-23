#!/usr/bin/env python3
"""Inspect YAML-backed pipeline configs and emit shell-friendly exports."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mil.config import PIPELINE_CONFIG_DIR, merge_runtime_config


def _shell_quote(value: str) -> str:
    return shlex.quote(value)


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value)
    return str(value)


def _cmd_exports(args: argparse.Namespace) -> int:
    runtime = merge_runtime_config(args.pipeline, config_path=args.config, preset=args.preset)
    lines = [
        f"CONFIG_NAME={_shell_quote(runtime.config_name)}",
        f"CONFIG_PATH={_shell_quote(str(runtime.config_path))}",
    ]
    for key, value in sorted(runtime.values.items()):
        shell_key = f"CFG_{key.upper()}"
        lines.append(f"{shell_key}={_shell_quote(_stringify(value))}")
    print("\n".join(lines))
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    runtime = merge_runtime_config(args.pipeline, config_path=args.config, preset=args.preset)
    payload = {
        "pipeline_name": runtime.pipeline_name,
        "config_name": runtime.config_name,
        "config_path": str(runtime.config_path),
        "values": {key: _stringify(value) for key, value in runtime.values.items()},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="MIL pipeline config helper")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("exports", help="Print shell-friendly exports for one pipeline")
    p.add_argument("--pipeline", required=True)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--preset", default=None)
    p.set_defaults(func=_cmd_exports)

    p = sub.add_parser("show", help="Print merged config as JSON")
    p.add_argument("--pipeline", required=True)
    p.add_argument("--config", type=Path, default=None)
    p.add_argument("--preset", default=None)
    p.set_defaults(func=_cmd_show)

    p = sub.add_parser("list", help="List default pipeline config names")
    p.set_defaults(
        func=lambda _args: print(
            "\n".join(sorted(path.stem for path in PIPELINE_CONFIG_DIR.glob("*.yaml")))
        )
        or 0
    )

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
