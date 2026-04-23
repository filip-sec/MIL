"""Encoder registry and Hugging Face auth helpers for pathology feature extractors."""

from __future__ import annotations

from os import environ
from pathlib import Path
from typing import Iterable, Mapping

from .config import EncoderSpec, load_encoder_registry_config


def _registry(config_path: Path | str | None = None):
    return load_encoder_registry_config(config_path)


def _alias_map(config_path: Path | str | None = None) -> dict[str, str]:
    return {
        alias.strip().lower(): spec.key
        for spec in _registry(config_path).encoders
        for alias in (spec.key, *spec.aliases)
    }


ENCODER_SPECS: tuple[EncoderSpec, ...] = _registry().encoders


def canonicalize_encoder_name(name: str, config_path: Path | str | None = None) -> str:
    """Return the canonical registry key for an encoder name or alias."""
    aliases = _alias_map(config_path)
    try:
        return aliases[name.strip().lower()]
    except KeyError as exc:
        known = sorted({spec.key for spec in _registry(config_path).encoders})
        raise KeyError(f"Unknown encoder '{name}'. Known keys: {known}") from exc


def get_encoder_spec(name: str, config_path: Path | str | None = None) -> EncoderSpec:
    """Lookup by canonical name or alias."""
    key = canonicalize_encoder_name(name, config_path=config_path)
    for spec in _registry(config_path).encoders:
        if spec.key == key:
            return spec
    raise KeyError(f"Unknown encoder '{name}'")


def list_encoder_specs(config_path: Path | str | None = None) -> tuple[EncoderSpec, ...]:
    """Return all registered encoder specs."""
    return _registry(config_path).encoders


def infer_encoder_from_feature_dir(path: Path | str, config_path: Path | str | None = None) -> str | None:
    """Infer encoder key from the feature directory basename."""
    name = Path(path).name
    for spec in _registry(config_path).encoders:
        if name == spec.feature_dir_name:
            return spec.key
    return None


def resolve_hf_token(
    repo_root: Path | str,
    encoder_name: str,
    env: Mapping[str, str] | None = None,
    config_path: Path | str | None = None,
) -> tuple[str, str]:
    """Resolve HF token using repo-local files first, then ``.hf_token``, then ``HF_TOKEN`` env."""
    spec = get_encoder_spec(encoder_name, config_path=config_path)
    root = Path(repo_root)
    env_map = environ if env is None else env

    for token_file in spec.token_files:
        path = root / token_file
        if path.exists():
            token = path.read_text(encoding="utf-8").strip()
            if token:
                return token, str(path)

    fallback = root / ".hf_token"
    if fallback.exists():
        token = fallback.read_text(encoding="utf-8").strip()
        if token:
            return token, str(fallback)

    token = env_map.get("HF_TOKEN", "").strip()
    if token:
        return token, "HF_TOKEN"

    file_msg = ", ".join(spec.token_files) if spec.token_files else "(no encoder-specific token file)"
    raise FileNotFoundError(
        f"No Hugging Face token found for encoder '{spec.key}'. Checked {file_msg}, .hf_token, then HF_TOKEN."
    )


def describe_encoder_table(
    keys: Iterable[str] | None = None,
    config_path: Path | str | None = None,
) -> list[dict[str, str]]:
    """Return a docs-friendly table payload."""
    selected = [get_encoder_spec(k, config_path=config_path) for k in keys] if keys is not None else list(
        _registry(config_path).encoders
    )
    rows = []
    for spec in selected:
        token_req = "encoder file -> .hf_token -> HF_TOKEN" if spec.token_files else ".hf_token -> HF_TOKEN"
        if spec.gated:
            token_req += " (access required)"
        rows.append(
            {
                "key": spec.key,
                "display_name": spec.display_name,
                "repo_id": spec.repo_id,
                "feature_dim": str(spec.feature_dim),
                "feature_dir_name": spec.feature_dir_name,
                "token_requirement": token_req,
                "extraction_mode": spec.extraction_mode,
                "downstream_route": spec.downstream_route,
            }
        )
    return rows
