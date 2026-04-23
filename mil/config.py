"""YAML-backed config loaders for encoder and pipeline defaults."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "configs"
PIPELINE_CONFIG_DIR = CONFIG_DIR / "pipelines"
ENCODER_CONFIG_PATH = CONFIG_DIR / "encoders.yaml"

PATH_KEYS = frozenset(
    {
        "attention_dir",
        "checkpoint_dir",
        "chunk_csv",
        "conda_env",
        "coords_dir",
        "emb_dir",
        "feat_dir",
        "job_dir",
        "labels_csv",
        "out_dir",
        "output",
        "python_bin",
        "repo_dir",
        "repo_root",
        "resume",
        "sample_submission",
        "slides_csv",
        "splits_csv",
        "train_csv",
        "wsi_dir",
    }
)
PATH_LIST_KEYS = frozenset({"checkpoint"})

BUILTIN_PIPELINE_DEFAULTS: dict[str, dict[str, Any]] = {
    "trident_feat_only_shared_coords": {
        "coords_dir": "20x_256px_0px_overlap",
        "job_dir": "/storage/brno2/home/filipsec/MIL/data/trident_out/panda_ensemble_grandqc_20x_256_ov0",
        "wsi_dir": "/storage/brno2/home/filipsec/MIL/data/raw/train_images",
        "repo_dir": "/storage/brno2/home/filipsec/repos/trident",
        "repo_root": "/storage/brno2/home/filipsec/MIL",
        "conda_env": "/storage/brno2/home/filipsec/.conda/envs/trident",
        "python_bin": "/storage/brno2/home/filipsec/.conda/envs/trident/bin/python",
        "wsi_ext": ".tiff",
        "mag": 20,
        "overlap": 0,
        "gpu": 0,
        "max_workers": 4,
        "cache_batch_size": 8,
        "patch_size": None,
        "feat_batch_size": None,
    },
    "mil_training": {
        "train_csv": "data/raw/train.csv",
        "splits_csv": "data/splits/panda_5fold_stratified.csv",
        "model": "attention",
        "hidden": None,
        "epochs": 30,
        "lr": 0.01,
        "batch_size": 8,
        "num_workers": 4,
        "warmup_epochs": 5,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "patience": 0,
        "patch_dropout": 0.0,
        "max_patches_train": 512,
        "max_patches_val": 512,
        "attention_dir": None,
        "ce_weight": 0.3,
        "top_k": 8,
        "clam_inst_weight": 0.7,
        "checkpoint_dir": "checkpoints",
        "resume": None,
        "coords_dir": None,
        "preload": False,
        "max_patches": None,
        "folds": None,
        "seed": 42,
    },
    "kaggle_predict": {
        "sample_submission": None,
        "output": "submission.csv",
        "key": "features",
        "device": "auto",
        "max_patches": None,
        "seed": 42,
        "default_grade": 0,
    },
    "gigapath_slide_embed": {
        "slides_csv": None,
        "coords_dir": None,
        "device": "auto",
        "repo_root": ".",
        "overwrite": False,
        "limit": None,
    },
    "gigapath_slide_train": {
        "train_csv": "data/raw/train.csv",
        "splits_csv": "data/splits/panda_5fold_stratified.csv",
        "checkpoint_dir": "checkpoints/gigapath_slide",
        "epochs": 30,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 64,
        "num_workers": 0,
        "patience": 10,
        "dropout": 0.25,
        "folds": None,
        "seed": 42,
        "device": "auto",
    },
    "feature_space_analysis": {
        "out_dir": "results/feature_space_analysis",
        "seed": 42,
        "per_slide_patch_cap": 512,
        "max_total_pairs": 120000,
        "use_coords_matching": True,
        "pairwise_projection": "pca",
        "pairwise_projection_dim": 256,
        "viz_methods": ["pca", "umap"],
        "viz_points": 20000,
        "viz_color_by": ["slide", "label"],
        "labels_csv": "data/raw/train.csv",
        "labels_id_col": "image_id",
        "labels_target_col": "isup_grade",
        "run_probe": True,
        "probe_model": "logreg",
        "probe_folds": 5,
        "run_svcca": True,
        "run_alignment": True,
        "alignment_train_fraction": 0.8,
        "alignment_alpha": 1.0,
        "run_spatial": True,
        "spatial_k": 5,
        "spatial_patch_cap": 256,
        "spatial_pair_sample": 2000,
    },
}


@dataclass(frozen=True)
class EncoderSpec:
    """Canonical config for one encoder family."""

    key: str
    display_name: str
    repo_id: str
    feature_dim: int
    feature_dir_name: str
    tile_input: str
    extraction_mode: str
    downstream_route: str
    token_files: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    gated: bool = False
    noncommercial: bool = False
    patch_size: int = 256
    feat_batch_size: int = 96
    notes: str = ""

    def matches_dir(self, path: Path | str) -> bool:
        return Path(path).name == self.feature_dir_name


@dataclass(frozen=True)
class EncoderRegistryConfig:
    """Loaded encoder registry metadata."""

    config_name: str
    config_path: Path
    encoders: tuple[EncoderSpec, ...]


@dataclass(frozen=True)
class PipelineConfig:
    """Loaded pipeline config defaults and presets."""

    pipeline_name: str
    config_name: str
    config_path: Path
    defaults: dict[str, Any]
    presets: dict[str, dict[str, Any]]


@dataclass(frozen=True)
class RuntimeConfig:
    """Merged runtime config for one pipeline."""

    pipeline_name: str
    config_name: str
    config_path: Path
    values: dict[str, Any]

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.values)


def _resolve_path(path: Path | str | None, default: Path) -> Path:
    candidate = default if path is None else Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping at the top level: {path}")
    return data


def _normalize_value(key: str, value: Any) -> Any:
    if value in ("", None):
        return None if value == "" else value
    if key in PATH_KEYS:
        path = Path(value)
        return path if path.is_absolute() else ROOT / path
    if key in PATH_LIST_KEYS and isinstance(value, list):
        paths: list[Path] = []
        for item in value:
            path = Path(item)
            paths.append(path if path.is_absolute() else ROOT / path)
        return paths
    return value


def _normalize_mapping(values: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _normalize_value(key, value) for key, value in values.items()}


@lru_cache(maxsize=None)
def load_encoder_registry_config(config_path: Path | str | None = None) -> EncoderRegistryConfig:
    path = _resolve_path(config_path, ENCODER_CONFIG_PATH)
    data = _load_yaml(path)
    raw_encoders = data.get("encoders")
    if not isinstance(raw_encoders, list) or not raw_encoders:
        raise ValueError(f"{path} must define a non-empty 'encoders' list")

    required = {
        "key",
        "display_name",
        "repo_id",
        "feature_dim",
        "feature_dir_name",
        "token_files",
        "aliases",
        "gated",
        "noncommercial",
        "patch_size",
        "feat_batch_size",
        "tile_input",
        "extraction_mode",
        "downstream_route",
    }
    encoders: list[EncoderSpec] = []
    seen: set[str] = set()
    for item in raw_encoders:
        if not isinstance(item, dict):
            raise ValueError(f"Encoder entries must be mappings in {path}")
        missing = sorted(required - set(item))
        if missing:
            raise ValueError(f"Encoder entry {item!r} is missing required keys: {missing}")
        spec = EncoderSpec(
            key=str(item["key"]),
            display_name=str(item["display_name"]),
            repo_id=str(item["repo_id"]),
            feature_dim=int(item["feature_dim"]),
            feature_dir_name=str(item["feature_dir_name"]),
            tile_input=str(item["tile_input"]),
            extraction_mode=str(item["extraction_mode"]),
            downstream_route=str(item["downstream_route"]),
            token_files=tuple(str(v) for v in item.get("token_files", [])),
            aliases=tuple(str(v) for v in item.get("aliases", [])),
            gated=bool(item.get("gated", False)),
            noncommercial=bool(item.get("noncommercial", False)),
            patch_size=int(item.get("patch_size", 256)),
            feat_batch_size=int(item.get("feat_batch_size", 96)),
            notes=str(item.get("notes", "")),
        )
        if spec.key in seen:
            raise ValueError(f"Duplicate encoder key {spec.key!r} in {path}")
        seen.add(spec.key)
        encoders.append(spec)

    return EncoderRegistryConfig(
        config_name=str(data.get("name") or path.stem),
        config_path=path,
        encoders=tuple(encoders),
    )


def default_pipeline_config_path(pipeline_name: str) -> Path:
    return PIPELINE_CONFIG_DIR / f"{pipeline_name}.yaml"


@lru_cache(maxsize=None)
def load_pipeline_config(pipeline_name: str, config_path: Path | str | None = None) -> PipelineConfig:
    path = _resolve_path(config_path, default_pipeline_config_path(pipeline_name))
    data = _load_yaml(path)

    declared = data.get("pipeline")
    if declared is not None and declared != pipeline_name:
        raise ValueError(f"{path} declares pipeline {declared!r}, expected {pipeline_name!r}")

    defaults = dict(BUILTIN_PIPELINE_DEFAULTS.get(pipeline_name, {}))
    yaml_defaults = data.get("defaults", {})
    if yaml_defaults and not isinstance(yaml_defaults, dict):
        raise ValueError(f"{path} 'defaults' must be a mapping")
    defaults.update(yaml_defaults or {})
    defaults = _normalize_mapping(defaults)

    raw_presets = data.get("presets", {})
    if raw_presets and not isinstance(raw_presets, dict):
        raise ValueError(f"{path} 'presets' must be a mapping")
    presets: dict[str, dict[str, Any]] = {}
    for name, values in (raw_presets or {}).items():
        if not isinstance(values, dict):
            raise ValueError(f"Preset {name!r} in {path} must be a mapping")
        presets[str(name)] = _normalize_mapping(values)

    return PipelineConfig(
        pipeline_name=pipeline_name,
        config_name=str(data.get("name") or path.stem),
        config_path=path,
        defaults=defaults,
        presets=presets,
    )


def merge_runtime_config(
    pipeline_name: str,
    *,
    config_path: Path | str | None = None,
    overrides: Mapping[str, Any] | None = None,
    preset: str | None = None,
) -> RuntimeConfig:
    pipeline = load_pipeline_config(pipeline_name, config_path=config_path)
    values = dict(pipeline.defaults)
    if preset is not None:
        try:
            values.update(pipeline.presets[preset])
        except KeyError as exc:
            raise KeyError(
                f"Unknown preset {preset!r} for pipeline {pipeline_name!r}. Known presets: {sorted(pipeline.presets)}"
            ) from exc
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                values[key] = _normalize_value(key, value)
    if pipeline_name == "trident_feat_only_shared_coords":
        coords_dir = values.get("coords_dir")
        if isinstance(coords_dir, Path):
            # TRIDENT expects the coords directory as a job-local subdirectory name,
            # not an absolute path to the patches folder.
            values["coords_dir"] = coords_dir.name
    return RuntimeConfig(
        pipeline_name=pipeline.pipeline_name,
        config_name=pipeline.config_name,
        config_path=pipeline.config_path,
        values=values,
    )
