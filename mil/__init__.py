"""MIL – Multiple Instance Learning for PANDA ISUP grading.

Submodules provide the API: ``mil.model``, ``mil.data``, ``mil.train``, ``mil.loss``,
``mil.utils``, ``mil.encoders``, and ``mil.config``.
"""

__version__ = "0.1.0"

from .config import (
    EncoderRegistryConfig,
    EncoderSpec,
    PipelineConfig,
    RuntimeConfig,
    load_encoder_registry_config,
    load_pipeline_config,
    merge_runtime_config,
)
from .encoders import get_encoder_spec, infer_encoder_from_feature_dir, list_encoder_specs

__all__ = [
    "__version__",
    "EncoderRegistryConfig",
    "EncoderSpec",
    "PipelineConfig",
    "RuntimeConfig",
    "get_encoder_spec",
    "infer_encoder_from_feature_dir",
    "load_encoder_registry_config",
    "load_pipeline_config",
    "list_encoder_specs",
    "merge_runtime_config",
]
