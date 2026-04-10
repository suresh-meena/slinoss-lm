"""Mamba-3 baseline integration for slinoss-lm."""

from .config import (
    Mamba3ExperimentConfig,
    Mamba3ModelConfig,
    config_to_dict,
    load_config,
)
from .model import Mamba3CausalLM

__all__ = [
    "Mamba3CausalLM",
    "Mamba3ExperimentConfig",
    "Mamba3ModelConfig",
    "config_to_dict",
    "load_config",
]
