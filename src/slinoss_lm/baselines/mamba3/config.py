from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ...config import (
    CheckpointConfig,
    DataConfig,
    EvalConfig,
    LoggingConfig,
    OptimConfig,
    RuntimeConfig,
    TrainConfig,
    ValidationConfig,
    WandbConfig,
    _merge_dict,
    _set_dotted,
)


@dataclass
class Mamba3ModelConfig:
    vocab_size: int = 128256
    d_model: int = 768
    d_intermediate: int = 1500
    n_layer: int = 11
    d_state: int = 128
    expand: int = 2
    headdim: int = 64
    chunk_size: int = 64
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True
    initializer_range: float = 0.02
    is_mimo: bool = False
    mimo_rank: int = 4

    def architecture_kwargs(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Mamba3ExperimentConfig:
    name: str = "mamba3-siso-180m"
    model: Mamba3ModelConfig = field(default_factory=Mamba3ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def load_config(
    paths: list[str], overrides: list[str] | None = None
) -> Mamba3ExperimentConfig:
    merged: dict[str, Any] = asdict(Mamba3ExperimentConfig())
    for path in paths:
        payload = yaml.safe_load(Path(path).read_text()) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"Config {path} must contain a mapping at the top level.")
        merged = _merge_dict(merged, payload)
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Override must be key=value, got {override!r}.")
        key, raw_value = override.split("=", 1)
        value = yaml.safe_load(raw_value)
        _set_dotted(merged, key, value)
    return Mamba3ExperimentConfig(
        name=merged["name"],
        model=Mamba3ModelConfig(**merged["model"]),
        data=DataConfig(**merged["data"]),
        optim=OptimConfig(**merged["optim"]),
        train=TrainConfig(**merged["train"]),
        runtime=RuntimeConfig(**merged["runtime"]),
        checkpoint=CheckpointConfig(**merged["checkpoint"]),
        logging=LoggingConfig(**merged["logging"]),
        wandb=WandbConfig(**merged["wandb"]),
        validation=ValidationConfig(**merged["validation"]),
        eval=EvalConfig(**merged["eval"]),
    )


def config_to_dict(config: Mamba3ExperimentConfig) -> dict[str, Any]:
    return asdict(config)
