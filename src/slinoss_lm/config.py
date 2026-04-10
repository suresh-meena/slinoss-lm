from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ModelConfig:
    vocab_size: int = 128256
    hidden_size: int = 512
    intermediate_size: int = 1536
    num_hidden_layers: int = 14
    d_state: int = 128
    expand: int = 2
    d_head: int = 64
    d_conv: int = 4
    chunk_size: int = 64
    dt_min: float = 1.0e-3
    dt_init_floor: float = 1.0e-3
    r_min: float = 0.8
    residual_in_fp32: bool = True
    mlp_multiple_of: int = 128
    rms_norm_eps: float = 1.0e-5
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = False

    def architecture_kwargs(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("gradient_checkpointing", None)
        return payload


@dataclass
class DataConfig:
    root: str | None = None
    root_env: str = "FWEDU_DATA_ROOT"
    dataset_dir: str = "data/fwedu-llama31-2k"
    seq_len: int = 2048
    tokenizer_name: str | None = "meta-llama/Llama-3.1-8B"
    tokenizer_path: str | None = None


@dataclass
class OptimConfig:
    peak_lr: float = 4.0e-4
    min_lr: float = 1.0e-5
    min_lr_ratio: float | None = None
    warmup_tokens: int = 1_000_000_000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1.0e-8
    grad_clip_norm: float = 1.0


@dataclass
class TrainConfig:
    seed: int = 1337
    target_tokens: int = 100_000_000_000
    precision: str = "bf16"
    allow_tf32: bool = True
    compile: bool = False
    compile_mode: str = "max-autotune-no-cudagraphs"
    max_runtime_seconds: int | None = None
    wall_clock_deadline_unix: int | None = None
    wall_clock_exit_margin_seconds: int = 900


@dataclass
class RuntimeConfig:
    per_device_batch_size: int = 16
    grad_accum_steps: int = 8
    dataloader_workers: int = 8
    prefetch_factor: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True
    ddp_static_graph: bool = False


@dataclass
class CheckpointConfig:
    run_root: str = "runs"
    save_every_steps: int = 5000
    save_every_minutes: int = 0
    keep_last: int = 3
    save_on_signal: bool = True


@dataclass
class LoggingConfig:
    log_every_steps: int = 10


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str | None = None
    entity: str | None = None
    run_name: str | None = None
    run_id: str | None = None
    group: str | None = None
    job_type: str = "train"
    tags: list[str] = field(default_factory=list)
    notes: str | None = None
    mode: str | None = None
    dir: str | None = None
    resume: str = "allow"
    upload_checkpoints: bool = False


@dataclass
class ValidationConfig:
    enabled: bool = False
    dataset_root: str | None = None
    dataset_dir: str | None = None
    start_sequence: int = 0
    num_sequences: int = 0
    batch_size: int = 8
    every_steps: int = 0


@dataclass
class EvalConfig:
    zero_shot_tasks: list[str] = field(
        default_factory=lambda: [
            "lambada_openai",
            "hellaswag",
            "piqa",
            "arc_easy",
            "arc_challenge",
            "winogrande",
            "openbookqa",
        ]
    )
    zero_shot_batch_size: str = "auto"


@dataclass
class ExperimentConfig:
    name: str = "fwedu-run"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def _set_dotted(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cur = cfg
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def load_config(
    paths: list[str], overrides: list[str] | None = None
) -> ExperimentConfig:
    merged: dict[str, Any] = asdict(ExperimentConfig())
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
    return ExperimentConfig(
        name=merged["name"],
        model=ModelConfig(**merged["model"]),
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


def config_to_dict(config: ExperimentConfig) -> dict[str, Any]:
    return asdict(config)
