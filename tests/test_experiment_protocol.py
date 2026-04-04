from __future__ import annotations

from pathlib import Path

import pytest

from slinoss_lm.common import tokens_per_step
from slinoss_lm.config import ExperimentConfig, load_config
from slinoss_lm.train import CosineSchedule


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "configs" / "experiments"


def _iter_experiment_configs() -> list[tuple[Path, ExperimentConfig]]:
    out: list[tuple[Path, ExperimentConfig]] = []
    for path in sorted(EXPERIMENT_DIR.glob("*.yaml")):
        out.append((path, load_config([str(path)])))
    return out


def test_all_experiment_configs_load() -> None:
    loaded = _iter_experiment_configs()
    assert [cfg.name for _, cfg in loaded] == [
        "fwedu-180m",
        "fwedu-1p5b",
        "fwedu-440m",
        "fwedu-880m",
    ]


def test_lm_protocol_is_locked() -> None:
    expected_peak_lr = {
        "fwedu-180m": 4.0e-4,
        "fwedu-440m": 3.0e-4,
        "fwedu-880m": 2.5e-4,
        "fwedu-1p5b": 2.0e-4,
    }
    expected_save_every_minutes = {
        "fwedu-180m": 0,
        "fwedu-440m": 0,
        "fwedu-880m": 30,
        "fwedu-1p5b": 30,
    }
    expected_tasks = [
        "lambada_openai",
        "hellaswag",
        "piqa",
        "arc_easy",
        "arc_challenge",
        "winogrande",
        "openbookqa",
    ]

    for _, cfg in _iter_experiment_configs():
        assert cfg.data.dataset_dir == "data/fwedu-llama31-2k"
        assert cfg.data.seq_len == 2048
        assert cfg.data.tokenizer_name == "meta-llama/Llama-3.1-8B"

        assert cfg.model.vocab_size == 128256
        assert cfg.model.d_state == 128
        assert cfg.model.expand == 2
        assert cfg.model.d_head == 64
        assert cfg.model.d_conv == 4
        assert cfg.model.chunk_size == 64

        assert cfg.train.target_tokens == 100_000_000_000
        assert cfg.train.precision == "bf16"

        assert cfg.optim.peak_lr == expected_peak_lr[cfg.name]
        assert cfg.optim.min_lr == 1.0e-5
        assert cfg.optim.min_lr_ratio is None
        assert cfg.optim.beta1 == 0.9
        assert cfg.optim.beta2 == 0.95
        assert cfg.optim.weight_decay == 0.1
        assert cfg.optim.grad_clip_norm == 1.0
        assert cfg.optim.warmup_tokens == 1_000_000_000

        assert cfg.checkpoint.save_every_steps == 5000
        assert (
            cfg.checkpoint.save_every_minutes == expected_save_every_minutes[cfg.name]
        )
        assert cfg.checkpoint.keep_last == 3
        assert cfg.checkpoint.save_on_signal is True

        assert cfg.eval.zero_shot_tasks == expected_tasks


def test_cosine_schedule_uses_fixed_min_lr_floor() -> None:
    cfg = load_config([str(EXPERIMENT_DIR / "fwedu-180m.yaml")])
    global_batch_tokens = tokens_per_step(
        seq_len=cfg.data.seq_len,
        per_device_batch_size=cfg.runtime.per_device_batch_size,
        grad_accum_steps=cfg.runtime.grad_accum_steps,
        world_size=1,
    )
    scheduler = CosineSchedule(cfg, global_batch_tokens)
    assert scheduler.min_lr == 1.0e-5
    assert scheduler.lr_at(scheduler.total_steps) == pytest.approx(1.0e-5)
    assert scheduler.lr_at(scheduler.total_steps + 123) == pytest.approx(1.0e-5)
