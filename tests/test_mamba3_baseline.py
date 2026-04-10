from __future__ import annotations

from pathlib import Path
from typing import Iterable, cast

import pytest
import torch

from slinoss_lm.baselines.mamba3.config import Mamba3ModelConfig, load_config
from slinoss_lm.baselines.mamba3.model import Mamba3CausalLM
from slinoss_lm.baselines.mamba3.train import build_optimizer
from slinoss_lm.data import PackedDatasetMeta, build_eval_loader


def test_mamba3_config_loads_repo_baseline() -> None:
    config = load_config(["configs/baselines/mamba3-siso-180m.yaml"])
    assert config.model.d_model == 768
    assert config.model.n_layer == 11
    assert config.model.d_intermediate == 1500
    assert config.validation.start_sequence == -4096


def test_build_eval_loader_supports_negative_start_sequence(tmp_path: Path) -> None:
    meta = PackedDatasetMeta(
        root=tmp_path,
        seq_len=8,
        n_sequences=10,
        n_tokens=80,
        tokenizer_id="tok",
        shard_paths=[],
        shard_sequences=[],
        cumulative_sequences=[],
    )
    loader = build_eval_loader(
        meta=meta, batch_size=2, start_sequence=-3, num_sequences=2
    )
    sampler = cast(Iterable[list[int]], loader.batch_sampler)
    batches = list(sampler)
    assert batches == [[7, 8]]


class _ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 4, bias=False)
        self.norm = torch.nn.LayerNorm(4)
        self.dt_bias = torch.nn.Parameter(torch.ones(4))
        setattr(self.dt_bias, "_no_weight_decay", True)


def test_mamba3_optimizer_respects_no_weight_decay_markers() -> None:
    cfg = load_config(["configs/baselines/mamba3-siso-180m.yaml"])
    toy = _ToyModel()
    optimizer = build_optimizer(toy, cfg)
    decays = {group["weight_decay"] for group in optimizer.param_groups}
    assert decays == {0.0, cfg.optim.weight_decay}
    zero_decay_params = {
        id(param)
        for group in optimizer.param_groups
        if group["weight_decay"] == 0.0
        for param in group["params"]
    }
    assert id(toy.dt_bias) in zero_decay_params
    assert id(toy.norm.weight) in zero_decay_params


def test_mamba3_model_import_error_is_actionable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib

    real_import_module = importlib.import_module

    def fake_import(name: str, package: str | None = None):
        if name.startswith("mamba_ssm"):
            raise ModuleNotFoundError(name)
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    with pytest.raises(RuntimeError, match="baseline dependencies"):
        Mamba3CausalLM(Mamba3ModelConfig())
