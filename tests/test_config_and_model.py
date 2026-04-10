from __future__ import annotations

from pathlib import Path
import warnings

import torch
import torch.nn as nn

from slinoss_lm.configuration_slinoss_lm import SLinOSSLMConfig
from slinoss_lm.config import ModelConfig, load_config
from slinoss_lm.inspect import inspect_config
from slinoss_lm.modeling_slinoss_lm import SLinOSSBlock, SLinOSSCausalLM
from slinoss_lm.train import build_optimizer


def test_model_forward_shape_and_loss() -> None:
    config = SLinOSSLMConfig(
        vocab_size=128256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        d_state=32,
        expand=2,
        d_head=32,
        d_conv=4,
        chunk_size=16,
    )
    model = SLinOSSCausalLM(config)
    import torch

    input_ids = torch.randint(0, config.vocab_size, (2, 32))
    outputs = model(input_ids=input_ids, labels=input_ids)
    assert outputs.logits.shape == (2, 32, config.vocab_size)
    assert outputs.loss is not None


def test_tied_embeddings() -> None:
    config = SLinOSSLMConfig(
        vocab_size=128256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        d_state=32,
        expand=2,
        d_head=32,
        d_conv=4,
        chunk_size=16,
    )
    model = SLinOSSCausalLM(config)
    assert model.embed_tokens.weight is model.lm_head.weight


def test_inspect_config_uses_explicit_world_size(tmp_path: Path) -> None:
    dataset_root = tmp_path / "packed" / "data" / "fwedu-llama31-2k"
    dataset_root.mkdir(parents=True)
    (dataset_root / "meta.json").write_text(
        """
{
  "seq_len": 128,
  "n_sequences": 1024,
  "n_tokens_emitted": 131072,
  "tokenizer_id": "test-tokenizer",
  "shards": [
    {"file": "shard-00000.bin", "n_sequences": 1024}
  ]
}
""".strip()
        + "\n"
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"""
name: inspect-test
model:
  vocab_size: 256
  hidden_size: 64
  intermediate_size: 128
  num_hidden_layers: 2
  d_state: 32
  expand: 2
  d_head: 32
  d_conv: 4
  chunk_size: 16
data:
  root: {str((tmp_path / "packed").resolve())!r}
  dataset_dir: data/fwedu-llama31-2k
  seq_len: 128
train:
  target_tokens: 4096
runtime:
  per_device_batch_size: 2
  grad_accum_steps: 4
"""
    )
    config = load_config([str(config_path)])
    payload = inspect_config(config, world_size=4)
    assert payload["world_size"] == 4
    assert payload["global_batch_tokens"] == 4096
    assert payload["optimizer_steps_to_target_tokens"] == 1


def test_model_passes_mixer_stability_defaults(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _DummyMixer(nn.Module):
        def __init__(self, d_model: int, **kwargs: object) -> None:
            super().__init__()
            captured["d_model"] = d_model
            captured["kwargs"] = kwargs

        def forward(self, x):  # type: ignore[no-untyped-def]
            return x

    import slinoss_lm.modeling_slinoss_lm as modeling

    monkeypatch.setattr(modeling, "SLinOSSMixer", _DummyMixer)
    config = SLinOSSLMConfig(
        vocab_size=128256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        d_state=32,
        expand=2,
        d_head=32,
        d_conv=4,
        chunk_size=16,
    )
    _ = modeling.SLinOSSCausalLM(config)
    assert captured["d_model"] == 64
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["r_min"] == 0.8
    assert kwargs["dt_min"] == 1.0e-3
    assert kwargs["dt_init_floor"] == 1.0e-3


def test_runtime_defaults_disable_ddp_static_graph() -> None:
    config = load_config([])
    assert config.runtime.ddp_static_graph is False


def test_model_architecture_kwargs_excludes_runtime_fields() -> None:
    config = ModelConfig(gradient_checkpointing=True)
    kwargs = config.architecture_kwargs()
    assert "gradient_checkpointing" not in kwargs
    assert kwargs["hidden_size"] == config.hidden_size


def test_hf_config_init_does_not_warn_for_gradient_checkpointing() -> None:
    config = ModelConfig(gradient_checkpointing=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _ = SLinOSSLMConfig(**config.architecture_kwargs())


def test_block_keeps_residual_in_fp32_when_enabled(monkeypatch) -> None:
    class _DummyMixer(nn.Module):
        def __init__(self, d_model: int, **_: object) -> None:
            super().__init__()
            self.proj = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    import slinoss_lm.modeling_slinoss_lm as modeling

    monkeypatch.setattr(modeling, "SLinOSSMixer", _DummyMixer)
    config = SLinOSSLMConfig(
        vocab_size=128256,
        hidden_size=64,
        intermediate_size=96,
        num_hidden_layers=1,
        d_state=32,
        expand=2,
        d_head=32,
        d_conv=4,
        chunk_size=16,
        residual_in_fp32=True,
        mlp_multiple_of=32,
    )
    block = SLinOSSBlock(config).to(dtype=torch.bfloat16)
    hidden = torch.randn(2, 4, 64, dtype=torch.bfloat16)
    out, residual = block(hidden)
    assert out.dtype == torch.bfloat16
    assert residual.dtype == torch.float32


def test_block_matches_mamba_unfused_residual_algebra(monkeypatch) -> None:
    class _DummyMixer(nn.Module):
        def __init__(self, d_model: int, **_: object) -> None:
            super().__init__()
            self.proj = nn.Linear(d_model, d_model, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    class _DummyMLP(nn.Module):
        def __init__(
            self,
            in_features: int,
            *,
            hidden_features: int,
            out_features: int | None = None,
            **_: object,
        ) -> None:
            super().__init__()
            del hidden_features
            out_features = in_features if out_features is None else out_features
            self.proj = nn.Linear(in_features, out_features, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    import slinoss_lm.modeling_slinoss_lm as modeling

    monkeypatch.setattr(modeling, "SLinOSSMixer", _DummyMixer)
    monkeypatch.setattr(modeling, "GatedMLP", _DummyMLP)
    config = SLinOSSLMConfig(
        vocab_size=128256,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        d_state=16,
        expand=2,
        d_head=16,
        d_conv=4,
        chunk_size=16,
        residual_in_fp32=True,
    )
    block = modeling.SLinOSSBlock(config).to(dtype=torch.bfloat16)
    hidden = torch.randn(2, 4, 32, dtype=torch.bfloat16)
    carried = torch.randn(2, 4, 32, dtype=torch.bfloat16)

    out, residual = block(hidden, carried)

    expected_residual = hidden + carried
    normed = block.norm(expected_residual.to(dtype=block.norm.weight.dtype))
    mixed = block.mixer(normed)
    expected_residual = (mixed + expected_residual).to(torch.float32)
    normed2 = block.norm2(expected_residual.to(dtype=block.norm2.weight.dtype))
    expected_out = block.mlp(normed2)

    assert residual.dtype == torch.float32
    assert torch.allclose(residual, expected_residual, atol=1.0e-2, rtol=1.0e-2)
    assert torch.allclose(out, expected_out, atol=1.0e-2, rtol=1.0e-2)


def test_optimizer_respects_no_weight_decay_markers() -> None:
    class _Toy(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 4, bias=False)
            self.norm = nn.LayerNorm(4)
            self.bias_param = nn.Parameter(torch.zeros(4))
            self.tagged = nn.Parameter(torch.ones(2, 2))
            setattr(self.tagged, "_no_weight_decay", True)

    cfg = load_config([])
    toy = _Toy()
    optimizer = build_optimizer(toy, cfg)
    decay_group, no_decay_group = optimizer.param_groups
    decay_ids = {id(param) for param in decay_group["params"]}
    no_decay_ids = {id(param) for param in no_decay_group["params"]}

    assert id(toy.linear.weight) in decay_ids
    assert id(toy.norm.weight) in no_decay_ids
    assert id(toy.bias_param) in no_decay_ids
    assert id(toy.tagged) in no_decay_ids
