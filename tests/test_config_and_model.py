from __future__ import annotations

from pathlib import Path

from slinoss_lm.configuration_slinoss_lm import SLinOSSLMConfig
from slinoss_lm.config import load_config
from slinoss_lm.inspect import inspect_config
from slinoss_lm.modeling_slinoss_lm import SLinOSSCausalLM


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
