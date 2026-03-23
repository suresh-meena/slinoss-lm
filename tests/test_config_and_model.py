from __future__ import annotations

from slinoss_lm.configuration_slinoss_lm import SLinOSSLMConfig
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
