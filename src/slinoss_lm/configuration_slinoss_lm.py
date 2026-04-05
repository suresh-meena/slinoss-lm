from __future__ import annotations

from transformers import PretrainedConfig


class SLinOSSLMConfig(PretrainedConfig):
    model_type = "slinoss_lm"

    def __init__(
        self,
        vocab_size: int = 128256,
        hidden_size: int = 512,
        intermediate_size: int = 1536,
        num_hidden_layers: int = 14,
        d_state: int = 128,
        expand: int = 2,
        d_head: int = 64,
        d_conv: int = 4,
        chunk_size: int = 64,
        dt_min: float = 1.0e-3,
        dt_init_floor: float = 1.0e-3,
        r_min: float = 0.5,
        rms_norm_eps: float = 1.0e-5,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.d_state = d_state
        self.expand = expand
        self.d_head = d_head
        self.d_conv = d_conv
        self.chunk_size = chunk_size
        self.dt_min = dt_min
        self.dt_init_floor = dt_init_floor
        self.r_min = r_min
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.tie_word_embeddings = tie_word_embeddings


SLinOSSLMConfig.register_for_auto_class()
