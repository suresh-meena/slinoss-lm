from __future__ import annotations

from typing import Any, Callable, cast

import torch
from slinoss.layers import SLinOSSMixer
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_slinoss_lm import SLinOSSLMConfig


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class SLinOSSBlock(nn.Module):
    def __init__(self, config: SLinOSSLMConfig) -> None:
        super().__init__()
        self.input_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mixer = SLinOSSMixer(
            config.hidden_size,
            d_state=config.d_state,
            expand=config.expand,
            d_head=config.d_head,
            d_conv=config.d_conv,
            chunk_size=config.chunk_size,
            dt_min=config.dt_min,
            dt_init_floor=config.dt_init_floor,
            r_min=config.r_min,
        )
        self.post_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = SwiGLU(config.hidden_size, config.intermediate_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = hidden_states + self.mixer(self.input_norm(hidden_states))
        hidden_states = hidden_states + self.mlp(self.post_norm(hidden_states))
        return hidden_states


class SLinOSSCausalLM(PreTrainedModel, GenerationMixin):
    config_class = SLinOSSLMConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    supports_gradient_checkpointing = True

    def __init__(self, config: SLinOSSLMConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [SLinOSSBlock(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.post_init()
        self.tie_weights()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.embed_tokens = value

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.lm_head = new_embeddings

    def _set_gradient_checkpointing(
        self,
        enable: bool = True,
        gradient_checkpointing_func: Callable[..., Any] = checkpoint,
    ) -> None:
        del gradient_checkpointing_func
        self.gradient_checkpointing = enable

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Any = None,
        attention_mask: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        cache_position: torch.LongTensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del past_key_values, cache_position, kwargs
        if inputs_embeds is not None and input_ids is None:
            return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        use_cache: bool | None = None,
        return_dict: bool | None = None,
        **_: object,
    ) -> CausalLMOutputWithPast | tuple[torch.Tensor, ...]:
        if input_ids is None and inputs_embeds is None:
            raise ValueError("Provide input_ids or inputs_embeds.")
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("Provide only one of input_ids or inputs_embeds.")
        if attention_mask is not None and not torch.all(attention_mask > 0):
            raise ValueError("Packed FineWeb rows do not support padding masks.")
        return_dict = (
            self.config.use_return_dict if return_dict is None else return_dict
        )

        hidden_states = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(layer, hidden_states, use_reentrant=False)
            else:
                hidden_states = layer(hidden_states)

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

        if not return_dict:
            out = (logits,)
            if loss is not None:
                out = (loss,) + out
            return out

        return CausalLMOutputWithPast(
            loss=cast(torch.FloatTensor | None, loss),
            logits=cast(torch.FloatTensor, logits),
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )


SLinOSSCausalLM.register_for_auto_class("AutoModelForCausalLM")
