from __future__ import annotations

import math
from typing import Any, Callable, cast

import torch
from slinoss.layers import SLinOSSMixer
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_slinoss_lm import SLinOSSLMConfig


class GatedMLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        *,
        hidden_features: int,
        out_features: int | None = None,
        bias: bool = False,
        multiple_of: int = 128,
    ) -> None:
        super().__init__()
        out_features = out_features if out_features is not None else in_features
        hidden_features = (
            (hidden_features + multiple_of - 1) // multiple_of * multiple_of
        )
        self.fc1 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, gate = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(y * F.silu(gate))


class SLinOSSBlock(nn.Module):
    def __init__(self, config: SLinOSSLMConfig) -> None:
        super().__init__()
        self.residual_in_fp32 = bool(config.residual_in_fp32)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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
        self.norm2 = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GatedMLP(
            config.hidden_size,
            hidden_features=config.intermediate_size,
            out_features=config.hidden_size,
            bias=False,
            multiple_of=config.mlp_multiple_of,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = (hidden_states + residual) if residual is not None else hidden_states
        hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mixer(hidden_states)

        residual = hidden_states + residual
        hidden_states = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


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
        self.norm_f = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = False
        self.post_init()
        self.tie_weights()

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            if module.bias is not None and not getattr(
                module.bias, "_no_reinit", False
            ):
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

        for name, param in module.named_parameters():
            if name in ["mixer.out_proj.weight", "mlp.fc2.weight"]:
                nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                with torch.no_grad():
                    param /= math.sqrt(2 * self.config.num_hidden_layers)

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
        del use_cache
        return_dict = (
            self.config.use_return_dict if return_dict is None else return_dict
        )

        hidden_states = (
            self.embed_tokens(input_ids) if inputs_embeds is None else inputs_embeds
        )
        residual: torch.Tensor | None = None
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                if residual is None:
                    hidden_states, residual = cast(
                        tuple[torch.Tensor, torch.Tensor],
                        checkpoint(
                            lambda hs: layer(hs, None),
                            hidden_states,
                            use_reentrant=False,
                        ),
                    )
                else:
                    hidden_states, residual = cast(
                        tuple[torch.Tensor, torch.Tensor],
                        checkpoint(
                            layer,
                            hidden_states,
                            residual,
                            use_reentrant=False,
                        ),
                    )
            else:
                hidden_states, residual = layer(hidden_states, residual)

        final_residual = (
            hidden_states + residual if residual is not None else hidden_states
        )
        hidden_states = self.norm_f(final_residual.to(dtype=self.norm_f.weight.dtype))
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
