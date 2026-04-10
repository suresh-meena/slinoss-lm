from __future__ import annotations

import copy
import importlib
import math
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Mamba3ModelConfig


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None


def _import_optional(module_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if not module_name.startswith("mamba_ssm"):
            raise
        raise RuntimeError(
            "Mamba-3 baseline support requires the optional baseline dependencies. "
            "Install them with `MAMBA_FORCE_BUILD=TRUE pip install --no-build-isolation -e .[baselines]`."
        ) from exc


def _load_upstream_symbols() -> dict[str, Any]:
    modules: dict[str, Any] = {}
    modules["block"] = _import_optional("mamba_ssm.modules.block")
    modules["mamba3"] = _import_optional("mamba_ssm.modules.mamba3")
    modules["mha"] = _import_optional("mamba_ssm.modules.mha")
    modules["mlp"] = _import_optional("mamba_ssm.modules.mlp")
    try:
        layer_norm = importlib.import_module("mamba_ssm.ops.triton.layer_norm")
    except ModuleNotFoundError:
        layer_norm = None
    return {
        "Block": modules["block"].Block,
        "Mamba3": modules["mamba3"].Mamba3,
        "MHA": modules["mha"].MHA,
        "GatedMLP": modules["mlp"].GatedMLP,
        "RMSNorm": getattr(layer_norm, "RMSNorm", None),
        "layer_norm_fn": getattr(layer_norm, "layer_norm_fn", None),
        "rms_norm_fn": getattr(layer_norm, "rms_norm_fn", None),
    }


def _init_weights(
    module: nn.Module,
    *,
    n_layer: int,
    initializer_range: float = 0.02,
    rescale_prenorm_residual: bool = True,
    n_residuals_per_layer: int = 1,
) -> None:
    if isinstance(module, nn.Linear):
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if not rescale_prenorm_residual:
        return
    for name, param in module.named_parameters():
        if name in ["out_proj.weight", "fc2.weight"]:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
            with torch.no_grad():
                param /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        *,
        config: Mamba3ModelConfig,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        syms = _load_upstream_symbols()
        Block = syms["Block"]
        Mamba3 = syms["Mamba3"]
        GatedMLP = syms["GatedMLP"]
        RMSNorm = syms["RMSNorm"]
        layer_norm_fn = syms["layer_norm_fn"]
        rms_norm_fn = syms["rms_norm_fn"]

        factory_kwargs = {"device": device, "dtype": dtype}
        self.residual_in_fp32 = config.residual_in_fp32
        self.fused_add_norm = config.fused_add_norm
        self._layer_norm_fn = layer_norm_fn
        self._rms_norm_cls = RMSNorm

        self.embedding = nn.Embedding(
            config.vocab_size, config.d_model, **factory_kwargs
        )
        if self.fused_add_norm and (
            layer_norm_fn is None or rms_norm_fn is None or RMSNorm is None
        ):
            raise RuntimeError(
                "Mamba-3 baseline requires Triton RMSNorm kernels from mamba_ssm for fused_add_norm=True."
            )

        def create_block(layer_idx: int) -> nn.Module:
            ssm_cfg = {
                "layer": "Mamba3",
                "d_state": config.d_state,
                "expand": config.expand,
                "headdim": config.headdim,
                "chunk_size": config.chunk_size,
                "is_mimo": config.is_mimo,
                "mimo_rank": config.mimo_rank,
            }
            mixer_cls = partial(
                Mamba3,
                layer_idx=layer_idx,
                **copy.deepcopy(ssm_cfg),
                **factory_kwargs,
            )
            norm_cls = partial(
                RMSNorm if config.rms_norm else nn.LayerNorm,
                eps=1.0e-5,
                **factory_kwargs,
            )
            mlp_cls = partial(
                GatedMLP,
                hidden_features=config.d_intermediate,
                out_features=config.d_model,
                **factory_kwargs,
            )
            block = Block(
                config.d_model,
                mixer_cls,
                mlp_cls,
                norm_cls=norm_cls,
                fused_add_norm=config.fused_add_norm,
                residual_in_fp32=config.residual_in_fp32,
            )
            block.layer_idx = layer_idx
            return block

        self.layers = nn.ModuleList([create_block(i) for i in range(config.n_layer)])
        self.norm_f = (nn.LayerNorm if not config.rms_norm else RMSNorm)(
            config.d_model,
            eps=1.0e-5,
            **factory_kwargs,
        )
        self.apply(
            partial(
                _init_weights,
                n_layer=config.n_layer,
                initializer_range=config.initializer_range,
                n_residuals_per_layer=2,
            )
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)
        if not self.fused_add_norm:
            residual = (
                (hidden_states + residual) if residual is not None else hidden_states
            )
            return self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        assert self._layer_norm_fn is not None
        assert self._rms_norm_cls is not None
        return self._layer_norm_fn(
            hidden_states,
            self.norm_f.weight,
            self.norm_f.bias,
            eps=self.norm_f.eps,
            residual=residual,
            prenorm=False,
            residual_in_fp32=self.residual_in_fp32,
            is_rms_norm=isinstance(self.norm_f, self._rms_norm_cls),
        )


class Mamba3CausalLM(nn.Module):
    def __init__(
        self,
        config: Mamba3ModelConfig,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        vocab_size = config.vocab_size
        if vocab_size % config.pad_vocab_size_multiple != 0:
            vocab_size += config.pad_vocab_size_multiple - (
                vocab_size % config.pad_vocab_size_multiple
            )
        backbone_config = copy.deepcopy(config)
        backbone_config.vocab_size = vocab_size
        self.backbone = MixerModel(config=backbone_config, device=device, dtype=dtype)
        self.lm_head = nn.Linear(
            backbone_config.d_model, vocab_size, bias=False, device=device, dtype=dtype
        )
        self.apply(
            partial(
                _init_weights,
                n_layer=config.n_layer,
                initializer_range=config.initializer_range,
            )
        )
        if config.tie_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> CausalLMOutput:
        hidden_states = self.backbone(input_ids)
        logits = self.lm_head(hidden_states)
        loss: torch.Tensor | None = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, logits.shape[-1]),
                labels[:, 1:].contiguous().view(-1),
            )
        return CausalLMOutput(logits=logits, loss=loss)
