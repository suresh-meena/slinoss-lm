from __future__ import annotations

import argparse
import json
import math
from typing import Any

from .common import format_int, tokens_per_step
from .config import ExperimentConfig, config_to_dict, load_config
from .configuration_slinoss_lm import SLinOSSLMConfig
from .data import load_packed_meta
from .modeling_slinoss_lm import SLinOSSCausalLM


def inspect_config(config: ExperimentConfig) -> dict[str, Any]:
    hf_config = SLinOSSLMConfig(**config.model.__dict__)
    model = SLinOSSCausalLM(hf_config)
    meta = load_packed_meta(config.data)
    world_size = 2
    global_batch_tokens = tokens_per_step(
        seq_len=meta.seq_len,
        per_device_batch_size=config.runtime.per_device_batch_size,
        grad_accum_steps=config.runtime.grad_accum_steps,
        world_size=world_size,
    )
    return {
        "name": config.name,
        "params": sum(p.numel() for p in model.parameters()),
        "dataset_root": str(meta.root),
        "dataset_sequences": meta.n_sequences,
        "dataset_tokens": meta.n_tokens,
        "tokenizer_id": meta.tokenizer_id,
        "global_batch_tokens_assuming_2_gpus": global_batch_tokens,
        "optimizer_steps_to_100b_tokens_assuming_2_gpus": math.ceil(
            config.train.target_tokens / global_batch_tokens
        ),
        "resolved_config": config_to_dict(config),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, args.overrides)
    payload = inspect_config(config)
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    print(f"name: {payload['name']}")
    print(f"params: {format_int(int(payload['params']))}")
    print(f"dataset_root: {payload['dataset_root']}")
    print(f"dataset_sequences: {format_int(int(payload['dataset_sequences']))}")
    print(f"dataset_tokens: {format_int(int(payload['dataset_tokens']))}")
    print(f"tokenizer_id: {payload['tokenizer_id']}")
    print(
        "global_batch_tokens_assuming_2_gpus: "
        f"{format_int(int(payload['global_batch_tokens_assuming_2_gpus']))}"
    )
    print(
        "optimizer_steps_to_100b_tokens_assuming_2_gpus: "
        f"{format_int(int(payload['optimizer_steps_to_100b_tokens_assuming_2_gpus']))}"
    )


if __name__ == "__main__":
    main()
