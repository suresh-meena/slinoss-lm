from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .checkpoint import load_checkpoint
from .common import atomic_write_json
from .config import DataConfig, load_config
from .configuration_slinoss_lm import SLinOSSLMConfig
from .data import build_eval_loader, load_packed_meta
from .modeling_slinoss_lm import SLinOSSCausalLM


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--start-sequence", type=int, default=0)
    parser.add_argument("--num-sequences", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default=None)
    return parser


@torch.no_grad()
def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args.config, args.overrides)
    ckpt = load_checkpoint(Path(args.checkpoint), map_location="cpu")
    model = SLinOSSCausalLM(SLinOSSLMConfig(**config.model.architecture_kwargs()))
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.nn.Module.cuda(model, device.index or 0)
    else:
        torch.nn.Module.cpu(model)
    model.eval()

    data_cfg = DataConfig(
        root=args.dataset_root,
        root_env=config.data.root_env,
        dataset_dir=args.dataset_dir or config.data.dataset_dir,
        seq_len=config.data.seq_len,
        tokenizer_name=config.data.tokenizer_name,
        tokenizer_path=config.data.tokenizer_path,
    )
    meta = load_packed_meta(data_cfg)
    loader = build_eval_loader(
        meta=meta,
        batch_size=args.batch_size,
        start_sequence=args.start_sequence,
        num_sequences=args.num_sequences,
    )

    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        assert loss is not None
        tokens = batch["input_ids"].numel()
        total_loss += float(loss) * tokens
        total_tokens += tokens

    mean_loss = total_loss / total_tokens
    payload = {
        "checkpoint": args.checkpoint,
        "dataset_root": str(meta.root),
        "start_sequence": args.start_sequence,
        "num_sequences": args.num_sequences,
        "loss": mean_loss,
        "ppl": math.exp(mean_loss),
    }
    print(json.dumps(payload, indent=2))
    if args.output:
        atomic_write_json(Path(args.output), payload)


if __name__ == "__main__":
    main()
