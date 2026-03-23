from __future__ import annotations

import argparse
from contextlib import AbstractContextManager, nullcontext
import math
import os
import signal
import time
from pathlib import Path
from typing import cast

import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

from .checkpoint import (
    capture_rng_state,
    find_latest_checkpoint,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
from .common import (
    append_jsonl,
    atomic_write_json,
    atomic_write_text,
    barrier,
    collect_system_info,
    count_parameters,
    ensure_dir,
    format_duration,
    format_int,
    init_logging,
    is_dist_initialized,
    is_main_process,
    set_seed,
    tokens_per_step,
)
from .config import ExperimentConfig, config_to_dict, load_config
from .configuration_slinoss_lm import SLinOSSLMConfig
from .data import (
    CudaPrefetcher,
    build_eval_loader,
    build_train_loader,
    load_packed_meta,
)
from .modeling_slinoss_lm import SLinOSSCausalLM


STOP_REQUESTED = False


def _signal_handler(signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", required=True)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--resume", default="auto")
    return parser


def init_distributed() -> tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def maybe_compile(module: torch.nn.Module, config: ExperimentConfig) -> torch.nn.Module:
    if not config.train.compile:
        return module
    return cast(torch.nn.Module, torch.compile(module, mode=config.train.compile_mode))


def build_optimizer(
    model: torch.nn.Module, config: ExperimentConfig
) -> torch.optim.Optimizer:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if (
            param.ndim < 2
            or "norm" in name.lower()
            or name.endswith("bias")
            or "skip" in name
        ):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": config.optim.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=config.optim.peak_lr,
        betas=(config.optim.beta1, config.optim.beta2),
        eps=config.optim.eps,
        fused=torch.cuda.is_available(),
    )


class CosineSchedule:
    def __init__(self, config: ExperimentConfig, global_batch_tokens: int) -> None:
        self.peak_lr = config.optim.peak_lr
        self.min_lr = config.optim.peak_lr * config.optim.min_lr_ratio
        self.warmup_steps = max(
            1, math.ceil(config.optim.warmup_tokens / global_batch_tokens)
        )
        self.total_steps = math.ceil(config.train.target_tokens / global_batch_tokens)

    def lr_at(self, step: int) -> float:
        if step < self.warmup_steps:
            return self.peak_lr * float(step + 1) / float(self.warmup_steps)
        progress = (step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine

    def step(self, optimizer: torch.optim.Optimizer, step: int) -> float:
        lr = self.lr_at(step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        return lr

    def state_dict(self) -> dict[str, int | float]:
        return {
            "peak_lr": self.peak_lr,
            "min_lr": self.min_lr,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
        }

    def load_state_dict(self, _state: dict[str, int | float]) -> None:
        return


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    *,
    config: ExperimentConfig,
    device: torch.device,
) -> dict[str, float] | None:
    if not config.validation.enabled or config.validation.num_sequences <= 0:
        return None
    data_cfg = config.data
    if config.validation.dataset_root:
        data_cfg = config.data.__class__(
            root=config.validation.dataset_root,
            root_env=config.data.root_env,
            dataset_dir=config.validation.dataset_dir or config.data.dataset_dir,
            seq_len=config.data.seq_len,
            tokenizer_name=config.data.tokenizer_name,
            tokenizer_path=config.data.tokenizer_path,
        )
    meta = load_packed_meta(data_cfg)
    loader = build_eval_loader(
        meta=meta,
        batch_size=config.validation.batch_size,
        start_sequence=config.validation.start_sequence,
        num_sequences=config.validation.num_sequences,
    )
    model.eval()
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
    ppl = math.exp(mean_loss)
    model.train()
    return {"loss": mean_loss, "ppl": ppl}


def save_full_checkpoint(
    *,
    run_dir: Path,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: CosineSchedule,
    state: dict[str, int | float],
    keep_last: int,
) -> Path:
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "state": state,
        "rng": capture_rng_state(),
    }
    return save_checkpoint(
        run_dir=run_dir, step=step, payload=payload, keep_last=keep_last
    )


def optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=device, non_blocking=True)


def main() -> None:
    global STOP_REQUESTED
    args = build_parser().parse_args()
    config = load_config(args.config, args.overrides)

    rank, local_rank, world_size = init_distributed()
    device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    if config.train.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    run_dir = Path(config.checkpoint.run_root) / config.name
    logger = init_logging(run_dir / "train.log")

    if is_main_process():
        ensure_dir(run_dir)
        atomic_write_json(run_dir / "system-info.json", collect_system_info())
        atomic_write_text(
            run_dir / "resolved-config.yaml",
            yaml.safe_dump(config_to_dict(config), sort_keys=False),
        )

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    set_seed(config.train.seed + rank)

    meta = load_packed_meta(config.data)
    global_batch_tokens = tokens_per_step(
        seq_len=meta.seq_len,
        per_device_batch_size=config.runtime.per_device_batch_size,
        grad_accum_steps=config.runtime.grad_accum_steps,
        world_size=world_size,
    )

    hf_config = SLinOSSLMConfig(**config.model.__dict__)
    base_model = SLinOSSCausalLM(hf_config)
    base_model.gradient_checkpointing = bool(config.model.gradient_checkpointing)
    if device.type == "cuda":
        torch.nn.Module.cuda(base_model, device.index or 0)
    else:
        torch.nn.Module.cpu(base_model)
    raw_model = maybe_compile(base_model, config)
    model: torch.nn.Module = raw_model
    if world_size > 1:
        model = DDP(
            raw_model,
            device_ids=[local_rank],
            output_device=local_rank,
            gradient_as_bucket_view=True,
        )

    optimizer = build_optimizer(base_model, config)
    scheduler = CosineSchedule(config, global_batch_tokens)

    step = 0
    tokens_consumed = 0
    sequences_consumed = 0
    latest = None if args.resume == "never" else find_latest_checkpoint(run_dir)
    if latest is not None and latest.exists():
        payload = load_checkpoint(latest, map_location="cpu")
        base_model.load_state_dict(payload["model"])
        optimizer.load_state_dict(payload["optimizer"])
        optimizer_to_device(optimizer, device)
        scheduler.load_state_dict(payload["scheduler"])
        state = payload["state"]
        step = int(state["step"])
        tokens_consumed = int(state["tokens_consumed"])
        sequences_consumed = int(state["sequences_consumed"])
        restore_rng_state(payload["rng"])
        logger.info(
            "Resumed from %s at step=%s tokens=%s sequences=%s",
            latest,
            format_int(step),
            format_int(tokens_consumed),
            format_int(sequences_consumed),
        )

    loader = build_train_loader(
        meta=meta,
        runtime=config.runtime,
        rank=rank,
        world_size=world_size,
        start_sequence=sequences_consumed,
    )
    iterator = iter(CudaPrefetcher(loader, device))

    if is_main_process():
        logger.info(
            "Run %s starting on %s GPU(s), params=%s, dataset=%s, batch_tokens=%s",
            config.name,
            world_size,
            format_int(count_parameters(base_model)),
            meta.root,
            format_int(global_batch_tokens),
        )

    start_time = time.perf_counter()
    last_log_time = start_time
    running_loss = 0.0
    running_steps = 0

    base_model.train()
    optimizer.zero_grad(set_to_none=True)

    while tokens_consumed < config.train.target_tokens:
        micro_losses: list[float] = []
        for micro_step in range(config.runtime.grad_accum_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                logger.info(
                    "Reached end of dataset after %s tokens.",
                    format_int(tokens_consumed),
                )
                tokens_consumed = config.train.target_tokens
                break

            sync_context: AbstractContextManager[object]
            if world_size > 1 and micro_step < config.runtime.grad_accum_steps - 1:
                sync_context = cast(DDP, model).no_sync()
            else:
                sync_context = nullcontext()
            with sync_context:
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16
                    if config.train.precision == "bf16"
                    else torch.float16,
                    enabled=device.type == "cuda",
                ):
                    outputs = model(**batch)
                    loss = outputs.loss
                    assert loss is not None
                    scaled_loss = loss / config.runtime.grad_accum_steps
                scaled_loss.backward()
                micro_losses.append(float(loss.detach()))

        if not micro_losses:
            break

        grad_norm = torch.nn.utils.clip_grad_norm_(
            base_model.parameters(),
            config.optim.grad_clip_norm,
        )
        lr = scheduler.step(optimizer, step)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        step += 1
        sequences_consumed += (
            config.runtime.per_device_batch_size
            * world_size
            * config.runtime.grad_accum_steps
        )
        tokens_consumed += global_batch_tokens
        running_loss += sum(micro_losses) / len(micro_losses)
        running_steps += 1

        if step % config.logging.log_every_steps == 0:
            now = time.perf_counter()
            elapsed = now - last_log_time
            steps = max(running_steps, 1)
            toks_per_sec = (global_batch_tokens * steps) / max(elapsed, 1e-6)
            avg_loss = running_loss / steps
            if is_main_process():
                payload = {
                    "step": step,
                    "tokens_consumed": tokens_consumed,
                    "sequences_consumed": sequences_consumed,
                    "loss": avg_loss,
                    "lr": lr,
                    "grad_norm": float(grad_norm),
                    "tokens_per_second": toks_per_sec,
                    "elapsed_seconds": time.perf_counter() - start_time,
                }
                append_jsonl(run_dir / "metrics.jsonl", payload)
                atomic_write_json(run_dir / "run-state.json", payload)
                logger.info(
                    "step=%s loss=%.4f lr=%.3e grad_norm=%.3f tokens=%s toks/s=%.1f",
                    format_int(step),
                    avg_loss,
                    lr,
                    float(grad_norm),
                    format_int(tokens_consumed),
                    toks_per_sec,
                )
            running_loss = 0.0
            running_steps = 0
            last_log_time = now

        should_validate = (
            config.validation.enabled
            and config.validation.every_steps > 0
            and step % config.validation.every_steps == 0
        )
        if should_validate and is_main_process():
            metrics = run_validation(base_model, config=config, device=device)
            if metrics is not None:
                append_jsonl(
                    run_dir / "metrics.jsonl", {"step": step, "validation": metrics}
                )
                logger.info(
                    "validation step=%s loss=%.4f ppl=%.3f",
                    step,
                    metrics["loss"],
                    metrics["ppl"],
                )

        should_save = step % config.checkpoint.save_every_steps == 0
        if STOP_REQUESTED and config.checkpoint.save_on_signal:
            should_save = True
        if should_save:
            barrier()
            if is_main_process():
                save_full_checkpoint(
                    run_dir=run_dir,
                    step=step,
                    model=base_model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    state={
                        "step": step,
                        "tokens_consumed": tokens_consumed,
                        "sequences_consumed": sequences_consumed,
                    },
                    keep_last=config.checkpoint.keep_last,
                )
                logger.info("Saved checkpoint at step=%s", format_int(step))
            barrier()

        if STOP_REQUESTED:
            if is_main_process():
                logger.warning(
                    "Stop requested; exiting after checkpoint-safe step boundary."
                )
            break

    barrier()
    if is_main_process():
        elapsed = time.perf_counter() - start_time
        logger.info(
            "Finished run=%s step=%s tokens=%s elapsed=%s",
            config.name,
            format_int(step),
            format_int(tokens_consumed),
            format_duration(elapsed),
        )
    if is_dist_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
