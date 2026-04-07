from __future__ import annotations

import argparse
from contextlib import AbstractContextManager, nullcontext
import math
import os
import signal
import time
from pathlib import Path
from typing import Mapping, cast

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
from .config import ExperimentConfig, TrainConfig, config_to_dict, load_config
from .configuration_slinoss_lm import SLinOSSLMConfig
from .data import (
    CudaPrefetcher,
    build_eval_loader,
    build_train_loader,
    load_packed_meta,
)
from .modeling_slinoss_lm import SLinOSSCausalLM
from .wandb_integration import WandbLogger, build_wandb_logger


STOP_REQUESTED = False
WALL_CLOCK_DEADLINE_ENV = "SLINOSS_WALL_CLOCK_DEADLINE_UNIX"
MAX_RUNTIME_ENV = "SLINOSS_MAX_RUNTIME_SECONDS"
WALL_CLOCK_MARGIN_ENV = "SLINOSS_WALL_CLOCK_EXIT_MARGIN_SECONDS"


def _signal_handler(signum: int, _frame: object) -> None:
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _parse_positive_int(raw: str, *, field_name: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer; got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{field_name} must be > 0; got {value}")
    return value


def _parse_non_negative_int(raw: str, *, field_name: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer; got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{field_name} must be >= 0; got {value}")
    return value


def resolve_wall_clock_controls(
    config: TrainConfig,
    *,
    launch_time_unix: float,
    env: Mapping[str, str] | None = None,
) -> tuple[int | None, int]:
    env_map = env if env is not None else os.environ
    margin_seconds = int(config.wall_clock_exit_margin_seconds)
    if margin_seconds < 0:
        raise ValueError(
            f"train.wall_clock_exit_margin_seconds must be >= 0; got {margin_seconds}"
        )

    env_margin = env_map.get(WALL_CLOCK_MARGIN_ENV)
    if env_margin:
        margin_seconds = _parse_non_negative_int(
            env_margin, field_name=f"environment variable {WALL_CLOCK_MARGIN_ENV}"
        )

    deadline_candidates: list[int] = []
    if config.wall_clock_deadline_unix is not None:
        deadline_candidates.append(
            _parse_positive_int(
                str(config.wall_clock_deadline_unix),
                field_name="train.wall_clock_deadline_unix",
            )
        )
    if (env_deadline := env_map.get(WALL_CLOCK_DEADLINE_ENV)) is not None:
        deadline_candidates.append(
            _parse_positive_int(
                env_deadline,
                field_name=f"environment variable {WALL_CLOCK_DEADLINE_ENV}",
            )
        )

    max_runtime_seconds: int | None = config.max_runtime_seconds
    if (env_runtime := env_map.get(MAX_RUNTIME_ENV)) is not None:
        max_runtime_seconds = _parse_positive_int(
            env_runtime, field_name=f"environment variable {MAX_RUNTIME_ENV}"
        )
    elif max_runtime_seconds is not None:
        max_runtime_seconds = _parse_positive_int(
            str(max_runtime_seconds), field_name="train.max_runtime_seconds"
        )

    if max_runtime_seconds is not None:
        deadline_candidates.append(int(launch_time_unix) + max_runtime_seconds)

    deadline_unix = min(deadline_candidates) if deadline_candidates else None
    return deadline_unix, margin_seconds


def should_request_stop_for_wall_clock(
    *,
    now_unix: float,
    deadline_unix: int | None,
    margin_seconds: int,
) -> bool:
    if deadline_unix is None:
        return False
    return now_unix >= (deadline_unix - margin_seconds)


def build_training_metrics_payload(
    *,
    step: int,
    tokens_consumed: int,
    sequences_consumed: int,
    loss: float,
    lr: float,
    grad_norm: float,
    tokens_per_second: float,
    step_time_seconds: float,
    elapsed_seconds: float,
    device: torch.device,
) -> dict[str, int | float]:
    payload: dict[str, int | float] = {
        "step": step,
        "tokens_consumed": tokens_consumed,
        "sequences_consumed": sequences_consumed,
        "loss": loss,
        "lr": lr,
        "grad_norm": grad_norm,
        "tokens_per_second": tokens_per_second,
        "step_time_seconds": step_time_seconds,
        "elapsed_seconds": elapsed_seconds,
    }
    if device.type == "cuda":
        payload["cuda_memory_allocated_bytes"] = int(
            torch.cuda.memory_allocated(device)
        )
        payload["cuda_memory_reserved_bytes"] = int(torch.cuda.memory_reserved(device))
        payload["cuda_max_memory_allocated_bytes"] = int(
            torch.cuda.max_memory_allocated(device)
        )
    return payload


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
        self.min_lr = (
            config.optim.peak_lr * config.optim.min_lr_ratio
            if config.optim.min_lr_ratio is not None
            else config.optim.min_lr
        )
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
    state: Mapping[str, int | float],
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
    wandb_logger: WandbLogger | None = None
    exit_code = 1

    try:
        rank, local_rank, world_size = init_distributed()
        launch_wall_unix = time.time()
        wall_clock_deadline_unix, wall_clock_exit_margin_seconds = (
            resolve_wall_clock_controls(
                config.train,
                launch_time_unix=launch_wall_unix,
            )
        )
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

        hf_config = SLinOSSLMConfig(**config.model.architecture_kwargs())
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
                static_graph=config.runtime.ddp_static_graph,
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

        if is_main_process():
            wandb_logger = build_wandb_logger(
                config=config,
                run_dir=run_dir,
                allow_resume=args.resume != "never",
                run_metadata={
                    "parameter_count": count_parameters(base_model),
                    "world_size": world_size,
                    "global_batch_tokens": global_batch_tokens,
                    "dataset_root": str(meta.root),
                    "dataset_sequences": meta.n_sequences,
                    "dataset_tokens": meta.n_tokens,
                    "tokenizer_id": meta.tokenizer_id,
                    "resume_checkpoint": str(latest) if latest is not None else None,
                    "wall_clock_deadline_unix": wall_clock_deadline_unix,
                    "wall_clock_exit_margin_seconds": wall_clock_exit_margin_seconds,
                },
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
            if wall_clock_deadline_unix is not None:
                logger.info(
                    "Wall-clock deadline enabled: deadline_unix=%s margin_seconds=%s",
                    format_int(wall_clock_deadline_unix),
                    wall_clock_exit_margin_seconds,
                )

        start_time = time.perf_counter()
        last_log_time = start_time
        last_checkpoint_time = start_time
        running_loss = 0.0
        running_step_time = 0.0
        running_steps = 0
        wall_clock_stop_announced = False
        last_lr = 0.0
        last_grad_norm = 0.0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        base_model.train()
        optimizer.zero_grad(set_to_none=True)

        while tokens_consumed < config.train.target_tokens:
            time_stop_requested = False
            if is_main_process():
                time_stop_requested = should_request_stop_for_wall_clock(
                    now_unix=time.time(),
                    deadline_unix=wall_clock_deadline_unix,
                    margin_seconds=wall_clock_exit_margin_seconds,
                )
            if world_size > 1:
                stop_flag = torch.tensor(
                    1 if time_stop_requested else 0, device=device, dtype=torch.int32
                )
                dist.broadcast(stop_flag, src=0)
                time_stop_requested = bool(int(stop_flag.item()))
            if time_stop_requested:
                STOP_REQUESTED = True
                if is_main_process() and not wall_clock_stop_announced:
                    logger.warning(
                        "Wall-clock margin reached; stopping after the next checkpoint-safe step."
                    )
                    wall_clock_stop_announced = True

            step_start = time.perf_counter()
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
            last_lr = lr
            last_grad_norm = float(grad_norm)
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
            running_step_time += time.perf_counter() - step_start
            running_steps += 1

            if step % config.logging.log_every_steps == 0:
                now = time.perf_counter()
                elapsed = now - last_log_time
                steps = max(running_steps, 1)
                toks_per_sec = (global_batch_tokens * steps) / max(elapsed, 1e-6)
                avg_loss = running_loss / steps
                avg_step_time = running_step_time / steps
                if is_main_process():
                    payload = build_training_metrics_payload(
                        step=step,
                        tokens_consumed=tokens_consumed,
                        sequences_consumed=sequences_consumed,
                        loss=avg_loss,
                        lr=last_lr,
                        grad_norm=last_grad_norm,
                        tokens_per_second=toks_per_sec,
                        step_time_seconds=avg_step_time,
                        elapsed_seconds=time.perf_counter() - start_time,
                        device=device,
                    )
                    append_jsonl(run_dir / "metrics.jsonl", payload)
                    atomic_write_json(run_dir / "run-state.json", payload)
                    if wandb_logger is not None:
                        wandb_logger.log_training(payload)
                    if device.type == "cuda":
                        logger.info(
                            "step=%s loss=%.4f lr=%.3e grad_norm=%.3f tokens=%s "
                            "toks/s=%.1f step=%.3fs peak_mem=%.2fGiB",
                            format_int(step),
                            avg_loss,
                            lr,
                            float(grad_norm),
                            format_int(tokens_consumed),
                            toks_per_sec,
                            avg_step_time,
                            torch.cuda.max_memory_allocated(device) / (1024**3),
                        )
                    else:
                        logger.info(
                            "step=%s loss=%.4f lr=%.3e grad_norm=%.3f tokens=%s "
                            "toks/s=%.1f step=%.3fs",
                            format_int(step),
                            avg_loss,
                            lr,
                            float(grad_norm),
                            format_int(tokens_consumed),
                            toks_per_sec,
                            avg_step_time,
                        )
                running_loss = 0.0
                running_step_time = 0.0
                running_steps = 0
                last_log_time = now
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)

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
                    if wandb_logger is not None:
                        wandb_logger.log_validation(step=step, metrics=metrics)
                    logger.info(
                        "validation step=%s loss=%.4f ppl=%.3f",
                        step,
                        metrics["loss"],
                        metrics["ppl"],
                    )

            should_save = step % config.checkpoint.save_every_steps == 0
            if is_main_process() and config.checkpoint.save_every_minutes > 0:
                should_save = should_save or (
                    (time.perf_counter() - last_checkpoint_time)
                    >= config.checkpoint.save_every_minutes * 60
                )
            if STOP_REQUESTED and config.checkpoint.save_on_signal:
                should_save = True
            if world_size > 1:
                save_flag = torch.tensor(
                    1 if should_save else 0, device=device, dtype=torch.int32
                )
                dist.broadcast(save_flag, src=0)
                should_save = bool(int(save_flag.item()))
            if should_save:
                barrier()
                if is_main_process():
                    state = {
                        "step": step,
                        "tokens_consumed": tokens_consumed,
                        "sequences_consumed": sequences_consumed,
                    }
                    checkpoint_dir = save_full_checkpoint(
                        run_dir=run_dir,
                        step=step,
                        model=base_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        state=state,
                        keep_last=config.checkpoint.keep_last,
                    )
                    if wandb_logger is not None:
                        wandb_logger.log_checkpoint(
                            step=step,
                            checkpoint_dir=checkpoint_dir,
                            state=state,
                        )
                    logger.info("Saved checkpoint at step=%s", format_int(step))
                barrier()
                last_checkpoint_time = time.perf_counter()

            if STOP_REQUESTED:
                if is_main_process():
                    logger.warning(
                        "Stop requested; exiting after checkpoint-safe step boundary."
                    )
                break

        barrier()
        if is_main_process():
            elapsed = time.perf_counter() - start_time
            if running_steps > 0:
                tail_payload = build_training_metrics_payload(
                    step=step,
                    tokens_consumed=tokens_consumed,
                    sequences_consumed=sequences_consumed,
                    loss=running_loss / running_steps,
                    lr=last_lr,
                    grad_norm=last_grad_norm,
                    tokens_per_second=(global_batch_tokens * running_steps)
                    / max(time.perf_counter() - last_log_time, 1e-6),
                    step_time_seconds=running_step_time / running_steps,
                    elapsed_seconds=elapsed,
                    device=device,
                )
                append_jsonl(run_dir / "metrics.jsonl", tail_payload)
                atomic_write_json(run_dir / "run-state.json", tail_payload)
                if wandb_logger is not None:
                    wandb_logger.log_training(tail_payload)
            if wandb_logger is not None:
                summary_payload: dict[str, int | float | bool] = {
                    "final_step": step,
                    "final_tokens_consumed": tokens_consumed,
                    "final_sequences_consumed": sequences_consumed,
                    "elapsed_seconds": elapsed,
                    "stop_requested": STOP_REQUESTED,
                }
                if device.type == "cuda":
                    summary_payload["cuda_max_memory_allocated_bytes"] = int(
                        torch.cuda.max_memory_allocated(device)
                    )
                wandb_logger.update_summary(summary_payload)
            if device.type == "cuda":
                logger.info(
                    "Finished run=%s step=%s tokens=%s elapsed=%s peak_mem=%.2fGiB",
                    config.name,
                    format_int(step),
                    format_int(tokens_consumed),
                    format_duration(elapsed),
                    torch.cuda.max_memory_allocated(device) / (1024**3),
                )
            else:
                logger.info(
                    "Finished run=%s step=%s tokens=%s elapsed=%s",
                    config.name,
                    format_int(step),
                    format_int(tokens_consumed),
                    format_duration(elapsed),
                )
        exit_code = 0
    finally:
        if is_main_process() and wandb_logger is not None:
            wandb_logger.finish(exit_code=exit_code)
        if is_dist_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
