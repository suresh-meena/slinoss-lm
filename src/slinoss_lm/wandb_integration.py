from __future__ import annotations

import json
import importlib
from pathlib import Path
from typing import Any, Mapping

from .common import atomic_write_json, ensure_dir
from .config import ExperimentConfig, config_to_dict


def _wandb_run_metadata_path(run_dir: Path) -> Path:
    return run_dir / "wandb-run.json"


def _load_previous_run_id(run_dir: Path) -> str | None:
    path = _wandb_run_metadata_path(run_dir)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    run_id = payload.get("id")
    return run_id if isinstance(run_id, str) and run_id else None


def _require_wandb(module: Any | None) -> Any:
    if module is not None:
        return module
    try:
        return importlib.import_module("wandb")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "W&B is enabled but the 'wandb' package is not installed. "
            "Install it with `pip install -r requirements-wandb.txt`."
        ) from exc


def _namespace_scalars(
    prefix: str, payload: Mapping[str, Any]
) -> dict[str, int | float]:
    out: dict[str, int | float] = {}
    for key, value in payload.items():
        if key == "step" or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            out[f"{prefix}/{key}"] = value
    return out


def _dashboard_training_aliases(payload: Mapping[str, Any]) -> dict[str, int | float]:
    aliases: dict[str, int | float] = {}
    scalar_aliases = {
        "loss": "loss",
        "lr": "lr",
        "grad_norm": "grad_norm",
        "tokens_per_second": "toks_per_sec",
        "step_time_seconds": "step_time_seconds",
    }
    for source_key, alias_key in scalar_aliases.items():
        value = payload.get(source_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            aliases[alias_key] = value
    max_mem = payload.get("cuda_max_memory_allocated_bytes")
    if isinstance(max_mem, (int, float)) and not isinstance(max_mem, bool):
        aliases["max_memory_gib"] = float(max_mem) / float(1024**3)
    return aliases


def _dashboard_validation_aliases(metrics: Mapping[str, Any]) -> dict[str, int | float]:
    aliases: dict[str, int | float] = {}
    scalar_aliases = {
        "loss": "val_loss",
        "ppl": "val_ppl",
    }
    for source_key, alias_key in scalar_aliases.items():
        value = metrics.get(source_key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            aliases[alias_key] = value
    return aliases


class WandbLogger:
    def __init__(
        self,
        *,
        module: Any | None = None,
        run: Any | None = None,
        upload_checkpoints: bool = False,
    ) -> None:
        self._module = module
        self._run = run
        self._upload_checkpoints = upload_checkpoints

    @property
    def enabled(self) -> bool:
        return self._run is not None

    def _log(self, payload: dict[str, Any], *, step: int) -> None:
        if self._run is None:
            return
        enriched = dict(payload)
        enriched["step"] = step
        self._run.log(enriched, step=step)

    def update_summary(self, payload: dict[str, Any]) -> None:
        if self._run is None:
            return
        self._run.summary.update(payload)

    def log_training(self, payload: dict[str, Any]) -> None:
        if self._run is None:
            return
        step = int(payload["step"])
        metrics = _namespace_scalars("train", payload)
        metrics.update(_dashboard_training_aliases(payload))
        self._log(metrics, step=step)

    def log_validation(self, *, step: int, metrics: dict[str, float]) -> None:
        if self._run is None:
            return
        payload = _namespace_scalars("validation", metrics)
        payload.update(_dashboard_validation_aliases(metrics))
        self._log(payload, step=step)

    def log_checkpoint(
        self,
        *,
        step: int,
        checkpoint_dir: Path,
        state: Mapping[str, int | float],
    ) -> None:
        if self._run is None:
            return
        payload = _namespace_scalars("checkpoint", state)
        payload["checkpoint/saved"] = 1
        self._log(payload, step=step)
        self._run.summary["latest_checkpoint_step"] = step
        self._run.summary["latest_checkpoint_path"] = str(checkpoint_dir)
        if not self._upload_checkpoints:
            return
        assert self._module is not None
        artifact = self._module.Artifact(
            name=f"{self._run.id}-checkpoint",
            type="checkpoint",
            metadata={"step": step, **state},
        )
        artifact.add_dir(str(checkpoint_dir))
        self._run.log_artifact(
            artifact,
            aliases=["latest", f"step-{step:09d}"],
        )

    def finish(self, *, exit_code: int) -> None:
        if self._run is None:
            return
        self._run.finish(exit_code=exit_code)


def build_wandb_logger(
    *,
    config: ExperimentConfig,
    run_dir: Path,
    allow_resume: bool,
    run_metadata: dict[str, Any],
    module: Any | None = None,
) -> WandbLogger:
    wandb_config = config.wandb
    if not wandb_config.enabled:
        return WandbLogger()

    if wandb_config.resume == "must" and not (
        wandb_config.run_id or _load_previous_run_id(run_dir)
    ):
        raise ValueError(
            "wandb.resume='must' requires wandb.run_id or an existing wandb-run.json."
        )

    wandb = _require_wandb(module)
    ensure_dir(run_dir)
    metadata_path = _wandb_run_metadata_path(run_dir)
    run_id = wandb_config.run_id
    if run_id is None and allow_resume and wandb_config.resume != "never":
        run_id = _load_previous_run_id(run_dir)
    run = wandb.init(
        project=wandb_config.project,
        entity=wandb_config.entity,
        name=wandb_config.run_name or config.name,
        id=run_id,
        resume=wandb_config.resume,
        group=wandb_config.group,
        job_type=wandb_config.job_type,
        tags=wandb_config.tags or None,
        notes=wandb_config.notes,
        mode=wandb_config.mode,
        dir=wandb_config.dir or str(run_dir / "wandb"),
        config=config_to_dict(config),
    )
    run.define_metric("step")
    run.define_metric("train/*", step_metric="step")
    run.define_metric("validation/*", step_metric="step")
    run.define_metric("checkpoint/*", step_metric="step")
    run.define_metric("loss", step_metric="step")
    run.define_metric("val_loss", step_metric="step")
    run.define_metric("val_ppl", step_metric="step")
    run.define_metric("lr", step_metric="step")
    run.define_metric("grad_norm", step_metric="step")
    run.define_metric("toks_per_sec", step_metric="step")
    run.define_metric("step_time_seconds", step_metric="step")
    run.define_metric("max_memory_gib", step_metric="step")

    metadata = {
        "id": getattr(run, "id", run_id),
        "name": getattr(run, "name", wandb_config.run_name or config.name),
        "url": getattr(run, "url", None),
        "project": wandb_config.project,
        "entity": wandb_config.entity,
        "mode": wandb_config.mode,
    }
    atomic_write_json(metadata_path, metadata)

    logger = WandbLogger(
        module=wandb,
        run=run,
        upload_checkpoints=wandb_config.upload_checkpoints,
    )
    logger.update_summary(run_metadata)
    return logger
