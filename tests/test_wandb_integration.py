from __future__ import annotations

import json
from pathlib import Path

import pytest

from slinoss_lm.config import ExperimentConfig, load_config
from slinoss_lm.wandb_integration import WandbLogger, build_wandb_logger


class FakeArtifact:
    def __init__(
        self, name: str, type: str, metadata: dict[str, int | float] | None = None
    ) -> None:
        self.name = name
        self.type = type
        self.metadata = metadata or {}
        self.directories: list[str] = []

    def add_dir(self, path: str) -> None:
        self.directories.append(path)


class FakeRun:
    def __init__(self, kwargs: dict[str, object]) -> None:
        self.id = str(kwargs.get("id") or "generated-run-id")
        self.name = str(kwargs.get("name") or "generated-run-name")
        self.url = f"https://wandb.invalid/{self.id}"
        self.summary: dict[str, object] = {}
        self.logged: list[tuple[dict[str, object], int | None]] = []
        self.metrics: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.artifacts: list[tuple[FakeArtifact, list[str] | None]] = []
        self.finish_calls: list[int] = []

    def define_metric(self, *args: object, **kwargs: object) -> None:
        self.metrics.append((args, kwargs))

    def log(self, payload: dict[str, object], step: int | None = None) -> None:
        self.logged.append((payload, step))

    def log_artifact(
        self, artifact: FakeArtifact, aliases: list[str] | None = None
    ) -> None:
        self.artifacts.append((artifact, aliases))

    def finish(self, exit_code: int = 0) -> None:
        self.finish_calls.append(exit_code)


class FakeWandb:
    Artifact = FakeArtifact

    def __init__(self) -> None:
        self.init_calls: list[dict[str, object]] = []
        self.runs: list[FakeRun] = []

    def init(self, **kwargs: object) -> FakeRun:
        self.init_calls.append(kwargs)
        run = FakeRun(kwargs)
        self.runs.append(run)
        return run


def test_wandb_config_loads_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
name: wandb-test
wandb:
  enabled: true
  project: slinoss-pretrain
  group: fw-100b
  tags: [slinoss, 180m]
  mode: offline
  resume: allow
  upload_checkpoints: true
"""
    )
    config = load_config([str(config_path)])
    assert config.wandb.enabled is True
    assert config.wandb.project == "slinoss-pretrain"
    assert config.wandb.group == "fw-100b"
    assert config.wandb.tags == ["slinoss", "180m"]
    assert config.wandb.mode == "offline"
    assert config.wandb.resume == "allow"
    assert config.wandb.upload_checkpoints is True


def test_build_wandb_logger_is_noop_when_disabled(tmp_path: Path) -> None:
    logger = build_wandb_logger(
        config=ExperimentConfig(),
        run_dir=tmp_path,
        allow_resume=False,
        run_metadata={"parameter_count": 1},
    )
    assert isinstance(logger, WandbLogger)
    assert logger.enabled is False
    assert not (tmp_path / "wandb-run.json").exists()


def test_build_wandb_logger_requires_package_when_enabled(tmp_path: Path) -> None:
    config = ExperimentConfig()
    config.wandb.enabled = True
    with pytest.raises(RuntimeError, match="requirements-wandb.txt"):
        build_wandb_logger(
            config=config,
            run_dir=tmp_path,
            allow_resume=False,
            run_metadata={"parameter_count": 1},
        )


def test_build_wandb_logger_reuses_run_id_and_logs_metrics(tmp_path: Path) -> None:
    config = ExperimentConfig()
    config.name = "fwedu-180m"
    config.wandb.enabled = True
    config.wandb.project = "slinoss-pretrain"
    config.wandb.group = "fw-100b"
    config.wandb.tags = ["slinoss", "180m"]
    config.wandb.mode = "offline"
    config.wandb.upload_checkpoints = True
    (tmp_path / "wandb-run.json").write_text(json.dumps({"id": "previous-run-id"}))
    fake_wandb = FakeWandb()

    logger = build_wandb_logger(
        config=config,
        run_dir=tmp_path,
        allow_resume=True,
        run_metadata={
            "parameter_count": 181145632,
            "world_size": 2,
            "global_batch_tokens": 524288,
        },
        module=fake_wandb,
    )

    assert logger.enabled is True
    init_call = fake_wandb.init_calls[0]
    assert init_call["id"] == "previous-run-id"
    assert init_call["name"] == "fwedu-180m"
    assert init_call["project"] == "slinoss-pretrain"
    assert init_call["group"] == "fw-100b"
    assert init_call["tags"] == ["slinoss", "180m"]
    assert init_call["dir"] == str(tmp_path / "wandb")

    logger.log_training(
        {
            "step": 10,
            "loss": 3.5,
            "lr": 2.5e-4,
            "grad_norm": 1.2,
            "tokens_per_second": 12345.0,
            "tokens_consumed": 5242880,
            "sequences_consumed": 2560,
        }
    )
    logger.log_validation(step=10, metrics={"loss": 3.4, "ppl": 29.9})

    checkpoint_dir = tmp_path / "checkpoints" / "step-000000010"
    checkpoint_dir.mkdir(parents=True)
    logger.log_checkpoint(
        step=10,
        checkpoint_dir=checkpoint_dir,
        state={
            "step": 10,
            "tokens_consumed": 5242880,
            "sequences_consumed": 2560,
        },
    )
    logger.finish(exit_code=0)

    run = fake_wandb.runs[0]
    metadata = json.loads((tmp_path / "wandb-run.json").read_text())
    assert metadata["id"] == "previous-run-id"
    assert run.summary["parameter_count"] == 181145632
    assert run.summary["latest_checkpoint_step"] == 10
    assert run.summary["latest_checkpoint_path"] == str(checkpoint_dir)
    assert run.finish_calls == [0]

    train_payload, train_step = run.logged[0]
    assert train_step == 10
    assert train_payload["step"] == 10
    assert train_payload["train/loss"] == 3.5
    assert train_payload["train/lr"] == 2.5e-4
    assert train_payload["train/tokens_per_second"] == 12345.0

    validation_payload, validation_step = run.logged[1]
    assert validation_step == 10
    assert validation_payload["validation/loss"] == 3.4
    assert validation_payload["validation/ppl"] == 29.9

    checkpoint_payload, checkpoint_step = run.logged[2]
    assert checkpoint_step == 10
    assert checkpoint_payload["checkpoint/saved"] == 1
    assert checkpoint_payload["checkpoint/tokens_consumed"] == 5242880

    artifact, aliases = run.artifacts[0]
    assert artifact.directories == [str(checkpoint_dir)]
    assert aliases == ["latest", "step-000000010"]
