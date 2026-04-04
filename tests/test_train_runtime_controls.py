from __future__ import annotations

import pytest

from slinoss_lm.config import TrainConfig
from slinoss_lm.train import (
    resolve_wall_clock_controls,
    should_request_stop_for_wall_clock,
)


def test_resolve_wall_clock_controls_uses_max_runtime_seconds() -> None:
    config = TrainConfig(max_runtime_seconds=3600)
    deadline, margin = resolve_wall_clock_controls(
        config,
        launch_time_unix=1_700_000_000.0,
        env={},
    )
    assert deadline == 1_700_003_600
    assert margin == 900


def test_resolve_wall_clock_controls_prefers_earliest_deadline() -> None:
    config = TrainConfig(
        max_runtime_seconds=7200, wall_clock_deadline_unix=1_700_009_500
    )
    deadline, margin = resolve_wall_clock_controls(
        config,
        launch_time_unix=1_700_000_000.0,
        env={
            "SLINOSS_WALL_CLOCK_DEADLINE_UNIX": "1700004500",
            "SLINOSS_WALL_CLOCK_EXIT_MARGIN_SECONDS": "300",
        },
    )
    assert deadline == 1_700_004_500
    assert margin == 300


def test_resolve_wall_clock_controls_rejects_invalid_max_runtime() -> None:
    config = TrainConfig(max_runtime_seconds=0)
    with pytest.raises(ValueError, match="train.max_runtime_seconds"):
        resolve_wall_clock_controls(config, launch_time_unix=1_700_000_000.0, env={})


def test_should_request_stop_for_wall_clock_honors_margin() -> None:
    assert not should_request_stop_for_wall_clock(
        now_unix=10_000,
        deadline_unix=11_000,
        margin_seconds=500,
    )
    assert should_request_stop_for_wall_clock(
        now_unix=10_500,
        deadline_unix=11_000,
        margin_seconds=500,
    )
    assert not should_request_stop_for_wall_clock(
        now_unix=10_500,
        deadline_unix=None,
        margin_seconds=500,
    )
