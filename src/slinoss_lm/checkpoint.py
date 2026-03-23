from __future__ import annotations

import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .common import atomic_write_json, ensure_dir


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def checkpoint_dir(run_dir: Path, step: int) -> Path:
    return run_dir / "checkpoints" / f"step-{step:09d}"


def latest_marker(run_dir: Path) -> Path:
    return run_dir / "checkpoints" / "latest.json"


def save_checkpoint(
    *,
    run_dir: Path,
    step: int,
    payload: dict[str, Any],
    keep_last: int,
) -> Path:
    ckpt_root = ensure_dir(run_dir / "checkpoints")
    final_dir = checkpoint_dir(run_dir, step)
    tmp_dir = ckpt_root / f".step-{step:09d}.tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)
    trainer_path = tmp_dir / "trainer.pt"
    torch.save(payload, trainer_path)
    atomic_write_json(
        tmp_dir / "meta.json",
        {
            "step": step,
            "trainer": "trainer.pt",
            "tokens_consumed": int(payload["state"]["tokens_consumed"]),
            "sequences_consumed": int(payload["state"]["sequences_consumed"]),
        },
    )
    if final_dir.exists():
        shutil.rmtree(final_dir)
    os.replace(tmp_dir, final_dir)
    atomic_write_json(
        latest_marker(run_dir),
        {
            "step": step,
            "path": str(final_dir),
            "trainer": str(final_dir / "trainer.pt"),
        },
    )
    if keep_last > 0:
        checkpoints = sorted(
            p for p in ckpt_root.iterdir() if p.is_dir() and p.name.startswith("step-")
        )
        stale = checkpoints[:-keep_last]
        for path in stale:
            shutil.rmtree(path, ignore_errors=True)
    return final_dir


def find_latest_checkpoint(run_dir: Path) -> Path | None:
    marker = latest_marker(run_dir)
    if not marker.exists():
        return None
    import json

    payload = json.loads(marker.read_text())
    return Path(payload["trainer"])


def load_checkpoint(
    path: Path, map_location: str | torch.device = "cpu"
) -> dict[str, Any]:
    return torch.load(path, map_location=map_location, weights_only=False)
