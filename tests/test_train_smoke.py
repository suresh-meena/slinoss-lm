from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from slinoss_lm.config import load_config
from slinoss_lm.data import load_packed_meta


def test_real_dataset_contract_is_loadable() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = os.environ.get("FWEDU_DATA_ROOT")
    if not data_root:
        pytest.skip("Set FWEDU_DATA_ROOT to run the real-dataset smoke test.")
    dataset_root = Path(data_root).expanduser() / "data" / "fwedu-llama31-2k"
    if not (dataset_root / "meta.json").exists():
        pytest.skip(f"Packed dataset metadata not found at {dataset_root}.")
    config = load_config(
        [str(repo_root / "configs/experiments/fwedu-180m.yaml")],
        overrides=[f"data.root={str(Path(data_root).expanduser())!r}"],
    )
    meta = load_packed_meta(config.data)
    assert meta.seq_len == 2048
    assert meta.n_sequences == 48_828_125
    assert meta.n_tokens == 100_000_000_000
    assert meta.tokenizer_id == "meta-llama/Llama-3.1-8B"
    payload = json.loads((meta.root / "meta.json").read_text())
    assert payload["append_eos"] is True
