from __future__ import annotations

import json

from slinoss_lm.config import load_config
from slinoss_lm.data import load_packed_meta


def test_real_dataset_contract_is_loadable() -> None:
    config = load_config(
        [
            "/home/b/projects/slinoss-lm/configs/experiments/fwedu-180m.yaml",
            "/home/b/projects/slinoss-lm/configs/runtime/ampere.yaml",
        ],
        overrides=["data.root='/run/media/b/T7 Shield/fineweb'"],
    )
    meta = load_packed_meta(config.data)
    assert meta.seq_len == 2048
    assert meta.n_sequences == 48_828_125
    assert meta.n_tokens == 100_000_000_000
    assert meta.tokenizer_id == "meta-llama/Llama-3.1-8B"
    payload = json.loads((meta.root / "meta.json").read_text())
    assert payload["append_eos"] is True
