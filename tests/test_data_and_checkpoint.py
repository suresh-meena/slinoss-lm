from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from slinoss_lm.checkpoint import load_checkpoint, save_checkpoint
from slinoss_lm.config import DataConfig
from slinoss_lm.data import PackedSequenceDataset, load_packed_meta


def test_packed_dataset_reads_rows(tmp_path: Path) -> None:
    root = tmp_path / "packed"
    shards = root / "shards"
    shards.mkdir(parents=True)
    seq_len = 8
    rows = np.arange(24, dtype=np.int32).reshape(3, seq_len)
    shard_path = shards / "part-000000.bin"
    rows.tofile(shard_path)
    meta = {
        "seq_len": seq_len,
        "n_sequences": 3,
        "n_tokens_emitted": 24,
        "tokenizer_id": "dummy",
        "shards": [{"file": "shards/part-000000.bin", "n_sequences": 3}],
    }
    (root / "meta.json").write_text(json.dumps(meta))
    packed = load_packed_meta(
        DataConfig(root=str(tmp_path), root_env="NOPE", dataset_dir="packed")
    )
    dataset = PackedSequenceDataset(packed)
    sample = dataset[1]
    assert torch.equal(sample["input_ids"], torch.tensor(rows[1], dtype=torch.long))
    assert torch.equal(sample["labels"], torch.tensor(rows[1], dtype=torch.long))


def test_checkpoint_roundtrip(tmp_path: Path) -> None:
    payload = {
        "model": {"x": torch.tensor([1.0])},
        "optimizer": {"y": 1},
        "scheduler": {"z": 2},
        "state": {"step": 3, "tokens_consumed": 4, "sequences_consumed": 5},
        "rng": {"python": None, "numpy": None, "torch": torch.get_rng_state()},
    }
    ckpt_dir = save_checkpoint(run_dir=tmp_path, step=3, payload=payload, keep_last=2)
    loaded = load_checkpoint(ckpt_dir / "trainer.pt")
    assert loaded["state"]["step"] == 3
    assert loaded["state"]["tokens_consumed"] == 4
