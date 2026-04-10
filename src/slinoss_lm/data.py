from __future__ import annotations

import bisect
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from .config import DataConfig, RuntimeConfig


Batch = dict[str, torch.Tensor]


@dataclass
class PackedDatasetMeta:
    root: Path
    seq_len: int
    n_sequences: int
    n_tokens: int
    tokenizer_id: str
    shard_paths: list[Path]
    shard_sequences: list[int]
    cumulative_sequences: list[int]


def resolve_data_root(config: DataConfig) -> Path:
    if config.root:
        return Path(config.root).expanduser().resolve()
    value = os.environ.get(config.root_env)
    if not value:
        raise RuntimeError(
            f"Dataset root not configured. Set {config.root_env} or override data.root."
        )
    return Path(value).expanduser().resolve()


def load_packed_meta(
    config: DataConfig, dataset_root: Path | None = None
) -> PackedDatasetMeta:
    root = dataset_root or (resolve_data_root(config) / config.dataset_dir)
    meta = json.loads((root / "meta.json").read_text())
    shard_paths: list[Path] = []
    shard_sequences: list[int] = []
    cumulative: list[int] = []
    running = 0
    for shard in meta["shards"]:
        shard_paths.append(root / shard["file"])
        n_sequences = int(shard["n_sequences"])
        shard_sequences.append(n_sequences)
        running += n_sequences
        cumulative.append(running)
    return PackedDatasetMeta(
        root=root,
        seq_len=int(meta["seq_len"]),
        n_sequences=int(meta["n_sequences"]),
        n_tokens=int(meta["n_tokens_emitted"]),
        tokenizer_id=str(meta["tokenizer_id"]),
        shard_paths=shard_paths,
        shard_sequences=shard_sequences,
        cumulative_sequences=cumulative,
    )


class PackedSequenceDataset(Dataset[Batch]):
    def __init__(self, meta: PackedDatasetMeta) -> None:
        self.meta = meta
        self.seq_len = meta.seq_len
        self._memmaps: dict[int, Any] = {}

    def __len__(self) -> int:
        return self.meta.n_sequences

    def _locate(self, index: int) -> tuple[int, int]:
        shard_idx = bisect.bisect_right(self.meta.cumulative_sequences, index)
        prev = 0 if shard_idx == 0 else self.meta.cumulative_sequences[shard_idx - 1]
        row_idx = index - prev
        return shard_idx, row_idx

    def _get_memmap(self, shard_idx: int) -> Any:
        arr = self._memmaps.get(shard_idx)
        if arr is None:
            path = self.meta.shard_paths[shard_idx]
            arr = np.memmap(path, mode="r", dtype=np.int32).reshape(-1, self.seq_len)
            self._memmaps[shard_idx] = arr
        return arr

    def __getitem__(self, index: int) -> Batch:
        shard_idx, row_idx = self._locate(index)
        arr = self._get_memmap(shard_idx)
        input_ids = torch.from_numpy(np.asarray(arr[row_idx], dtype=np.int64)).long()
        return {"input_ids": input_ids, "labels": input_ids.clone()}


class DistributedBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        *,
        total_sequences: int,
        per_rank_batch_size: int,
        rank: int,
        world_size: int,
        start_sequence: int = 0,
    ) -> None:
        self.total_sequences = int(total_sequences)
        self.per_rank_batch_size = int(per_rank_batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.start_sequence = int(start_sequence)
        self.global_batch_size = self.per_rank_batch_size * self.world_size

    def __iter__(self) -> Iterator[list[int]]:
        base = self.start_sequence
        while base + self.global_batch_size <= self.total_sequences:
            local_start = base + self.rank * self.per_rank_batch_size
            yield list(range(local_start, local_start + self.per_rank_batch_size))
            base += self.global_batch_size

    def __len__(self) -> int:
        remaining = max(self.total_sequences - self.start_sequence, 0)
        return remaining // self.global_batch_size


def build_train_loader(
    *,
    meta: PackedDatasetMeta,
    runtime: RuntimeConfig,
    rank: int,
    world_size: int,
    start_sequence: int,
) -> DataLoader[Batch]:
    dataset = PackedSequenceDataset(meta)
    sampler = DistributedBatchSampler(
        total_sequences=meta.n_sequences,
        per_rank_batch_size=runtime.per_device_batch_size,
        rank=rank,
        world_size=world_size,
        start_sequence=start_sequence,
    )
    if runtime.dataloader_workers > 0:
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=runtime.dataloader_workers,
            pin_memory=runtime.pin_memory and torch.cuda.is_available(),
            persistent_workers=runtime.persistent_workers,
            prefetch_factor=runtime.prefetch_factor,
        )
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=runtime.pin_memory and torch.cuda.is_available(),
        persistent_workers=False,
    )


def build_eval_loader(
    *,
    meta: PackedDatasetMeta,
    batch_size: int,
    start_sequence: int,
    num_sequences: int,
    num_workers: int = 0,
) -> DataLoader[Batch]:
    dataset = PackedSequenceDataset(meta)
    start = (
        max(meta.n_sequences + start_sequence, 0)
        if start_sequence < 0
        else start_sequence
    )
    stop = min(start + num_sequences, meta.n_sequences)
    indices = list(range(start, stop))

    class _SubsetBatchSampler(Sampler[list[int]]):
        def __iter__(self) -> Iterator[list[int]]:
            batch: list[int] = []
            for index in indices:
                batch.append(index)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

        def __len__(self) -> int:
            return (len(indices) + batch_size - 1) // batch_size

    return DataLoader(
        dataset,
        batch_sampler=_SubsetBatchSampler(),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


class CudaPrefetcher:
    def __init__(self, loader: DataLoader[Batch], device: torch.device) -> None:
        self.loader = loader
        self.device = device
        self.stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )

    def __iter__(self) -> Iterator[Batch]:
        if self.stream is None:
            yield from self.loader
            return
        first = True
        next_batch: Batch | None = None
        for batch in self.loader:
            with torch.cuda.stream(self.stream):
                moved = {
                    key: value.to(self.device, non_blocking=True)
                    for key, value in batch.items()
                }
            if not first:
                torch.cuda.current_stream(self.device).wait_stream(self.stream)
                assert next_batch is not None
                yield next_batch
            else:
                first = False
            next_batch = moved
        if next_batch is not None:
            torch.cuda.current_stream(self.device).wait_stream(self.stream)
            yield next_batch
