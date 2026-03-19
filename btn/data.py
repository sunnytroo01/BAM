"""
Byte-level binary data pipeline — optimized.

Optimizations:
  - Returns uint8 tensors (8x smaller H2D transfer, cast to long on GPU)
  - persistent_workers=True (no worker respawn between epochs)
  - prefetch_factor=4 (deeper prefetch queue)
  - Loads dataset into RAM by default (eliminates network storage page faults)
  - num_workers scales with CPU count
"""

import os
import glob
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class ByteDataset(Dataset):
    """
    Byte-level dataset. Returns uint8 tensors — cast to long on GPU in training loop.
    Loads into RAM by default to eliminate network storage page fault latency.
    """

    def __init__(
        self,
        data_dir: str,
        context_length: int,
        split: str = "train",
        val_fraction: float = 0.01,
        load_into_ram: bool = True,
    ):
        self.context_length = context_length
        self.seq_len = context_length + 1

        cache_dir = Path(data_dir) / ".cache"
        cache_dir.mkdir(exist_ok=True)
        memmap_path = cache_dir / "bytes.bin"
        meta_path = cache_dir / "meta.txt"

        if not memmap_path.exists():
            print(f"[ByteDataset] Building byte cache from {data_dir} ...")
            byte_stream = self._read_all_files(data_dir)
            total = len(byte_stream)
            fp = np.memmap(str(memmap_path), dtype=np.uint8, mode="w+", shape=(total,))
            fp[:] = byte_stream[:]
            fp.flush()
            with open(meta_path, "w") as f:
                f.write(str(total))
            print(f"[ByteDataset] Cached {total:,} bytes ({total / 1e9:.2f} GB)")
        else:
            with open(meta_path, "r") as f:
                total = int(f.read().strip())

        # Load into RAM: eliminates random page fault latency on network storage
        if load_into_ram:
            print(f"[ByteDataset] Loading {total / 1e9:.2f} GB into RAM...")
            self.data = np.fromfile(str(memmap_path), dtype=np.uint8)
            print(f"[ByteDataset] Loaded into RAM")
        else:
            self.data = np.memmap(str(memmap_path), dtype=np.uint8, mode="r", shape=(total,))
            print(f"[ByteDataset] Using memmap ({total / 1e9:.2f} GB)")

        # Train/val split
        n_val = max(int(total * val_fraction), self.seq_len * 10)
        if split == "train":
            self.start = 0
            self.end = total - n_val
        else:
            self.start = total - n_val
            self.end = total

        self.n_samples = (self.end - self.start - self.seq_len) // self.seq_len

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Random-offset sampling: instead of fixed chunk boundaries,
        # pick a random start within the range. More data diversity per epoch,
        # no wasted compute on the same fixed splits.
        max_start = self.end - self.seq_len
        if self.start < max_start:
            # Use idx as seed-offset for reproducibility within epoch
            offset = self.start + (idx * 104729) % (max_start - self.start)
        else:
            offset = self.start
        chunk = self.data[offset : offset + self.seq_len]
        if not chunk.flags['C_CONTIGUOUS']:
            chunk = np.ascontiguousarray(chunk)
        return torch.from_numpy(chunk.copy())

    @staticmethod
    def _read_all_files(data_dir: str) -> np.ndarray:
        patterns = ["*.txt", "*.md", "*.json", "*.jsonl", "*.csv", "*.html", "*.xml"]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(data_dir, "**", pattern), recursive=True))
        if not files:
            raise FileNotFoundError(f"No text files found in {data_dir}")
        files.sort()
        print(f"[ByteDataset] Found {len(files)} files")
        chunks = []
        for f in files:
            try:
                with open(f, "rb") as fp:
                    chunks.append(np.frombuffer(fp.read(), dtype=np.uint8))
            except Exception as e:
                print(f"[ByteDataset] Warning: skipping {f}: {e}")
        return np.concatenate(chunks)


class SyntheticByteDataset(Dataset):
    def __init__(self, context_length: int, n_samples: int = 10_000):
        self.seq_len = context_length + 1
        self.n_samples = n_samples

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.randint(0, 256, (self.seq_len,), dtype=torch.uint8)


def create_dataloader(
    data_dir: Optional[str],
    config,
    split: str = "train",
    world_size: int = 1,
    rank: int = 0,
    synthetic: bool = False,
) -> DataLoader:
    if synthetic:
        dataset = SyntheticByteDataset(config.context_length, n_samples=1000)
    else:
        dataset = ByteDataset(
            data_dir=data_dir,
            context_length=config.context_length,
            split=split,
        )

    sampler = None
    shuffle = split == "train"
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)
        shuffle = False

    n_workers = min(8, max(1, (os.cpu_count() or 4) // max(world_size, 1)))

    return DataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=n_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=n_workers > 0,
        prefetch_factor=4 if n_workers > 0 else None,
    )
