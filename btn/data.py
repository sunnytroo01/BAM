"""
Byte-level binary data pipeline for BTN.

Reads raw text files, converts to byte streams, and chunks into
fixed-length sequences for next-byte prediction training.

No tokenizer. No vocabulary. Just bytes.
"""

import os
import glob
import mmap
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


class ByteDataset(Dataset):
    """
    Memory-mapped byte-level dataset.

    Reads text files, concatenates into a single byte stream stored as a
    numpy memmap file (for efficient random access without loading into RAM),
    and serves fixed-length chunks for autoregressive training.

    Each sample is (context_length + 1) bytes:
      - input:  bytes[0 : context_length]
      - target: bytes[1 : context_length + 1]
    """

    def __init__(
        self,
        data_dir: str,
        context_length: int,
        split: str = "train",
        val_fraction: float = 0.01,
    ):
        self.context_length = context_length
        self.seq_len = context_length + 1  # +1 for the target offset

        cache_dir = Path(data_dir) / ".cache"
        cache_dir.mkdir(exist_ok=True)
        memmap_path = cache_dir / "bytes.bin"
        meta_path = cache_dir / "meta.txt"

        # Build or load the byte stream
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
            print(f"[ByteDataset] Loaded cache: {total:,} bytes")

        # Memory-map the byte file (read-only)
        self.data = np.memmap(str(memmap_path), dtype=np.uint8, mode="r", shape=(total,))

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
        offset = self.start + idx * self.seq_len
        chunk = self.data[offset : offset + self.seq_len].astype(np.int64)
        return torch.from_numpy(chunk)

    @staticmethod
    def _read_all_files(data_dir: str) -> np.ndarray:
        """Read all text files in directory and concatenate as bytes."""
        patterns = ["*.txt", "*.md", "*.json", "*.jsonl", "*.csv", "*.html", "*.xml"]
        files = []
        for pattern in patterns:
            files.extend(glob.glob(os.path.join(data_dir, "**", pattern), recursive=True))

        if not files:
            raise FileNotFoundError(
                f"No text files found in {data_dir}. "
                f"Supported extensions: {patterns}"
            )

        files.sort()  # deterministic order
        print(f"[ByteDataset] Found {len(files)} files")

        chunks = []
        total_bytes = 0
        for f in files:
            try:
                with open(f, "rb") as fp:
                    raw = fp.read()
                chunks.append(np.frombuffer(raw, dtype=np.uint8))
                total_bytes += len(raw)
            except Exception as e:
                print(f"[ByteDataset] Warning: skipping {f}: {e}")

        return np.concatenate(chunks)


class SyntheticByteDataset(Dataset):
    """
    Synthetic random byte dataset for preflight validation and benchmarking.
    No disk I/O required.
    """

    def __init__(self, context_length: int, n_samples: int = 10_000):
        self.context_length = context_length
        self.seq_len = context_length + 1
        self.n_samples = n_samples

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.randint(0, 256, (self.seq_len,), dtype=torch.long)


def create_dataloader(
    data_dir: Optional[str],
    config,
    split: str = "train",
    world_size: int = 1,
    rank: int = 0,
    synthetic: bool = False,
) -> DataLoader:
    """
    Create a DataLoader for BTN training.

    Args:
        data_dir: path to directory of text files (ignored if synthetic=True)
        config: BTNConfig
        split: "train" or "val"
        world_size: total number of GPUs
        rank: this GPU's rank
        synthetic: use random data (for preflight checks)
    """
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
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
        )
        shuffle = False  # sampler handles shuffling

    return DataLoader(
        dataset,
        batch_size=config.micro_batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
