#!/usr/bin/env python3
"""
Download and prepare training data for BTN.

Run this on a CHEAP GPU pod with your network volume attached.

Usage:
    # 4TB diverse mix (recommended for 200B model)
    python setup_data.py --dataset 4tb-mix

    # Single source
    python setup_data.py --dataset fineweb-edu --size 100GB

    # Wikipedia only
    python setup_data.py --dataset wikipedia --size 22GB

    # Use your own text files
    python setup_data.py --local_dir /path/to/my/text/files
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

DATASETS = {
    "fineweb-edu": "HuggingFaceFW/fineweb-edu",
    "fineweb": "HuggingFaceFW/fineweb",
    "slimpajama": "cerebras/SlimPajama-627B",
    "c4": "allenai/c4",
    "openwebtext": "Skylion007/openwebtext",
    "redpajama": "togethercomputer/RedPajama-Data-1T-Sample",
    "wikipedia": "legacy-datasets/wikipedia",
}

DATASET_CONFIGS = {
    "legacy-datasets/wikipedia": "20220301.en",
    "wikimedia/wikipedia": "20231101.en",
    "allenai/c4": "en",
}

# 4TB diverse mix: high-quality web + diverse sources + reference
# Optimized for training 200B parameter models
DATA_MIX_4TB = [
    ("HuggingFaceFW/fineweb-edu", 2500),  # 2.5TB high-quality educational web
    ("cerebras/SlimPajama-627B",   900),  # 900GB diverse (CC, C4, GitHub, Books, ArXiv, Wiki)
    ("allenai/c4",                 300),  # 300GB cleaned Common Crawl
    ("Skylion007/openwebtext",      24),  # 24GB GPT-2 quality web
    ("legacy-datasets/wikipedia",   22),  # 22GB encyclopedia (high signal)
]
# Total: ~3,746GB ≈ 3.75TB (rounds to ~4TB with encoding overhead)


# ---------------------------------------------------------------------------
# Core download function
# ---------------------------------------------------------------------------

def stream_dataset_to_file(
    dataset_name: str,
    file_handle,
    target_bytes: int,
    start_bytes: int = 0,
):
    """Stream a HuggingFace dataset and write raw bytes to an open file."""
    from datasets import load_dataset

    ds_config = DATASET_CONFIGS.get(dataset_name)
    if ds_config:
        ds = load_dataset(dataset_name, ds_config, split="train", streaming=True)
    else:
        ds = load_dataset(dataset_name, split="train", streaming=True)

    # Detect text column
    sample = next(iter(ds))
    text_col = None
    for col in ["text", "content", "document", "passage"]:
        if col in sample:
            text_col = col
            break
    if text_col is None:
        for key, val in sample.items():
            if isinstance(val, str) and len(val) > 50:
                text_col = key
                break
    if text_col is None:
        print(f"  WARNING: No text column found in {dataset_name}, skipping")
        return 0

    written = 0
    t_start = time.time()
    t_last = t_start

    for example in ds:
        text = example[text_col]
        if not text:
            continue
        raw = text.encode("utf-8")
        file_handle.write(raw)
        written += len(raw)

        now = time.time()
        if now - t_last > 10.0:
            total = start_bytes + written
            speed = written / (now - t_start) / 1e6
            print(
                f"    {dataset_name}: {written / 1e9:.1f}/{target_bytes / 1e9:.0f} GB "
                f"(total: {total / 1e9:.1f} GB) | {speed:.0f} MB/s",
                flush=True,
            )
            t_last = now

        if written >= target_bytes:
            break

    return written


def download_single(dataset_name: str, output_dir: str, size_gb: float):
    """Download a single dataset."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    memmap_path = output_path / "bytes.bin"
    meta_path = output_path / "meta.txt"

    if memmap_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            existing = int(f.read().strip())
        print(f"Data already exists: {existing:,} bytes ({existing / 1e9:.2f} GB)")
        print(f"Delete {memmap_path} to re-download.")
        return

    target = int(size_gb * 1e9)
    print(f"Downloading {dataset_name} ({size_gb:.0f} GB)...")

    tmp_path = output_path / "bytes.bin.tmp"
    with open(tmp_path, "wb") as f:
        written = stream_dataset_to_file(dataset_name, f, target)

    _finalize(output_path, tmp_path, written)


def download_mix(mix: list, output_dir: str):
    """Download multiple datasets sequentially into one file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    memmap_path = output_path / "bytes.bin"
    meta_path = output_path / "meta.txt"

    if memmap_path.exists() and meta_path.exists():
        with open(meta_path) as f:
            existing = int(f.read().strip())
        print(f"Data already exists: {existing:,} bytes ({existing / 1e9:.2f} GB)")
        print(f"Delete {memmap_path} to re-download.")
        return

    total_target = sum(gb for _, gb in mix)
    print(f"Downloading {len(mix)}-source mix ({total_target:.0f} GB total)...")
    print()
    for ds_name, gb in mix:
        print(f"  {ds_name}: {gb} GB")
    print()

    tmp_path = output_path / "bytes.bin.tmp"
    total_written = 0
    t_start = time.time()

    with open(tmp_path, "wb") as f:
        for i, (ds_name, gb) in enumerate(mix):
            target = int(gb * 1e9)
            print(f"[{i+1}/{len(mix)}] {ds_name} ({gb} GB)...")
            written = stream_dataset_to_file(ds_name, f, target, start_bytes=total_written)
            total_written += written
            elapsed = time.time() - t_start
            print(
                f"  Done: {written / 1e9:.1f} GB from {ds_name} "
                f"(total: {total_written / 1e9:.1f} GB, {elapsed / 3600:.1f}h elapsed)"
            )
            print()

    _finalize(output_path, tmp_path, total_written)


def _finalize(output_path: Path, tmp_path: Path, total_bytes: int):
    """Rename tmp file and create cache structure."""
    memmap_path = output_path / "bytes.bin"
    meta_path = output_path / "meta.txt"

    os.rename(tmp_path, memmap_path)
    with open(meta_path, "w") as f:
        f.write(str(total_bytes))

    # Create .cache structure that ByteDataset expects
    cache_dir = output_path / ".cache"
    cache_dir.mkdir(exist_ok=True)
    cache_bytes = cache_dir / "bytes.bin"
    cache_meta = cache_dir / "meta.txt"
    # Copy instead of symlink (more portable, works everywhere)
    if not cache_bytes.exists():
        os.link(memmap_path, cache_bytes)  # hardlink = no extra disk space
    if not cache_meta.exists():
        os.link(meta_path, cache_meta)

    print(f"Done! {total_bytes:,} bytes ({total_bytes / 1e9:.1f} GB)")
    print(f"Saved to: {output_path}")


# ---------------------------------------------------------------------------
# Process local text files
# ---------------------------------------------------------------------------

def process_local_files(local_dir: str, output_dir: str):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_dir = output_path / ".cache"
    cache_dir.mkdir(exist_ok=True)
    memmap_path = cache_dir / "bytes.bin"
    meta_path = cache_dir / "meta.txt"

    if memmap_path.exists():
        with open(meta_path) as f:
            existing = int(f.read().strip())
        print(f"Cache exists: {existing:,} bytes. Delete {memmap_path} to rebuild.")
        return

    patterns = ["*.txt", "*.md", "*.json", "*.jsonl", "*.csv", "*.html"]
    files = []
    for p in patterns:
        files.extend(glob.glob(os.path.join(local_dir, "**", p), recursive=True))
    if not files:
        print(f"ERROR: No text files found in {local_dir}")
        sys.exit(1)

    files.sort()
    print(f"Found {len(files)} files in {local_dir}")

    total_bytes = 0
    chunks = []
    for f in files:
        try:
            with open(f, "rb") as fp:
                raw = fp.read()
            chunks.append(raw)
            total_bytes += len(raw)
        except Exception as e:
            print(f"  Warning: skipping {f}: {e}")

    print(f"Total: {total_bytes:,} bytes ({total_bytes / 1e9:.2f} GB)")
    data = np.concatenate([np.frombuffer(c, dtype=np.uint8) for c in chunks])
    fp = np.memmap(str(memmap_path), dtype=np.uint8, mode="w+", shape=(total_bytes,))
    fp[:] = data[:]
    fp.flush()
    with open(meta_path, "w") as f:
        f.write(str(total_bytes))
    print(f"Done! Data ready at {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare BTN training data to network storage"
    )
    parser.add_argument(
        "--output", type=str, default="/workspace/data/btn",
        help="Output directory (on RunPod network volume)",
    )
    parser.add_argument(
        "--dataset", type=str, default="fineweb-edu",
        help=(
            f"Dataset: {list(DATASETS.keys())} or '4tb-mix' for the full "
            f"4TB diverse training mix. Or any HuggingFace path."
        ),
    )
    parser.add_argument(
        "--size", type=str, default="50GB",
        help="Target download size (e.g., 10GB, 100GB, 1TB). Ignored for 4tb-mix.",
    )
    parser.add_argument(
        "--local_dir", type=str, default=None,
        help="Use local text files instead of downloading",
    )
    args = parser.parse_args()

    # Parse size
    size_str = args.size.upper().replace(" ", "")
    if size_str.endswith("TB"):
        size_gb = float(size_str[:-2]) * 1000
    elif size_str.endswith("GB"):
        size_gb = float(size_str[:-2])
    elif size_str.endswith("MB"):
        size_gb = float(size_str[:-2]) / 1000
    else:
        size_gb = float(size_str)

    print("=" * 60)
    print("  BTN Data Setup")
    print("=" * 60)
    print(f"  Output: {args.output}")
    print()

    if args.local_dir:
        print(f"  Source: local files from {args.local_dir}")
        print("=" * 60)
        process_local_files(args.local_dir, args.output)
    elif args.dataset == "4tb-mix":
        total_gb = sum(gb for _, gb in DATA_MIX_4TB)
        print(f"  Source: 4TB diverse training mix ({len(DATA_MIX_4TB)} sources)")
        print(f"  Size:   {total_gb:.0f} GB")
        print("=" * 60)
        download_mix(DATA_MIX_4TB, args.output)
    else:
        dataset_path = DATASETS.get(args.dataset, args.dataset)
        print(f"  Source: {dataset_path}")
        print(f"  Size:   {size_gb:.0f} GB")
        print("=" * 60)
        download_single(dataset_path, args.output, size_gb=size_gb)

    print()
    print("Next step: start training on your B200 pod:")
    print("  git clone https://github.com/sunnytroo01/BAM.git && cd BAM && bash train.sh")


if __name__ == "__main__":
    main()
