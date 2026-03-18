#!/usr/bin/env python3
"""
Download and prepare training data for BTN.

Run this on a CHEAP GPU pod with your network volume attached.
The data gets saved to network storage so your expensive B200 pod
can read it instantly without re-downloading.

Usage:
    # Default: download FineWeb-Edu (~50GB sample) to network storage
    python setup_data.py

    # Custom dataset and size
    python setup_data.py --dataset HuggingFaceFW/fineweb-edu --size 100GB

    # Use your own text files
    python setup_data.py --local_dir /path/to/my/text/files

    # Custom output path
    python setup_data.py --output /workspace/data/btn
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Download from HuggingFace
# ---------------------------------------------------------------------------

def download_hf_dataset(
    dataset_name: str,
    output_dir: str,
    size_gb: float = 50.0,
    split: str = "train",
):
    """Stream a HuggingFace dataset and save raw bytes to a memmap file."""
    from datasets import load_dataset

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    memmap_path = output_path / "bytes.bin"
    meta_path = output_path / "meta.txt"

    if memmap_path.exists():
        with open(meta_path) as f:
            existing_size = int(f.read().strip())
        print(f"Data already exists: {existing_size:,} bytes ({existing_size / 1e9:.2f} GB)")
        print(f"Delete {memmap_path} to re-download.")
        return

    target_bytes = int(size_gb * 1e9)
    print(f"Downloading {dataset_name} (target: {size_gb:.0f} GB)...")
    print(f"Streaming to {memmap_path}")
    print()

    # Stream dataset — never loads everything into RAM
    ds = load_dataset(dataset_name, split=split, streaming=True)

    # Figure out the text column name
    text_col = None
    for col in ["text", "content", "document", "passage"]:
        try:
            sample = next(iter(ds))
            if col in sample:
                text_col = col
                break
        except Exception:
            pass

    if text_col is None:
        # Try to detect from first sample
        sample = next(iter(ds))
        for key, val in sample.items():
            if isinstance(val, str) and len(val) > 50:
                text_col = key
                break

    if text_col is None:
        print(f"ERROR: Could not find text column in dataset. Columns: {list(sample.keys())}")
        sys.exit(1)

    print(f"Using text column: '{text_col}'")

    # Write to a temporary file, then rename
    tmp_path = output_path / "bytes.bin.tmp"
    total_bytes = 0
    t_start = time.time()
    t_last = t_start

    with open(tmp_path, "wb") as f:
        for i, example in enumerate(ds):
            text = example[text_col]
            if not text:
                continue

            raw = text.encode("utf-8")
            f.write(raw)
            total_bytes += len(raw)

            # Progress update every 5 seconds
            now = time.time()
            if now - t_last > 5.0:
                elapsed = now - t_start
                speed = total_bytes / elapsed / 1e6
                pct = total_bytes / target_bytes * 100
                print(
                    f"  {total_bytes / 1e9:.2f} / {size_gb:.0f} GB "
                    f"({pct:.1f}%) | {speed:.1f} MB/s | "
                    f"{elapsed:.0f}s elapsed",
                    flush=True,
                )
                t_last = now

            if total_bytes >= target_bytes:
                break

    # Finalize
    os.rename(tmp_path, memmap_path)
    with open(meta_path, "w") as f:
        f.write(str(total_bytes))

    # Also create the .cache structure that ByteDataset expects
    cache_dir = output_path / ".cache"
    cache_dir.mkdir(exist_ok=True)
    # Symlink so ByteDataset finds it
    cache_bytes = cache_dir / "bytes.bin"
    cache_meta = cache_dir / "meta.txt"
    if not cache_bytes.exists():
        os.symlink(memmap_path, cache_bytes)
    if not cache_meta.exists():
        os.symlink(meta_path, cache_meta)

    elapsed = time.time() - t_start
    print(f"\nDone! {total_bytes:,} bytes ({total_bytes / 1e9:.2f} GB) in {elapsed / 60:.1f} min")
    print(f"Saved to: {output_path}")


# ---------------------------------------------------------------------------
# Process local text files
# ---------------------------------------------------------------------------

def process_local_files(local_dir: str, output_dir: str):
    """Read local text files and convert to byte memmap."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cache_dir = output_path / ".cache"
    cache_dir.mkdir(exist_ok=True)
    memmap_path = cache_dir / "bytes.bin"
    meta_path = cache_dir / "meta.txt"

    if memmap_path.exists():
        with open(meta_path) as f:
            existing = int(f.read().strip())
        print(f"Cache already exists: {existing:,} bytes. Delete {memmap_path} to rebuild.")
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
    print(f"Writing to {memmap_path}...")

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

DATASETS = {
    "fineweb-edu": "HuggingFaceFW/fineweb-edu",
    "fineweb": "HuggingFaceFW/fineweb",
    "c4": "allenai/c4",
    "pile": "EleutherAI/the_pile",
    "openwebtext": "Skylion007/openwebtext",
    "redpajama": "togethercomputer/RedPajama-Data-1T-Sample",
}


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare BTN training data to network storage"
    )
    parser.add_argument(
        "--output", type=str, default="/workspace/data/btn",
        help="Output directory (should be on RunPod network volume)",
    )
    parser.add_argument(
        "--dataset", type=str, default="fineweb-edu",
        help=f"Dataset name. Shortcuts: {list(DATASETS.keys())}. Or any HuggingFace dataset path.",
    )
    parser.add_argument(
        "--size", type=str, default="50GB",
        help="Target download size (e.g., 10GB, 100GB, 1TB)",
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

    print("=" * 50)
    print("  BTN Data Setup")
    print("=" * 50)
    print(f"  Output: {args.output}")
    print()

    if args.local_dir:
        print(f"  Source: local files from {args.local_dir}")
        print("=" * 50)
        process_local_files(args.local_dir, args.output)
    else:
        dataset_path = DATASETS.get(args.dataset, args.dataset)
        print(f"  Source: {dataset_path}")
        print(f"  Size:   {size_gb:.0f} GB")
        print("=" * 50)
        download_hf_dataset(dataset_path, args.output, size_gb=size_gb)

    print("\nNext step: start training on your B200 pod:")
    print("  git clone <your-repo> && cd BAM && bash train.sh")


if __name__ == "__main__":
    main()
