#!/bin/bash
# =============================================================
# BTN Data Setup — Run on a CHEAP pod with network volume
#
# This downloads training data to your RunPod network storage.
# Once done, terminate this pod and start the B200 training pod.
#
# Usage:
#   git clone https://github.com/sunnytroo01/BAM.git
#   cd BAM
#   bash setup_data.sh
#
# Options (via environment variables):
#   DATASET=fineweb-edu  (or: fineweb, c4, pile, openwebtext, redpajama)
#   SIZE=50GB            (target download size)
#   OUTPUT=/workspace/data/btn
# =============================================================
set -euo pipefail

DATASET="${DATASET:-fineweb-edu}"
SIZE="${SIZE:-50GB}"
OUTPUT="${OUTPUT:-/workspace/data/btn}"

echo ""
echo "================================================"
echo "  BTN Data Setup"
echo "  Dataset:  $DATASET"
echo "  Size:     $SIZE"
echo "  Output:   $OUTPUT (network storage)"
echo "================================================"
echo ""

# Install deps
pip install -q torch numpy datasets huggingface_hub

# Download
python setup_data.py --dataset "$DATASET" --size "$SIZE" --output "$OUTPUT"

echo ""
echo "================================================"
echo "  DATA READY on network storage at: $OUTPUT"
echo ""
echo "  You can now terminate this cheap pod."
echo "  Start your B200 pod with the same network volume"
echo "  and run:  bash train.sh"
echo "================================================"
