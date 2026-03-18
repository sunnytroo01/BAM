#!/bin/bash
# =============================================================
# BTN Training — Run on B200 pod(s) with network volume
#
# Reads data from network storage, saves checkpoints to
# network storage every 50 steps. Auto-resumes if restarted.
#
# Usage:
#   git clone https://github.com/sunnytroo01/BAM.git
#   cd BAM
#   bash train.sh
#
# Options (via environment variables):
#   CONFIG=175b                             Model size
#   DATA_DIR=/workspace/data/btn            Where setup_data.sh put data
#   OUTPUT_DIR=/workspace/checkpoints/btn   Checkpoint dir (network storage)
#   MICRO_BATCH=4                           Per-GPU batch size (tune to VRAM)
#   SAVE_EVERY=50                           Checkpoint frequency (steps)
# =============================================================
set -euo pipefail

CONFIG="${CONFIG:-175b}"
DATA_DIR="${DATA_DIR:-/workspace/data/btn}"
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/checkpoints/btn}"
MICRO_BATCH="${MICRO_BATCH:-}"
SAVE_EVERY="${SAVE_EVERY:-}"

# Auto-detect GPUs
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)

echo ""
echo "================================================"
echo "  Binary Thinking Net (BTN) — Training"
echo "================================================"
echo "  Config:       $CONFIG"
echo "  GPUs:         $NUM_GPUS"
echo "  Data:         $DATA_DIR"
echo "  Checkpoints:  $OUTPUT_DIR"
echo "================================================"
echo ""

# ---- Check data exists ----
if [ ! -d "$DATA_DIR" ]; then
    echo "ERROR: Data directory not found: $DATA_DIR"
    echo ""
    echo "Did you run setup_data.sh first on a cheap pod?"
    echo "  bash setup_data.sh"
    echo ""
    exit 1
fi

# ---- Install deps ----
echo "[1/3] Installing dependencies..."
pip install -q -r requirements.txt

# ---- Preflight ----
echo "[2/3] Preflight check..."
python preflight.py --config "$CONFIG" --data_dir "$DATA_DIR" --full
if [ $? -ne 0 ]; then
    echo ""
    echo "PREFLIGHT FAILED. Fix the issues above before training."
    exit 1
fi

# ---- Build args ----
EXTRA_ARGS=""
if [ -n "${MICRO_BATCH}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --micro_batch_size ${MICRO_BATCH}"
fi
if [ -n "${SAVE_EVERY}" ]; then
    EXTRA_ARGS="${EXTRA_ARGS} --save_interval ${SAVE_EVERY}"
fi

# ---- Launch ----
echo ""
echo "[3/3] Launching training..."
echo ""

deepspeed --num_gpus "$NUM_GPUS" \
    train.py \
    --config "$CONFIG" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    $EXTRA_ARGS
