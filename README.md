# BTN — Binary Thinking Net

A tokenizer-free language model that operates directly on raw bytes using **binary associative memory** instead of attention.

**Key properties:**
- **No tokenizer** — input is raw bytes, converted to 8-bit binary vectors
- **O(1) context memory** at inference — fixed-size association matrix vs O(T×d) KV cache
- **Structural anti-plagiarism** — superposed outer products cannot reproduce training data verbatim
- **~175B parameters** — d=12288, 96 layers, 96 heads

## Architecture

```
byte (0-255) → 8-bit vector → Linear(8, d) → +pos_emb
    → [BinaryAssociativeMemory + FFN] × 96 layers
    → RMSNorm → Linear(d, 8) → 8-bit logits → next byte
```

The core innovation replaces multi-head attention with **binary associative memory**:

1. Project to Q, K, V
2. Binarize K, V to {-1, +1} using Straight-Through Estimator
3. Accumulate causal outer products: `M_t = Σ_{i≤t} sign(k_i) ⊗ sign(v_i)`
4. Retrieve: `output_t = M_t @ q_t`

At inference, the memory matrix M is fixed-size (`H × d_head²`) regardless of how long the conversation is. At 64K sequence length this saves **~8,000x memory** vs a transformer's KV cache.

## Quick Start (RunPod)

**Two commands. Two pods. That's it.**

### Step 1: Download data (cheap pod)

Spin up any cheap GPU pod with a **network volume** attached. Then:

```bash
git clone https://github.com/YOUR_USER/BAM.git
cd BAM
bash setup_data.sh
```

This downloads training data to your network volume at `/workspace/data/btn`. Terminate the cheap pod when done.

### Step 2: Train (B200 pod)

Spin up your B200 pod(s) with the **same network volume**. Then:

```bash
git clone https://github.com/YOUR_USER/BAM.git
cd BAM
bash train.sh
```

That's it. The script:
- Installs dependencies
- Runs preflight checks (catches errors before wasting GPU time)
- Auto-detects all GPUs
- Auto-resumes from the latest checkpoint if the pod restarts
- Saves checkpoints every 50 steps to network storage
- Cleans up old checkpoints automatically (keeps last 5)

## Configuration

Environment variables for `setup_data.sh`:
| Variable | Default | Description |
|----------|---------|-------------|
| `DATASET` | `fineweb-edu` | HuggingFace dataset (`fineweb`, `c4`, `pile`, `openwebtext`, `redpajama`) |
| `SIZE` | `50GB` | Target download size |
| `OUTPUT` | `/workspace/data/btn` | Output path on network volume |

Environment variables for `train.sh`:
| Variable | Default | Description |
|----------|---------|-------------|
| `CONFIG` | `175b` | Model size (`debug`, `1b`, `13b`, `70b`, `175b`) |
| `DATA_DIR` | `/workspace/data/btn` | Path to prepared data |
| `OUTPUT_DIR` | `/workspace/checkpoints/btn` | Checkpoint path on network volume |
| `MICRO_BATCH` | `4` | Per-GPU batch size (tune to fit VRAM) |
| `SAVE_EVERY` | `50` | Checkpoint frequency in steps |

## Model Configs

| Config | Params | d_model | Layers | Heads | Use |
|--------|--------|---------|--------|-------|-----|
| `debug` | 14M | 384 | 8 | 6 | Local testing |
| `1b` | 1.2B | 2048 | 24 | 16 | Fast iteration |
| `13b` | 12.6B | 5120 | 40 | 40 | Mid-scale |
| `70b` | 64.4B | 8192 | 80 | 64 | Large-scale |
| `175b` | 174B | 12288 | 96 | 96 | Full-scale |

## Preflight Checks

Run before training to catch errors without wasting GPU money:

```bash
python preflight.py --config 175b --full
```

Validates: dependencies, config math, forward/backward pass, data pipeline, DeepSpeed config, VRAM estimation, checkpoint I/O.

## Project Structure

```
BAM/
├── btn/
│   ├── config.py       # Model configs (debug → 175B)
│   ├── model.py        # Full BTN architecture
│   └── data.py         # Byte-level data pipeline
├── train.py            # Training loop (DeepSpeed ZeRO-3)
├── setup_data.py       # Download data to network storage
├── preflight.py        # Pre-flight validation
├── train.sh            # One-command training launcher
├── setup_data.sh       # One-command data download
├── ds_config.json      # DeepSpeed ZeRO Stage 3 config
└── requirements.txt
```

## Citation

```
@article{neal2026btn,
  title={Binary Associative Thinking: A Tokenizer-Free Language Model Architecture
         with Constant-Size Context Memory and Structural Anti-Plagiarism},
  author={Mark Neal},
  year={2026}
}
```
