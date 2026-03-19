#!/usr/bin/env python3
"""
BTN Training Script — Fully optimized distributed training with DeepSpeed ZeRO-3.

Optimizations over naive training loop:
  - FIX: grad accum step counting (was counting micro-steps, not optimizer steps)
  - FusedAdam (single CUDA kernel for all Adam ops)
  - Deferred GPU sync: loss.item() only on log steps (eliminates 2 syncs/micro-step)
  - non_blocking host→device transfer
  - TF32 + medium matmul precision
  - torch.compile applied per-block (ZeRO-3 compatible)
  - bf16 autocast on evaluation
  - Async checkpoint cleanup (background thread)
  - Persistent val iterator (covers more data across evals)
"""

import argparse
import json
import math
import re
import shutil
import threading
import time
from pathlib import Path

import torch
import torch.distributed as dist
import deepspeed

from btn.config import BTNConfig
from btn.model import BinaryThinkingNet
from btn.data import create_dataloader

# BPB conversion constant
_BPB_SCALE = 8.0 / math.log(2)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_lr(step: int, config: BTNConfig) -> float:
    """WSD schedule: Warmup → Stable → Decay. Faster convergence than cosine.

    - Warmup: linear ramp to peak LR
    - Stable: hold at peak LR for 80% of training (allows early stopping)
    - Decay: linear decay to min_lr over final 20%
    """
    if step < config.warmup_steps:
        # Start from 1e-7 instead of 0 (strictly better — avoids wasted first step)
        min_warmup_lr = 1e-7
        return min_warmup_lr + (config.learning_rate - min_warmup_lr) * step / max(config.warmup_steps, 1)
    stable_end = int(config.total_steps * 0.80)
    if step < stable_end:
        return config.learning_rate
    # Linear decay over final 20%
    decay_steps = config.total_steps - stable_end
    progress = (step - stable_end) / max(decay_steps, 1)
    return config.learning_rate - progress * (config.learning_rate - config.min_lr)


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def log(msg: str):
    if is_main_process():
        print(msg, flush=True)


def get_world_size() -> int:
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def find_latest_checkpoint(output_dir: str) -> str | None:
    output_path = Path(output_dir)
    if not output_path.exists():
        return None
    ckpt_dirs = []
    for d in output_path.iterdir():
        if d.is_dir():
            match = re.match(r"step_(\d+)", d.name)
            if match:
                ckpt_dirs.append((int(match.group(1)), d))
    if not ckpt_dirs:
        return None
    ckpt_dirs.sort(key=lambda x: x[0])
    return str(ckpt_dirs[-1][1])


def cleanup_old_checkpoints(output_dir: str, max_keep: int):
    """Delete old checkpoints in a background thread (non-blocking)."""
    if max_keep <= 0:
        return
    output_path = Path(output_dir)
    if not output_path.exists():
        return
    ckpt_dirs = []
    for d in output_path.iterdir():
        if d.is_dir():
            match = re.match(r"step_(\d+)$", d.name)
            if match:
                ckpt_dirs.append((int(match.group(1)), d))
    if len(ckpt_dirs) <= max_keep:
        return
    ckpt_dirs.sort(key=lambda x: x[0])
    to_delete = ckpt_dirs[: len(ckpt_dirs) - max_keep]

    def _rm():
        for _, d in to_delete:
            try:
                shutil.rmtree(d)
            except OSError:
                pass

    threading.Thread(target=_rm, daemon=True).start()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    # ---- Perf flags (before any CUDA ops) ----
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")

    # ---- Config ----
    config_map = {
        "debug": BTNConfig.btn_debug,
        "1b": BTNConfig.btn_1b,
        "13b": BTNConfig.btn_13b,
        "70b": BTNConfig.btn_70b,
        "175b": BTNConfig.btn_175b,
    }
    config = config_map[args.config]()

    if args.micro_batch_size is not None:
        config.micro_batch_size = args.micro_batch_size
    if args.context_length is not None:
        config.context_length = args.context_length
    if args.total_steps is not None:
        config.total_steps = args.total_steps
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.save_interval is not None:
        config.save_interval = args.save_interval

    # ---- DeepSpeed init ----
    deepspeed.init_distributed()
    world_size = get_world_size()
    rank = get_rank()

    log(f"\n{'='*60}")
    log(f"  Binary Thinking Net (BTN) — Training")
    log(f"{'='*60}")
    log(f"  Config:    {config}")
    log(f"  GPUs:      {world_size}")
    log(f"  Data:      {args.data_dir}")
    log(f"  Output:    {args.output_dir}")
    log(f"{'='*60}\n")

    # ---- Gradient accumulation ----
    global_batch_samples = config.batch_size_bytes // config.context_length
    config.gradient_accumulation_steps = max(
        1, global_batch_samples // (config.micro_batch_size * world_size),
    )
    effective_batch = (
        config.micro_batch_size * world_size * config.gradient_accumulation_steps
    )
    effective_bytes = effective_batch * config.context_length

    log(f"  Micro batch:    {config.micro_batch_size} per GPU")
    log(f"  Grad accum:     {config.gradient_accumulation_steps} steps")
    log(f"  Global batch:   {effective_batch} seqs = {effective_bytes:,} bytes/step")
    log(f"  Save every:     {config.save_interval} steps (keep last {config.max_checkpoints})")

    # ---- Model ----
    log("\nInitializing model...")
    model = BinaryThinkingNet(config)
    log(f"  Parameters: {model.num_parameters():,} ({model.num_parameters() / 1e9:.2f}B)")

    # torch.compile per-block (ZeRO-3 compatible, unlike whole-model compile)
    if config.compile_model:
        log("  Compiling model blocks with torch.compile...")
        for block in model.blocks:
            block.assoc = torch.compile(block.assoc)
            block.ffn = torch.compile(block.ffn)

    # ---- DeepSpeed config ----
    ds_config_path = Path(__file__).parent / "ds_config.json"
    with open(ds_config_path) as f:
        ds_config = json.load(f)

    ds_config["train_micro_batch_size_per_gpu"] = config.micro_batch_size
    ds_config["gradient_accumulation_steps"] = config.gradient_accumulation_steps
    ds_config["train_batch_size"] = effective_batch
    ds_config["gradient_clipping"] = config.grad_clip

    if "optimizer" not in ds_config:
        ds_config["optimizer"] = {
            "type": "FusedAdam",
            "params": {
                "lr": config.learning_rate,
                "betas": [config.adam_beta1, config.adam_beta2],
                "eps": config.adam_eps,
                "weight_decay": config.weight_decay,
                "adam_w_mode": True,
            },
        }

    # ---- DeepSpeed engine ----
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )
    # CRITICAL: use engine's module for all forward/loss calls (ZeRO-3 shards params)
    model = model_engine.module

    # ---- Data ----
    log("\nLoading data...")
    use_synthetic = args.data_dir is None
    if use_synthetic:
        log("  WARNING: no --data_dir, using synthetic random data")

    train_loader = create_dataloader(
        data_dir=args.data_dir, config=config, split="train",
        world_size=world_size, rank=rank, synthetic=use_synthetic,
    )
    val_loader = create_dataloader(
        data_dir=args.data_dir, config=config, split="val",
        world_size=world_size, rank=rank, synthetic=use_synthetic,
    )
    val_iter = iter(val_loader)  # persistent iterator

    # ---- Auto-resume ----
    global_step = 0
    resume_path = args.resume or find_latest_checkpoint(args.output_dir)
    if resume_path:
        log(f"\nResuming from: {resume_path}")
        try:
            _, client_state = model_engine.load_checkpoint(resume_path)
            if client_state:
                global_step = client_state.get("global_step", 0)
            log(f"  Resumed at step {global_step}")
        except Exception as e:
            log(f"  Warning: could not load checkpoint ({e}), starting from scratch")
            global_step = 0
    else:
        log("\nNo checkpoint found, starting from scratch.")

    output_dir = Path(args.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training loop ----
    log(f"\nTraining from step {global_step} to {config.total_steps}...\n")

    train_iter = iter(train_loader)
    # Accumulate loss as GPU tensor (no .item() sync until log step)
    running_loss = torch.zeros(1, device=model_engine.device)
    micro_steps_since_log = 0
    t_start = time.time()
    t_last_log = t_start

    while global_step < config.total_steps:
        # ---- Get batch ----
        try:
            batch = next(train_iter)
        except StopIteration:
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(global_step)
            train_iter = iter(train_loader)
            batch = next(train_iter)

        # non_blocking: overlap H2D transfer with previous step's GPU tail
        # Data arrives as uint8 (8x smaller transfer), cast to long on GPU
        batch = batch.to(model_engine.device, non_blocking=True).long()

        # Curriculum learning: truncate batch to current stage's context length
        # Short sequences early → faster convergence in early training
        if config.curriculum:
            curr_ctx = config.context_length
            for frac, ctx in config.curriculum:
                if global_step < frac * config.total_steps:
                    curr_ctx = ctx
                    break
            # +1 for target offset, + n_aux for multi-byte targets
            max_offset = len(model.aux_heads) + 1
            batch = batch[:, :curr_ctx + max_offset]

        # ---- LR schedule ----
        lr = get_lr(global_step, config)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # ---- Forward + backward + step ----
        loss = model.compute_loss(batch, use_checkpoint=config.gradient_checkpointing)
        model_engine.backward(loss)
        model_engine.step()

        # Accumulate loss on GPU (no sync!)
        running_loss += loss.detach()
        micro_steps_since_log += 1

        # FIX: only count OPTIMIZER steps, not micro-steps
        if not model_engine.is_gradient_accumulation_boundary():
            continue
        global_step += 1

        # ---- Log (only on optimizer steps, sync loss here) ----
        if global_step % config.log_interval == 0 and micro_steps_since_log > 0:
            now = time.time()
            avg_loss = (running_loss / micro_steps_since_log).item()  # single GPU sync
            avg_bpb = avg_loss * _BPB_SCALE
            elapsed = now - t_start
            interval = now - t_last_log
            steps_in_interval = global_step % config.log_interval or config.log_interval
            bytes_per_sec = (steps_in_interval * effective_bytes) / max(interval, 1e-6)

            log(
                f"step {global_step:>7d}/{config.total_steps} | "
                f"loss {avg_loss:.4f} | bpb {avg_bpb:.3f} | "
                f"lr {lr:.2e} | "
                f"{bytes_per_sec / 1e6:.1f} MB/s | "
                f"{elapsed:.0f}s"
            )
            running_loss.zero_()
            micro_steps_since_log = 0
            t_last_log = now

        # ---- Eval ----
        if global_step % config.eval_interval == 0:
            eval_loss, eval_bpb, val_iter = evaluate(
                model_engine, model, val_iter, val_loader, config
            )
            log(f"  [EVAL] step {global_step} | loss {eval_loss:.4f} | bpb {eval_bpb:.3f}")

        # ---- Save checkpoint ----
        if global_step % config.save_interval == 0:
            ckpt_dir = output_dir / f"step_{global_step}"
            log(f"  Saving checkpoint -> {ckpt_dir}")
            model_engine.save_checkpoint(
                str(ckpt_dir),
                client_state={"global_step": global_step, "config": config},
            )
            if is_main_process():
                cleanup_old_checkpoints(str(output_dir), config.max_checkpoints)

    # ---- Final save ----
    ckpt_dir = output_dir / f"step_{global_step}_final"
    log(f"\nTraining complete! Saving final checkpoint -> {ckpt_dir}")
    model_engine.save_checkpoint(
        str(ckpt_dir),
        client_state={"global_step": global_step, "config": config},
    )
    log(f"Done. {global_step} steps in {(time.time() - t_start) / 3600:.1f} hours.")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(engine, model, val_iter, val_loader, config):
    """Eval with bf16 autocast. Returns (loss, bpb, updated_val_iter)."""
    model.eval()
    total_loss = 0.0
    n = 0

    for _ in range(config.eval_steps):
        try:
            batch = next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            batch = next(val_iter)

        batch = batch.to(engine.device, non_blocking=True).long()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = model.compute_loss(batch)
        total_loss += loss.item()
        n += 1

    model.train()
    avg_loss = total_loss / max(n, 1)
    # FIX: return val_iter so caller gets the updated iterator
    return avg_loss, avg_loss * _BPB_SCALE, val_iter


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Binary Thinking Net")
    parser.add_argument("--config", type=str, default="175b",
                        choices=["debug", "1b", "13b", "70b", "175b"])
    parser.add_argument("--data_dir", type=str, default="/workspace/data/btn")
    parser.add_argument("--output_dir", type=str, default="/workspace/checkpoints/btn")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
