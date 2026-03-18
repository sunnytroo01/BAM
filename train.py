#!/usr/bin/env python3
"""
BTN Training Script — Distributed training with DeepSpeed ZeRO-3.

Scales to any number of GPUs. Saves checkpoints frequently to network storage
so you never lose more than a few minutes of work if a pod crashes.

Usage:
  # On RunPod B200 (just clone and go):
  bash train.sh

  # Or manually:
  deepspeed train.py --config 175b \\
      --data_dir /workspace/data/btn \\
      --output_dir /workspace/checkpoints/btn

  # Auto-resumes from latest checkpoint if output_dir has previous runs.
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import deepspeed

from btn.config import BTNConfig
from btn.model import BinaryThinkingNet
from btn.data import create_dataloader


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def get_lr(step: int, config: BTNConfig) -> float:
    """Cosine decay with linear warmup."""
    if step < config.warmup_steps:
        return config.learning_rate * step / max(config.warmup_steps, 1)
    progress = (step - config.warmup_steps) / max(config.total_steps - config.warmup_steps, 1)
    progress = min(progress, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)


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
    """Find the most recent checkpoint in output_dir for auto-resume."""
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    # DeepSpeed saves checkpoints as directories like step_500/, step_1000/
    ckpt_dirs = []
    for d in output_path.iterdir():
        if d.is_dir():
            match = re.match(r"step_(\d+)", d.name)
            if match:
                ckpt_dirs.append((int(match.group(1)), d))

    if not ckpt_dirs:
        return None

    # Return the one with the highest step number
    ckpt_dirs.sort(key=lambda x: x[0])
    latest = ckpt_dirs[-1][1]
    return str(latest)


def cleanup_old_checkpoints(output_dir: str, max_keep: int):
    """Delete old checkpoints, keeping only the most recent `max_keep`."""
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

    # Sort by step, delete oldest
    ckpt_dirs.sort(key=lambda x: x[0])
    to_delete = ckpt_dirs[: len(ckpt_dirs) - max_keep]
    for step_num, d in to_delete:
        try:
            shutil.rmtree(d)
            log(f"  Cleaned up old checkpoint: {d.name}")
        except OSError:
            pass  # network storage can be slow, don't crash


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train(args):
    # ---- Config ----
    config_map = {
        "debug": BTNConfig.btn_debug,
        "1b": BTNConfig.btn_1b,
        "13b": BTNConfig.btn_13b,
        "70b": BTNConfig.btn_70b,
        "175b": BTNConfig.btn_175b,
    }
    config = config_map[args.config]()

    # Override config with CLI args
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

    # ---- DeepSpeed initialization ----
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
    log(f"  Grad ckpt: {config.gradient_checkpointing}")
    log(f"{'='*60}\n")

    # ---- Compute gradient accumulation to hit target batch size ----
    global_batch_samples = config.batch_size_bytes // config.context_length
    config.gradient_accumulation_steps = max(
        1,
        global_batch_samples // (config.micro_batch_size * world_size),
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
            "type": "AdamW",
            "params": {
                "lr": config.learning_rate,
                "betas": [config.adam_beta1, config.adam_beta2],
                "eps": config.adam_eps,
                "weight_decay": config.weight_decay,
            },
        }

    # ---- DeepSpeed engine ----
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )

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

    # ---- Auto-resume from latest checkpoint ----
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

    # ---- Output directory ----
    output_dir = Path(args.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training loop ----
    log(f"\nTraining from step {global_step} to {config.total_steps}...\n")

    train_iter = iter(train_loader)
    total_bytes_processed = 0
    running_loss = 0.0
    running_bpb = 0.0
    log_steps = 0
    t_start = time.time()
    t_last_log = t_start

    while global_step < config.total_steps:
        # Get batch (with epoch handling for distributed sampler)
        try:
            batch = next(train_iter)
        except StopIteration:
            if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(global_step)
            train_iter = iter(train_loader)
            batch = next(train_iter)

        batch = batch.to(model_engine.device)

        # Learning rate schedule
        lr = get_lr(global_step, config)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Forward + backward + step
        loss, metrics = model.compute_loss(
            batch, use_checkpoint=config.gradient_checkpointing
        )
        model_engine.backward(loss)
        model_engine.step()

        # Accumulate metrics
        running_loss += loss.item()
        running_bpb += metrics["bpb"]
        total_bytes_processed += batch.numel()
        log_steps += 1
        global_step += 1

        # ---- Log ----
        if global_step % config.log_interval == 0 and log_steps > 0:
            now = time.time()
            avg_loss = running_loss / log_steps
            avg_bpb = running_bpb / log_steps
            elapsed = now - t_start
            interval = now - t_last_log
            bytes_per_sec = (log_steps * effective_bytes) / interval if interval > 0 else 0

            log(
                f"step {global_step:>7d}/{config.total_steps} | "
                f"loss {avg_loss:.4f} | bpb {avg_bpb:.3f} | "
                f"lr {lr:.2e} | "
                f"{bytes_per_sec / 1e6:.1f} MB/s | "
                f"{elapsed:.0f}s"
            )
            running_loss = 0.0
            running_bpb = 0.0
            log_steps = 0
            t_last_log = now

        # ---- Eval ----
        if global_step % config.eval_interval == 0:
            eval_loss, eval_bpb = evaluate(model_engine, model, val_loader, config)
            log(f"  [EVAL] step {global_step} | loss {eval_loss:.4f} | bpb {eval_bpb:.3f}")

        # ---- Save checkpoint (frequent, to network storage) ----
        if global_step % config.save_interval == 0:
            ckpt_dir = output_dir / f"step_{global_step}"
            log(f"  Saving checkpoint -> {ckpt_dir}")
            model_engine.save_checkpoint(
                str(ckpt_dir),
                client_state={"global_step": global_step, "config": config},
            )
            # Clean up old checkpoints (rolling window)
            if is_main_process():
                cleanup_old_checkpoints(str(output_dir), config.max_checkpoints)

    # ---- Final save ----
    ckpt_dir = output_dir / f"step_{global_step}_final"
    log(f"\nTraining complete! Saving final checkpoint -> {ckpt_dir}")
    model_engine.save_checkpoint(
        str(ckpt_dir),
        client_state={"global_step": global_step, "config": config},
    )

    total_time = time.time() - t_start
    log(f"Done. {global_step} steps in {total_time / 3600:.1f} hours.")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(engine, model, val_loader, config) -> tuple:
    model.eval()
    total_loss = 0.0
    total_bpb = 0.0
    n = 0

    for batch in val_loader:
        if n >= config.eval_steps:
            break
        batch = batch.to(engine.device)
        loss, metrics = model.compute_loss(batch)
        total_loss += loss.item()
        total_bpb += metrics["bpb"]
        n += 1

    model.train()
    return (total_loss / max(n, 1), total_bpb / max(n, 1))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Binary Thinking Net")
    parser.add_argument(
        "--config", type=str, default="175b",
        choices=["debug", "1b", "13b", "70b", "175b"],
    )
    parser.add_argument(
        "--data_dir", type=str, default="/workspace/data/btn",
        help="Path to prepared training data (on network storage)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="/workspace/checkpoints/btn",
        help="Checkpoint dir (on network storage). Auto-resumes if checkpoints exist.",
    )
    parser.add_argument("--resume", type=str, default=None, help="Explicit checkpoint path (overrides auto-resume)")

    # Config overrides
    parser.add_argument("--micro_batch_size", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--total_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--save_interval", type=int, default=None)

    # DeepSpeed
    parser.add_argument("--local_rank", type=int, default=-1)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
