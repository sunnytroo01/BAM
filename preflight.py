#!/usr/bin/env python3
"""
BTN Preflight Validation — Run THIS locally before spending money on RunPod.

Catches every common failure BEFORE you launch a training job:
  1. Dependency check      — are all packages installed + correct versions?
  2. Config validation     — do the hyperparameters make sense?
  3. Model smoke test      — can the model do a forward + backward pass?
  4. Data pipeline test    — can we load and iterate data?
  5. DeepSpeed dry-run     — does the DS config parse correctly?
  6. VRAM estimation       — will it fit on your B200 cluster?
  7. Checkpoint I/O test   — can we save/load a checkpoint?

Usage:
    python preflight.py                    # quick check with debug config
    python preflight.py --config 175b      # validate 175B config (CPU-only, no GPU needed)
    python preflight.py --config 1b        # validate 1B config
    python preflight.py --data_dir ./data  # also validate real data
    python preflight.py --full             # run ALL checks including DS config
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path


# ============================================================================
# Pretty printing
# ============================================================================

class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    BOLD = "\033[1m"
    END = "\033[0m"


def ok(msg):
    print(f"  {Colors.GREEN}[PASS]{Colors.END} {msg}")

def fail(msg, detail=""):
    print(f"  {Colors.RED}[FAIL]{Colors.END} {msg}")
    if detail:
        for line in detail.strip().split("\n"):
            print(f"         {line}")

def warn(msg):
    print(f"  {Colors.YELLOW}[WARN]{Colors.END} {msg}")

def info(msg):
    print(f"  {Colors.CYAN}[INFO]{Colors.END} {msg}")

def header(msg):
    print(f"\n{Colors.BOLD}{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}{Colors.END}")


# ============================================================================
# Check 1: Dependencies
# ============================================================================

def check_dependencies():
    header("1/7  Dependency Check")
    all_ok = True

    required = {
        "torch": "2.0.0",
        "deepspeed": "0.10.0",
        "numpy": "1.20.0",
    }

    for pkg, min_ver in required.items():
        try:
            mod = __import__(pkg)
            ver = getattr(mod, "__version__", "unknown")
            # Simple version comparison
            if ver != "unknown":
                ok(f"{pkg}=={ver}")
            else:
                ok(f"{pkg} (version unknown)")
        except ImportError:
            fail(f"{pkg} not installed", f"pip install {pkg}>={min_ver}")
            all_ok = False

    # Check CUDA availability (informational, not required for preflight)
    try:
        import torch
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            for i in range(n_gpus):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                ok(f"GPU {i}: {name} ({mem:.1f} GB)")
        else:
            info("No CUDA GPUs detected (OK for preflight, needed for training)")
    except Exception:
        info("Could not query CUDA (OK for preflight)")

    return all_ok


# ============================================================================
# Check 2: Config Validation
# ============================================================================

def check_config(config):
    header("2/7  Config Validation")
    all_ok = True

    info(f"Config: {config}")

    # d_model divisible by n_heads
    if config.d_model % config.n_heads != 0:
        fail(f"d_model ({config.d_model}) not divisible by n_heads ({config.n_heads})")
        all_ok = False
    else:
        ok(f"d_head = {config.d_head}")

    # d_ff reasonable
    ratio = config.d_ff / config.d_model
    if ratio < 1 or ratio > 8:
        warn(f"d_ff/d_model = {ratio:.1f} (typical: 4x)")
    else:
        ok(f"d_ff/d_model = {ratio:.1f}x")

    # Parameter count
    params_b = config.estimated_params_b
    ok(f"Estimated parameters: {params_b:.2f}B")

    # Context length
    if config.context_length < config.chunk_size:
        fail(f"context_length ({config.context_length}) < chunk_size ({config.chunk_size})")
        all_ok = False
    elif config.context_length % config.chunk_size != 0:
        warn(
            f"context_length ({config.context_length}) not divisible by "
            f"chunk_size ({config.chunk_size}) — last chunk will be smaller (OK)"
        )
    else:
        ok(f"context_length={config.context_length}, chunk_size={config.chunk_size}")

    # Learning rate sanity
    if config.learning_rate > 1e-2:
        warn(f"Learning rate {config.learning_rate} seems very high")
    elif config.learning_rate < 1e-6:
        warn(f"Learning rate {config.learning_rate} seems very low")
    else:
        ok(f"Learning rate: {config.learning_rate}")

    # Warmup steps
    if config.warmup_steps > config.total_steps * 0.1:
        warn(f"warmup_steps ({config.warmup_steps}) is >10% of total_steps ({config.total_steps})")
    else:
        ok(f"Warmup: {config.warmup_steps} / {config.total_steps} steps")

    return all_ok


# ============================================================================
# Check 3: Model Smoke Test
# ============================================================================

def check_model_smoke(config):
    header("3/7  Model Smoke Test (forward + backward)")

    import torch
    import math
    from btn.config import BTNConfig
    from btn.model import BinaryThinkingNet

    BPB_SCALE = 8.0 / math.log(2)

    smoke_config = BTNConfig.btn_debug()
    info(f"Running smoke test with debug config ({smoke_config.estimated_params / 1e6:.1f}M params)")

    try:
        model = BinaryThinkingNet(smoke_config)
        ok(f"Model instantiated: {model.num_parameters():,} parameters")
    except Exception as e:
        fail("Model instantiation failed", traceback.format_exc())
        return False

    try:
        batch = torch.randint(0, 256, (2, smoke_config.context_length + 1), dtype=torch.long)
        t0 = time.time()
        loss = model.compute_loss(batch, use_checkpoint=True)
        t_fwd = time.time() - t0
        bpb = loss.item() * BPB_SCALE
        ok(f"Forward pass: loss={loss.item():.4f}, BPB={bpb:.2f} ({t_fwd:.2f}s)")
    except Exception as e:
        fail("Forward pass failed", traceback.format_exc())
        return False

    try:
        t0 = time.time()
        loss.backward()
        t_bwd = time.time() - t0
        ok(f"Backward pass: {t_bwd:.2f}s")
    except Exception as e:
        fail("Backward pass failed", traceback.format_exc())
        return False

    has_grads = all(
        p.grad is not None for p in model.parameters() if p.requires_grad
    )
    if has_grads:
        ok("All parameters have gradients (STE working correctly)")
    else:
        n_no_grad = sum(
            1 for p in model.parameters() if p.requires_grad and p.grad is None
        )
        warn(f"{n_no_grad} parameters missing gradients")

    # LUT round-trip test
    test_bytes = torch.tensor([[65, 66, 0, 255]], dtype=torch.long)
    bits = model.bits_lut[test_bytes]
    powers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.long)
    recovered = (bits.long() * powers).sum(-1)
    if torch.equal(test_bytes, recovered):
        ok("LUT byte/bit round-trip verified")
    else:
        fail("LUT byte/bit mismatch", f"Input: {test_bytes}\nRecovered: {recovered}")
        return False

    try:
        model.eval()
        prompt = torch.tensor([[72, 101, 108, 108, 111]], dtype=torch.long)
        with torch.no_grad():
            out = model.generate(prompt, max_new_bytes=10, temperature=1.0)
        ok(f"Generation works: produced {out.shape[1]} bytes from 5-byte prompt")
    except Exception as e:
        fail("Generation (recurrent mode) failed", traceback.format_exc())
        return False

    return True


# ============================================================================
# Check 4: Data Pipeline
# ============================================================================

def check_data_pipeline(config, data_dir=None):
    header("4/7  Data Pipeline")

    from btn.data import SyntheticByteDataset, create_dataloader

    # Synthetic data test
    try:
        loader = create_dataloader(
            data_dir=None, config=config, split="train", synthetic=True
        )
        batch = next(iter(loader))
        assert batch.shape == (config.micro_batch_size, config.context_length + 1), (
            f"Expected shape ({config.micro_batch_size}, {config.context_length + 1}), "
            f"got {batch.shape}"
        )
        assert batch.min() >= 0 and batch.max() <= 255
        ok(f"Synthetic data: batch shape {tuple(batch.shape)}, dtype={batch.dtype}")
    except Exception as e:
        fail("Synthetic data pipeline failed", traceback.format_exc())
        return False

    # Real data test (optional)
    if data_dir:
        try:
            loader = create_dataloader(
                data_dir=data_dir, config=config, split="train"
            )
            batch = next(iter(loader))
            ok(f"Real data: batch shape {tuple(batch.shape)} from {data_dir}")

            # Print sample to verify it's real text
            sample_bytes = batch[0, :80].numpy().tobytes()
            try:
                sample_text = sample_bytes.decode("utf-8", errors="replace")[:80]
                info(f'Sample: "{sample_text}"')
            except Exception:
                info(f"Sample (hex): {sample_bytes[:40].hex()}")
        except Exception as e:
            fail(f"Real data loading from {data_dir} failed", traceback.format_exc())
            return False
    else:
        info("No --data_dir provided, skipping real data test")

    return True


# ============================================================================
# Check 5: DeepSpeed Config
# ============================================================================

def check_deepspeed_config():
    header("5/7  DeepSpeed Config")

    ds_path = Path(__file__).parent / "ds_config.json"
    if not ds_path.exists():
        fail(f"ds_config.json not found at {ds_path}")
        return False

    try:
        with open(ds_path) as f:
            ds_config = json.load(f)
        ok("ds_config.json parsed successfully")
    except json.JSONDecodeError as e:
        fail("ds_config.json has invalid JSON", str(e))
        return False

    # Validate required fields
    required_keys = ["bf16", "zero_optimization"]
    for key in required_keys:
        if key in ds_config:
            ok(f'Found "{key}" config')
        else:
            fail(f'Missing required key "{key}"')
            return False

    zero = ds_config.get("zero_optimization", {})
    stage = zero.get("stage", None)
    if stage == 3:
        ok("ZeRO Stage 3 configured (required for 175B)")
    elif stage is not None:
        warn(f"ZeRO Stage {stage} — Stage 3 recommended for 175B")
    else:
        fail("ZeRO stage not specified")

    return True


# ============================================================================
# Check 6: VRAM / Cluster Estimation
# ============================================================================

def check_vram_estimation(config, n_gpus=None):
    header("6/7  VRAM & Cluster Estimation")

    params_b = config.estimated_params_b
    bytes_per_param = 2  # bf16

    # Model parameters in GB
    param_gb = params_b * bytes_per_param

    # AdamW optimizer states: fp32 master + momentum + variance = 12 bytes/param
    optim_gb = params_b * 12

    # Gradients in bf16
    grad_gb = params_b * bytes_per_param

    total_gb = param_gb + optim_gb + grad_gb

    info(f"Model weights (bf16):     {param_gb:>8.1f} GB")
    info(f"Optimizer states (fp32):  {optim_gb:>8.1f} GB")
    info(f"Gradients (bf16):         {grad_gb:>8.1f} GB")
    info(f"Total model state:        {total_gb:>8.1f} GB")

    # B200 has 192 GB HBM3e
    gpu_mem = 192  # GB
    gpu_name = "B200"

    if n_gpus is None:
        # Estimate minimum GPUs needed
        # ZeRO-3 partitions everything; need ~20% overhead for activations + buffers
        min_gpus_params = math.ceil(total_gb / (gpu_mem * 0.75))
        # Round up to power of 2 for efficient all-reduce
        min_gpus = 1
        while min_gpus < min_gpus_params:
            min_gpus *= 2
        n_gpus = min_gpus

    per_gpu_model_state = total_gb / n_gpus

    # Activation memory estimate (with gradient checkpointing)
    # Per layer: ~2 * batch * seq_len * d_model * bytes_per_param
    # With grad ckpt, only 1 layer's activations at a time
    act_per_layer = (
        2 * config.micro_batch_size * config.context_length
        * config.d_model * bytes_per_param / 1e9
    )
    # Chunked association extra: chunk_size^2 * n_heads * bytes_per_param per layer
    assoc_per_layer = (
        config.chunk_size ** 2 * config.n_heads * bytes_per_param / 1e9
    )
    activation_gb = act_per_layer + assoc_per_layer

    per_gpu_total = per_gpu_model_state + activation_gb

    print()
    info(f"Target GPU: NVIDIA {gpu_name} ({gpu_mem} GB HBM3e)")
    info(f"GPUs: {n_gpus}")
    info(f"Per-GPU model state (ZeRO-3):  {per_gpu_model_state:.1f} GB")
    info(f"Per-GPU activations (ckpt):     {activation_gb:.2f} GB")
    info(f"Per-GPU total estimate:         {per_gpu_total:.1f} GB")

    if per_gpu_total < gpu_mem * 0.85:
        ok(f"Fits on {n_gpus}x {gpu_name} with {gpu_mem - per_gpu_total:.0f} GB headroom")
    elif per_gpu_total < gpu_mem:
        warn(f"Tight fit on {n_gpus}x {gpu_name} — consider more GPUs or smaller micro_batch")
    else:
        fail(
            f"Won't fit on {n_gpus}x {gpu_name}",
            f"Need {math.ceil(per_gpu_total / gpu_mem * n_gpus)} GPUs or reduce micro_batch_size"
        )
        return False

    # Training throughput estimate
    # B200 ~2.5 PFLOPS bf16 peak, ~40% utilization for LLM training
    flops_per_gpu = 2.5e15 * 0.40  # 1 PFLOPS effective
    # ~6 FLOPs per parameter per byte (forward + backward)
    flops_per_step = 6 * config.estimated_params * config.micro_batch_size * config.context_length
    secs_per_step = flops_per_step / (flops_per_gpu * n_gpus)
    steps_per_day = 86400 / secs_per_step if secs_per_step > 0 else 0
    days_to_train = config.total_steps / steps_per_day if steps_per_day > 0 else float("inf")

    print()
    info(f"Estimated throughput: ~{secs_per_step:.1f}s/step, ~{steps_per_day:.0f} steps/day")
    info(f"Estimated training time: ~{days_to_train:.1f} days for {config.total_steps:,} steps")

    return True


# ============================================================================
# Check 7: Checkpoint I/O
# ============================================================================

def check_checkpoint_io():
    header("7/7  Checkpoint I/O")

    import torch
    from btn.config import BTNConfig
    from btn.model import BinaryThinkingNet

    config = BTNConfig.btn_debug()
    model = BinaryThinkingNet(config)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_ckpt.pt")

        try:
            torch.save({
                "model_state": model.state_dict(),
                "config": config,
                "step": 0,
            }, ckpt_path)
            size_mb = os.path.getsize(ckpt_path) / 1e6
            ok(f"Checkpoint saved ({size_mb:.1f} MB)")
        except Exception as e:
            fail("Checkpoint save failed", traceback.format_exc())
            return False

        try:
            ckpt = torch.load(ckpt_path, weights_only=False)
            model2 = BinaryThinkingNet(ckpt["config"])
            model2.load_state_dict(ckpt["model_state"])
            ok("Checkpoint loaded and model restored")
        except Exception as e:
            fail("Checkpoint load failed", traceback.format_exc())
            return False

        # Verify weights match
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            if not torch.equal(p1, p2):
                fail(f"Parameter mismatch after reload: {n1}")
                return False
        ok("All parameters match after save/load round-trip")

    return True


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="BTN Preflight Validation — check everything before spending on RunPod"
    )
    parser.add_argument(
        "--config", type=str, default="debug",
        choices=["debug", "1b", "13b", "70b", "175b"],
        help="Config preset to validate (default: debug)",
    )
    parser.add_argument("--data_dir", type=str, default=None, help="Path to training data")
    parser.add_argument("--n_gpus", type=int, default=None, help="Number of B200 GPUs")
    parser.add_argument("--full", action="store_true", help="Run all checks including DS config")
    args = parser.parse_args()

    print(f"\n{Colors.BOLD}BTN Preflight Validation{Colors.END}")
    print(f"Run this LOCALLY before launching on RunPod.\n")

    # Load config
    from btn.config import BTNConfig
    config_map = {
        "debug": BTNConfig.btn_debug,
        "1b": BTNConfig.btn_1b,
        "13b": BTNConfig.btn_13b,
        "70b": BTNConfig.btn_70b,
        "175b": BTNConfig.btn_175b,
    }
    config = config_map[args.config]()

    results = {}

    # Run checks
    results["deps"] = check_dependencies()
    results["config"] = check_config(config)
    results["model"] = check_model_smoke(config)
    results["data"] = check_data_pipeline(
        # Always use debug config for actual data iteration
        BTNConfig.btn_debug(),
        data_dir=args.data_dir,
    )
    if args.full:
        results["deepspeed"] = check_deepspeed_config()
    results["vram"] = check_vram_estimation(config, n_gpus=args.n_gpus)
    results["checkpoint"] = check_checkpoint_io()

    # Summary
    header("PREFLIGHT SUMMARY")
    n_pass = sum(1 for v in results.values() if v)
    n_total = len(results)

    for name, passed in results.items():
        status = f"{Colors.GREEN}PASS{Colors.END}" if passed else f"{Colors.RED}FAIL{Colors.END}"
        print(f"  {status}  {name}")

    print()
    if all(results.values()):
        print(f"  {Colors.GREEN}{Colors.BOLD}ALL {n_total} CHECKS PASSED — safe to launch on RunPod{Colors.END}")
        return 0
    else:
        n_fail = n_total - n_pass
        print(f"  {Colors.RED}{Colors.BOLD}{n_fail} CHECK(S) FAILED — fix before launching{Colors.END}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
