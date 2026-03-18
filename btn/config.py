"""BTN model and training configuration."""

from dataclasses import dataclass
import math


@dataclass
class BTNConfig:
    """
    Binary Thinking Net configuration.

    SwiGLU FFN: 3 weight matrices at d_ff = (8/3)*d_model gives identical
    param count to standard 2-matrix FFN at d_ff = 4*d_model:
        3 * d * (8d/3) = 8d² = 2 * d * 4d  ✓

    175B default: d=12288, L=96, H=96, d_ff=32768 (SwiGLU)
    Per layer: 4d² + 3*d*d_ff + 2d ≈ 1.81B → 96 layers ≈ 174B
    """

    # --- Model architecture ---
    d_model: int = 12288
    n_layers: int = 96
    n_heads: int = 96
    d_ff: int = 32768           # (8/3)*d_model for SwiGLU (same params as 4x with GELU)
    context_length: int = 2048  # bytes (= 2KB of text)
    chunk_size: int = 64        # for chunked association (memory vs parallelism tradeoff)
    dropout: float = 0.0
    norm_eps: float = 1e-6

    # --- Training ---
    learning_rate: float = 6e-5       # GPT-3 175B used 0.6e-4
    min_lr: float = 6e-6              # 10% of max LR
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 375
    total_steps: int = 300_000
    batch_size_bytes: int = 3_276_800  # ~3.2M bytes per global batch
    micro_batch_size: int = 4          # per GPU, adjust to fit VRAM
    gradient_accumulation_steps: int = 1  # auto-computed at runtime
    seed: int = 42

    # --- Infrastructure ---
    gradient_checkpointing: bool = True
    bf16: bool = True
    compile_model: bool = True   # torch.compile for fused kernels

    # --- Logging / Checkpointing ---
    log_interval: int = 10
    eval_interval: int = 200
    save_interval: int = 50          # frequent saves to network storage
    eval_steps: int = 20
    max_checkpoints: int = 5         # rolling window: keep last N, delete older

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        return self.d_model // self.n_heads

    @property
    def estimated_params(self) -> int:
        """Estimate total parameter count."""
        per_layer = (
            4 * self.d_model ** 2           # Q, K, V, O projections
            + 3 * self.d_model * self.d_ff  # SwiGLU: w1, w2, w3
            + 2 * self.d_model              # 2x RMSNorm
        )
        total = (
            self.n_layers * per_layer
            + 8 * self.d_model                         # bit_in (8 -> d)
            + self.d_model * 8                          # bit_out (d -> 8)
            + self.context_length * self.d_model        # pos_emb
            + self.d_model                              # final norm
        )
        return total

    @property
    def estimated_params_b(self) -> float:
        return self.estimated_params / 1e9

    # ---- Presets ----
    @classmethod
    def btn_175b(cls) -> "BTNConfig":
        """175B parameter model (default)."""
        return cls()

    @classmethod
    def btn_70b(cls) -> "BTNConfig":
        """~70B parameter model."""
        return cls(
            d_model=8192, n_layers=80, n_heads=64, d_ff=21760,
            context_length=2048, learning_rate=1.0e-4,
        )

    @classmethod
    def btn_13b(cls) -> "BTNConfig":
        """~13B parameter model."""
        return cls(
            d_model=5120, n_layers=40, n_heads=40, d_ff=13568,
            context_length=2048, learning_rate=1.5e-4,
        )

    @classmethod
    def btn_1b(cls) -> "BTNConfig":
        """~1.3B parameter model for fast iteration."""
        return cls(
            d_model=2048, n_layers=24, n_heads=16, d_ff=5504,
            context_length=2048, learning_rate=2e-4,
        )

    @classmethod
    def btn_debug(cls) -> "BTNConfig":
        """~12M parameter model for local testing / preflight checks."""
        return cls(
            d_model=384, n_layers=8, n_heads=6, d_ff=1024,
            context_length=512, chunk_size=32,
            micro_batch_size=8, learning_rate=3e-4,
            total_steps=1000, warmup_steps=50,
            log_interval=1, eval_interval=100, save_interval=200,
            max_checkpoints=3, compile_model=False,
        )

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0

    def __repr__(self):
        return (
            f"BTNConfig(~{self.estimated_params_b:.1f}B params | "
            f"d={self.d_model}, L={self.n_layers}, H={self.n_heads}, "
            f"d_ff={self.d_ff}, ctx={self.context_length})"
        )
