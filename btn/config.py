"""BTN model and training configuration."""

from dataclasses import dataclass
import math


@dataclass
class BTNConfig:
    """
    Binary Thinking Net configuration.

    SwiGLU FFN: 3 weight matrices at d_ff = (8/3)*d_model
    DyT normalization: tanh(alpha*x)*weight (replaces RMSNorm, ~8% faster)
    Multi-byte prediction: n_aux_predict extra heads for bytes t+2..t+N
    Chunk size = d_head for balanced intra/cross FLOPs
    """

    # --- Model architecture ---
    d_model: int = 12288
    n_layers: int = 96
    n_heads: int = 96
    d_ff: int = 32768
    context_length: int = 2048
    chunk_size: int = 128        # = d_head: balanced intra/cross FLOPs, better tensor cores
    n_aux_predict: int = 3       # multi-byte: predict t+2, t+3, t+4 (0 = disabled)
    aux_loss_weight: float = 0.1 # weight for auxiliary prediction loss
    dropout: float = 0.0
    norm_eps: float = 1e-6

    # --- Training ---
    learning_rate: float = 6e-5
    min_lr: float = 6e-6
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    grad_clip: float = 1.0
    warmup_steps: int = 375
    total_steps: int = 300_000
    batch_size_bytes: int = 3_276_800
    micro_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    seed: int = 42

    # Curriculum learning: (fraction_of_training, context_length) stages
    # Train on short sequences first -> converge faster early on
    curriculum: tuple = ((0.10, 512), (0.30, 1024))  # rest at full context_length

    # --- Infrastructure ---
    gradient_checkpointing: bool = True
    bf16: bool = True
    compile_model: bool = True

    # --- Logging / Checkpointing ---
    log_interval: int = 10
    eval_interval: int = 200
    save_interval: int = 50
    eval_steps: int = 20
    max_checkpoints: int = 5

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    @property
    def estimated_params(self) -> int:
        per_layer = (
            4 * self.d_model ** 2
            + 3 * self.d_model * self.d_ff
            + 2 * self.d_model  # 2x DyT (alpha + weight)
        )
        total = (
            self.n_layers * per_layer
            + 8 * self.d_model
            + self.d_model * 8
            + self.d_model * 8 * self.n_aux_predict  # aux heads
            + self.context_length * self.d_model
            + self.d_model
        )
        return total

    @property
    def estimated_params_b(self) -> float:
        return self.estimated_params / 1e9

    @classmethod
    def btn_175b(cls) -> "BTNConfig":
        return cls()

    @classmethod
    def btn_70b(cls) -> "BTNConfig":
        return cls(
            d_model=8192, n_layers=80, n_heads=64, d_ff=21760,
            context_length=2048, chunk_size=128, learning_rate=1.0e-4,
        )

    @classmethod
    def btn_13b(cls) -> "BTNConfig":
        return cls(
            d_model=5120, n_layers=40, n_heads=40, d_ff=13568,
            context_length=2048, chunk_size=128, learning_rate=1.5e-4,
        )

    @classmethod
    def btn_1b(cls) -> "BTNConfig":
        return cls(
            d_model=2048, n_layers=24, n_heads=16, d_ff=5504,
            context_length=2048, chunk_size=128, learning_rate=2e-4,
        )

    @classmethod
    def btn_debug(cls) -> "BTNConfig":
        return cls(
            d_model=384, n_layers=8, n_heads=6, d_ff=1024,
            context_length=512, chunk_size=64,
            micro_batch_size=8, learning_rate=3e-4,
            total_steps=1000, warmup_steps=50,
            log_interval=1, eval_interval=100, save_interval=200,
            max_checkpoints=3, compile_model=False,
            n_aux_predict=3, curriculum=((0.10, 128), (0.30, 256)),
        )

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0

    def __repr__(self):
        return (
            f"BTNConfig(~{self.estimated_params_b:.1f}B params | "
            f"d={self.d_model}, L={self.n_layers}, H={self.n_heads}, "
            f"d_ff={self.d_ff}, ctx={self.context_length}, chunk={self.chunk_size})"
        )
