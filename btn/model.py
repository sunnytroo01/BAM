"""
Binary Thinking Net (BTN) — fully optimized model.

Optimizations applied (all zero quality change):
  - Fused QKV projection (3 linears → 1, saves 67% input bandwidth)
  - Fused SwiGLU gate+up (2 linears → 1, same savings)
  - BinarizeSTE with scalar torch.where (no tensor allocation)
  - bytes↔bits via 256×8 LUT (single gather, no compute)
  - All einsums replaced with torch.matmul (direct cuBLAS dispatch)
  - Exclusive prefix sum via cumsum-subtract (no branching/zeros_like)
  - Generation: batch prefill prompt in parallel, then recurrent
  - Generation: pre-allocated output buffer (no O(n²) torch.cat)
  - Position IDs registered as buffer (no arange per forward)
  - Causal mask registered as buffer
  - compute_loss returns only loss tensor (no GPU sync inside model)
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from btn.config import BTNConfig


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    __constants__ = ["eps"]

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add_(self.eps).rsqrt_()
        return (x * norm).type_as(x) * self.weight


class BinarizeSTE(torch.autograd.Function):
    """Binarize to {-1, +1} with straight-through gradient estimator."""

    @staticmethod
    def forward(ctx, x):
        # Scalar args: no tensor allocation (unlike torch.ones_like)
        return torch.where(x >= 0, 1.0, -1.0)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def binarize(x: torch.Tensor) -> torch.Tensor:
    return BinarizeSTE.apply(x)


# ---------------------------------------------------------------------------
# Binary Associative Memory (replaces multi-head attention)
# ---------------------------------------------------------------------------

class BinaryAssociativeMemory(nn.Module):
    """
    Core: replaces softmax attention with binary associative memory.

    Optimizations:
    - Fused QKV: single nn.Linear(d, 3d) instead of 3 separate projections
    - All einsums replaced with torch.matmul (direct cuBLAS)
    - Exclusive prefix sum via cumsum-subtract (no branching)
    - Causal mask as persistent buffer
    """

    def __init__(self, config: BTNConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.chunk_size = config.chunk_size
        self.scale = 1.0 / math.sqrt(self.d_head)

        # Fused QKV: one matmul instead of three (saves 2 input reads from HBM)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.chunk_size, config.chunk_size)),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, D = x.shape
        H, d = self.n_heads, self.d_head

        # Fused QKV projection + reshape
        qkv = self.qkv_proj(x).view(B, T, 3, H, d)
        q, k, v = qkv.unbind(dim=2)  # each [B, T, H, d]
        q = q.transpose(1, 2) * self.scale  # [B, H, T, d]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Binarize K, V to {-1, +1} via STE
        k = binarize(k)
        v = binarize(v)

        if state is not None:
            out, new_state = self._recurrent(q, k, v, state)
        else:
            out, new_state = self._parallel_chunked(q, k, v)

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.o_proj(out), new_state

    def _parallel_chunked(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Vectorized parallel chunked association — all matmul, no einsum."""
        B, H, T, d = q.shape
        C = self.chunk_size

        # Pad to multiple of chunk_size
        remainder = T % C
        if remainder:
            pad = C - remainder
            q = F.pad(q, (0, 0, 0, pad))
            k = F.pad(k, (0, 0, 0, pad))
            v = F.pad(v, (0, 0, 0, pad))
            T_pad = T + pad
        else:
            T_pad = T
            pad = 0

        N = T_pad // C

        # Reshape: [B, H, N, C, d]
        qc = q.reshape(B, H, N, C, d)
        kc = k.reshape(B, H, N, C, d)
        vc = v.reshape(B, H, N, C, d)

        # 1. Intra-chunk causal (all N chunks in parallel)
        intra_scores = torch.matmul(qc, kc.transpose(-1, -2))  # [B,H,N,C,C]
        intra_scores = intra_scores * self.causal_mask
        intra = torch.matmul(intra_scores, vc)                  # [B,H,N,C,d]

        # 2. Cross-chunk: per-chunk outer product sums
        chunk_kv = torch.matmul(kc.transpose(-1, -2), vc)       # [B,H,N,d,d]

        # Exclusive prefix sum: prefix[n] = sum_{m<n} chunk_kv[m]
        prefix = torch.cumsum(chunk_kv, dim=2) - chunk_kv

        # Query into accumulated state from all prior chunks
        cross = torch.matmul(qc, prefix)                        # [B,H,N,C,d]

        # 3. Combine
        output = (intra + cross).reshape(B, H, T_pad, d)
        if pad:
            output = output[:, :, :T]

        # Final accumulated state
        final_state = prefix[:, :, -1] + chunk_kv[:, :, -1]
        return output, final_state

    def _recurrent(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recurrent O(1) inference. Optimized for T=1 (generation)."""
        B, H, T, d = q.shape

        if T == 1:
            # Fast path for generation: no loop, no list, no stack
            kt = k.squeeze(2)  # [B, H, d]
            vt = v.squeeze(2)
            qt = q.squeeze(2)
            state = state + kt.unsqueeze(-1) * vt.unsqueeze(-2)
            out = (qt.unsqueeze(-2) @ state).squeeze(-2)  # [B, H, d]
            return out.unsqueeze(2), state

        outputs = []
        for t in range(T):
            kt = k[:, :, t]
            vt = v[:, :, t]
            qt = q[:, :, t]
            state = state + kt.unsqueeze(-1) * vt.unsqueeze(-2)
            out_t = (qt.unsqueeze(-2) @ state).squeeze(-2)
            outputs.append(out_t)

        return torch.stack(outputs, dim=2), state


# ---------------------------------------------------------------------------
# SwiGLU FFN — fused gate+up projection
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """
    SwiGLU with fused gate+up projection (2 linears → 1).
    Saves one full read of the input activation from HBM.
    """

    def __init__(self, config: BTNConfig):
        super().__init__()
        # Fused: one matmul for both gate and up projections
        self.gate_up = nn.Linear(config.d_model, 2 * config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        return self.w2(F.silu(gate) * up)


# ---------------------------------------------------------------------------
# BTN Block
# ---------------------------------------------------------------------------

class BTNBlock(nn.Module):
    def __init__(self, config: BTNConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm1 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.assoc = BinaryAssociativeMemory(config)
        self.norm2 = RMSNorm(config.d_model, eps=config.norm_eps)
        self.ffn = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, new_state = self.assoc(self.norm1(x), state=state)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x, new_state


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class BinaryThinkingNet(nn.Module):
    """
    Binary Thinking Net — 175B-class tokenizer-free language model.

    Binary I/O: bytes → 8-bit LUT lookup → Linear(8,d) → +pos_emb
    → [BTNBlock × L] → RMSNorm → Linear(d,8) → 8-bit logits
    """

    def __init__(self, config: BTNConfig):
        super().__init__()
        self.config = config

        # Binary I/O
        self.bit_in = nn.Linear(8, config.d_model, bias=False)
        self.pos_emb = nn.Embedding(config.context_length, config.d_model)

        # Blocks
        self.blocks = nn.ModuleList([
            BTNBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # Output
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.bit_out = nn.Linear(config.d_model, 8, bias=False)

        # --- Constant buffers (moved to device once, never re-created) ---
        # 256×8 LUT: bytes_to_bits as a single gather operation
        lut = ((torch.arange(256).unsqueeze(-1) >> torch.arange(7, -1, -1)) & 1).float()
        self.register_buffer("bits_lut", lut, persistent=False)
        # Position IDs
        self.register_buffer(
            "position_ids", torch.arange(config.context_length), persistent=False
        )
        # Bit powers for generation
        self.register_buffer(
            "bit_powers",
            torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.long),
            persistent=False,
        )

        # Initialize weights
        self.apply(self._init_weights)
        for block in self.blocks:
            nn.init.normal_(
                block.assoc.o_proj.weight,
                mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers),
            )
            nn.init.normal_(
                block.ffn.w2.weight,
                mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers),
            )

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        byte_ids: torch.Tensor,
        states: Optional[list[torch.Tensor]] = None,
        use_checkpoint: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        B, T = byte_ids.shape

        # Binary I/O: LUT gather (no compute, just a table lookup)
        bits = self.bits_lut[byte_ids]      # [B, T, 8]
        x = self.bit_in(bits)               # [B, T, d_model]
        x = x + self.pos_emb(self.position_ids[:T])

        new_states = []
        for i, block in enumerate(self.blocks):
            layer_state = states[i] if states is not None else None

            if use_checkpoint and self.training:
                def _ckpt_fn(blk, st):
                    def fn(h):
                        return blk(h, state=st)
                    return fn
                x, new_state = checkpoint(
                    _ckpt_fn(block, layer_state), x, use_reentrant=False,
                )
            else:
                x, new_state = block(x, state=layer_state)

            new_states.append(new_state)

        x = self.norm(x)
        logits = self.bit_out(x)
        return logits, new_states

    def compute_loss(
        self, byte_ids: torch.Tensor, use_checkpoint: bool = False
    ) -> torch.Tensor:
        """
        Next-byte prediction loss. Returns ONLY the loss tensor.
        No .item() call — avoids GPU sync inside the model.
        BPB is computed in the training loop only on log steps.
        """
        input_bytes = byte_ids[:, :-1]
        target_bytes = byte_ids[:, 1:]
        logits, _ = self(input_bytes, use_checkpoint=use_checkpoint)
        target_bits = self.bits_lut[target_bytes]
        return F.binary_cross_entropy_with_logits(logits, target_bits)

    @torch.no_grad()
    def generate(
        self,
        prompt_bytes: torch.Tensor,
        max_new_bytes: int = 256,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generation with batch-prefilled prompt (parallel) + recurrent decode.

        The prompt is processed in one parallel forward pass (not byte-by-byte),
        then switches to O(1) recurrent mode for autoregressive generation.
        """
        self.eval()
        device = prompt_bytes.device
        B = prompt_bytes.shape[0]
        T_prompt = prompt_bytes.shape[1]

        # Pre-allocate output (no O(n²) torch.cat)
        total_len = T_prompt + max_new_bytes
        generated = torch.empty(B, total_len, device=device, dtype=torch.long)
        generated[:, :T_prompt] = prompt_bytes

        # Batch prefill: process entire prompt in parallel (not byte-by-byte!)
        logits, states = self(prompt_bytes, states=None)

        # Autoregressive generation with recurrent O(1) memory
        write_pos = T_prompt
        for _ in range(max_new_bytes):
            bit_logits = logits[:, -1, :] / temperature
            sampled_bits = torch.bernoulli(torch.sigmoid(bit_logits)).long()
            next_byte = (sampled_bits * self.bit_powers).sum(-1, keepdim=True)
            generated[:, write_pos] = next_byte.squeeze(-1)
            write_pos += 1
            logits, states = self(next_byte, states=states)

        return generated[:, :write_pos]

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            total -= self.pos_emb.weight.numel()
        return total
