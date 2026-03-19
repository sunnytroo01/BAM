"""
Binary Thinking Net (BTN) — fully optimized model.

Bug fixes applied in this version:
  - FIXED: Generation position embeddings (was always position 0 in recurrent mode)
  - FIXED: cumsum precision loss in bf16 (force float32 accumulation)
  - FIXED: Buffers now persistent=True for ZeRO-3 device safety
  - ADDED: Per-position normalization (stabilizes output magnitude across sequence)
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

    def __init__(self, config: BTNConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.chunk_size = config.chunk_size
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.chunk_size, config.chunk_size)),
            persistent=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            state: (association_matrix [B,H,d,d], assoc_count [B,H,1,1]) or None
        Returns:
            output, (new_matrix, new_count)
        """
        B, T, D = x.shape
        H, d = self.n_heads, self.d_head

        qkv = self.qkv_proj(x).view(B, T, 3, H, d)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2) * self.scale
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

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
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, H, T, d = q.shape
        C = self.chunk_size

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
        qc = q.reshape(B, H, N, C, d)
        kc = k.reshape(B, H, N, C, d)
        vc = v.reshape(B, H, N, C, d)

        # 1. Intra-chunk causal
        intra_scores = torch.matmul(qc, kc.transpose(-1, -2))
        intra_scores = intra_scores * self.causal_mask
        intra = torch.matmul(intra_scores, vc)

        # 2. Cross-chunk outer products
        chunk_kv = torch.matmul(kc.transpose(-1, -2), vc)  # [B,H,N,d,d]

        # FIX: Force float32 for cumsum (bf16 loses integer precision above 256)
        chunk_kv_f32 = chunk_kv.float()
        prefix = (torch.cumsum(chunk_kv_f32, dim=2) - chunk_kv_f32).to(chunk_kv.dtype)

        cross = torch.matmul(qc, prefix)

        # 3. Per-position normalization: divide by cumulative association count
        # Stabilizes output magnitude (prevents linear growth with sequence length)
        pos_in_chunk = torch.arange(1, C + 1, device=q.device, dtype=q.dtype)
        chunk_offsets = torch.arange(N, device=q.device, dtype=q.dtype) * C
        total_assocs = (chunk_offsets.unsqueeze(-1) + pos_in_chunk.unsqueeze(0))
        total_assocs = total_assocs.reshape(1, 1, N, C, 1).clamp(min=1.0)

        output = (intra + cross) / total_assocs
        output = output.reshape(B, H, T_pad, d)
        if pad:
            output = output[:, :, :T]

        # State: (accumulated matrix, total association count)
        final_matrix = (prefix[:, :, -1] + chunk_kv[:, :, -1])  # [B,H,d,d]
        final_count = torch.full(
            (B, H, 1, 1), T, device=q.device, dtype=q.dtype
        )
        return output, (final_matrix, final_count)

    def _recurrent(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Recurrent O(1) inference with per-position normalization."""
        B, H, T, d = q.shape
        matrix, count = state

        if T == 1:
            kt = k.squeeze(2)
            vt = v.squeeze(2)
            qt = q.squeeze(2)
            matrix = matrix + kt.unsqueeze(-1) * vt.unsqueeze(-2)
            count = count + 1
            out = (qt.unsqueeze(-2) @ matrix).squeeze(-2) / count.squeeze(-1)
            return out.unsqueeze(2), (matrix, count)

        outputs = []
        for t in range(T):
            kt = k[:, :, t]
            vt = v[:, :, t]
            qt = q[:, :, t]
            matrix = matrix + kt.unsqueeze(-1) * vt.unsqueeze(-2)
            count = count + 1
            out_t = (qt.unsqueeze(-2) @ matrix).squeeze(-2) / count.squeeze(-1)
            outputs.append(out_t)

        return torch.stack(outputs, dim=2), (matrix, count)


# ---------------------------------------------------------------------------
# SwiGLU FFN — fused gate+up projection
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):

    def __init__(self, config: BTNConfig):
        super().__init__()
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

    def forward(self, x, state=None):
        h, new_state = self.assoc(self.norm1(x), state=state)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x, new_state


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class BinaryThinkingNet(nn.Module):

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

        # Buffers (persistent=True for ZeRO-3 device safety)
        lut = ((torch.arange(256).unsqueeze(-1) >> torch.arange(7, -1, -1)) & 1).float()
        self.register_buffer("bits_lut", lut, persistent=True)
        self.register_buffer(
            "position_ids", torch.arange(config.context_length), persistent=True
        )
        self.register_buffer(
            "bit_powers",
            torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.long),
            persistent=True,
        )

        # Init
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
        states: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        use_checkpoint: bool = False,
        position_offset: int = 0,
    ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            position_offset: starting position index (for correct pos_emb in generation)
        """
        B, T = byte_ids.shape

        bits = self.bits_lut[byte_ids]
        x = self.bit_in(bits)
        # FIX: Use position_offset for correct position embeddings in generation
        x = x + self.pos_emb(self.position_ids[position_offset:position_offset + T])

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
        """Generation with batch-prefilled prompt + recurrent decode with correct positions."""
        self.eval()
        device = prompt_bytes.device
        B = prompt_bytes.shape[0]
        T_prompt = prompt_bytes.shape[1]

        total_len = T_prompt + max_new_bytes
        generated = torch.empty(B, total_len, device=device, dtype=torch.long)
        generated[:, :T_prompt] = prompt_bytes

        # Batch prefill (parallel, positions 0..T_prompt-1)
        logits, states = self(prompt_bytes, states=None, position_offset=0)

        # Autoregressive decode with correct position offsets
        pos = T_prompt
        for _ in range(max_new_bytes):
            bit_logits = logits[:, -1, :] / temperature
            sampled_bits = torch.bernoulli(torch.sigmoid(bit_logits)).long()
            next_byte = (sampled_bits * self.bit_powers).sum(-1, keepdim=True)
            generated[:, pos] = next_byte.squeeze(-1)
            # FIX: pass correct position (was always 0 before!)
            logits, states = self(next_byte, states=states, position_offset=pos)
            pos += 1

        return generated[:, :pos]

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        total = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            total -= self.pos_emb.weight.numel()
        return total
