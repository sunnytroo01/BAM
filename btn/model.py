"""
Binary Thinking Net (BTN) — fully optimized model.

This version adds:
  - DyT normalization (Dynamic Tanh, CVPR 2025) — replaces RMSNorm, ~8% faster
  - Clipped STE — zero gradient for saturated binarization (better training dynamics)
  - Multi-byte prediction — auxiliary heads predict t+2..t+N (better sample efficiency)
  - Optional Flash Linear Attention — fused Triton kernel when fla is installed
  - chunk_size=128 (= d_head) — balanced intra/cross FLOPs, better tensor cores
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from btn.config import BTNConfig

# Optional: Flash Linear Attention fused kernel
try:
    from fla.ops.linear_attn import chunk_linear_attn as _fla_chunk_linear_attn
    HAS_FLA = True
except ImportError:
    HAS_FLA = False

# Optional: Liger Kernel fused SiLU*Mul (saves one full activation read+write)
try:
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    HAS_LIGER = True
except ImportError:
    HAS_LIGER = False


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class DyT(nn.Module):
    """
    Dynamic Tanh normalization (CVPR 2025).
    Replaces RMSNorm: no mean/variance computation, single tanh + scale.
    ~8% faster training, matches or exceeds RMSNorm quality.
    """

    def __init__(self, dim: int, alpha_init: float = 0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha_init))
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.alpha * x) * self.weight


class BinarizeSTE(torch.autograd.Function):
    """Clipped STE: zero gradient for values far from decision boundary."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x >= 0, 1.0, -1.0)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        # Clip: only pass gradient where |x| <= 1 (near decision boundary)
        # Values far from 0 are already firmly decided — gradient is noise
        return grad_output * (x.abs() <= 1.0).to(grad_output.dtype)


def binarize(x: torch.Tensor) -> torch.Tensor:
    return BinarizeSTE.apply(x)


# ---------------------------------------------------------------------------
# Binary Associative Memory
# ---------------------------------------------------------------------------

class BinaryAssociativeMemory(nn.Module):

    def __init__(self, config: BTNConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.chunk_size = config.chunk_size
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.use_fla = HAS_FLA

        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(config.chunk_size, config.chunk_size)),
            persistent=True,
        )

    def forward(self, x, state=None):
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
        elif self.use_fla and not self.training:
            # FLA fused kernel for inference (skip during training to keep
            # consistent normalization with our per-position /t scheme)
            out, new_state = self._fla_forward(q, k, v)
        else:
            out, new_state = self._parallel_chunked(q, k, v)

        out = out.transpose(1, 2).reshape(B, T, D)
        return self.o_proj(out), new_state

    def _fla_forward(self, q, k, v):
        """Flash Linear Attention fused kernel path."""
        B, H, T, d = q.shape
        # FLA expects [B, H, T, d] — same as ours
        out = _fla_chunk_linear_attn(q, k, v, normalize=True)
        # Approximate state (FLA doesn't return it, use zeros for non-recurrent)
        dummy_state = (
            torch.zeros(B, H, d, d, device=q.device, dtype=q.dtype),
            torch.full((B, H, 1, 1), T, device=q.device, dtype=q.dtype),
        )
        return out, dummy_state

    def _parallel_chunked(self, q, k, v):
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

        intra_scores = torch.matmul(qc, kc.transpose(-1, -2))
        intra_scores = intra_scores * self.causal_mask
        intra = torch.matmul(intra_scores, vc)

        chunk_kv = torch.matmul(kc.transpose(-1, -2), vc)
        chunk_kv_f32 = chunk_kv.float()
        prefix = (torch.cumsum(chunk_kv_f32, dim=2) - chunk_kv_f32).to(chunk_kv.dtype)
        cross = torch.matmul(qc, prefix)

        # Per-position normalization
        pos_in_chunk = torch.arange(1, C + 1, device=q.device, dtype=q.dtype)
        chunk_offsets = torch.arange(N, device=q.device, dtype=q.dtype) * C
        total_assocs = (chunk_offsets.unsqueeze(-1) + pos_in_chunk.unsqueeze(0))
        total_assocs = total_assocs.reshape(1, 1, N, C, 1).clamp(min=1.0)

        output = (intra + cross) / total_assocs
        output = output.reshape(B, H, T_pad, d)
        if pad:
            output = output[:, :, :T]

        final_matrix = prefix[:, :, -1] + chunk_kv[:, :, -1]
        final_count = torch.full((B, H, 1, 1), T, device=q.device, dtype=q.dtype)
        return output, (final_matrix, final_count)

    def _recurrent(self, q, k, v, state):
        B, H, T, d = q.shape
        matrix, count = state

        if T == 1:
            kt, vt, qt = k.squeeze(2), v.squeeze(2), q.squeeze(2)
            matrix = matrix + kt.unsqueeze(-1) * vt.unsqueeze(-2)
            count = count + 1
            out = (qt.unsqueeze(-2) @ matrix).squeeze(-2) / count.squeeze(-1)
            return out.unsqueeze(2), (matrix, count)

        outputs = []
        for t in range(T):
            kt, vt, qt = k[:, :, t], v[:, :, t], q[:, :, t]
            matrix = matrix + kt.unsqueeze(-1) * vt.unsqueeze(-2)
            count = count + 1
            out_t = (qt.unsqueeze(-2) @ matrix).squeeze(-2) / count.squeeze(-1)
            outputs.append(out_t)
        return torch.stack(outputs, dim=2), (matrix, count)


# ---------------------------------------------------------------------------
# SwiGLU FFN
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    def __init__(self, config: BTNConfig):
        super().__init__()
        self.gate_up = nn.Linear(config.d_model, 2 * config.d_ff, bias=False)
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)

    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        if HAS_LIGER:
            # Fused SiLU * Mul: single kernel, ~1.5x less memory on FFN
            return self.w2(LigerSiLUMulFunction.apply(gate, up))
        return self.w2(F.silu(gate) * up)


# ---------------------------------------------------------------------------
# BTN Block — uses DyT instead of RMSNorm
# ---------------------------------------------------------------------------

class BTNBlock(nn.Module):
    def __init__(self, config: BTNConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.norm1 = DyT(config.d_model)
        self.assoc = BinaryAssociativeMemory(config)
        self.norm2 = DyT(config.d_model)
        self.ffn = SwiGLUFFN(config)

    def forward(self, x, state=None):
        h, new_state = self.assoc(self.norm1(x), state=state)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x, new_state


# ---------------------------------------------------------------------------
# Full Model — with multi-byte prediction
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

        # Output: main head + auxiliary multi-byte prediction heads
        self.norm = DyT(config.d_model)
        self.bit_out = nn.Linear(config.d_model, 8, bias=False)

        # Multi-byte prediction: predict bytes t+2, t+3, ..., t+N+1
        # Improves sample efficiency by ~15-25% (Meta ICLR 2025)
        self.aux_heads = nn.ModuleList([
            nn.Linear(config.d_model, 8, bias=False)
            for _ in range(config.n_aux_predict)
        ])

        # Buffers
        lut = ((torch.arange(256).unsqueeze(-1) >> torch.arange(7, -1, -1)) & 1).float()
        self.register_buffer("bits_lut", lut, persistent=True)
        self.register_buffer("position_ids", torch.arange(config.context_length), persistent=True)
        self.register_buffer(
            "bit_powers", torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.long),
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

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, byte_ids, states=None, use_checkpoint=False, position_offset=0):
        B, T = byte_ids.shape

        bits = self.bits_lut[byte_ids]
        x = self.bit_in(bits)
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
        return logits, new_states, x  # also return hidden for aux heads

    def compute_loss(self, byte_ids, use_checkpoint=False):
        """
        Next-byte prediction + multi-byte auxiliary losses.
        Uses byte_ids[:, :-1] as input, byte_ids[:, 1:] for main target,
        byte_ids[:, 2:], [:, 3:], ... for auxiliary targets.
        """
        n_aux = len(self.aux_heads)
        max_offset = n_aux + 1
        input_bytes = byte_ids[:, :-max_offset]
        T = input_bytes.shape[1]

        logits, _, hidden = self(input_bytes, use_checkpoint=use_checkpoint)

        # Main loss: predict byte t+1
        target_main = self.bits_lut[byte_ids[:, 1:T + 1]]
        loss = F.binary_cross_entropy_with_logits(logits, target_main)

        # Auxiliary losses: predict bytes t+2, t+3, ..., t+N+1
        if n_aux > 0 and self.training:
            aux_weight = self.config.aux_loss_weight
            for i, head in enumerate(self.aux_heads):
                offset = i + 2
                target_aux = self.bits_lut[byte_ids[:, offset:offset + T]]
                aux_logits = head(hidden)
                loss = loss + aux_weight * F.binary_cross_entropy_with_logits(
                    aux_logits, target_aux
                )

        return loss

    @torch.no_grad()
    def generate(self, prompt_bytes, max_new_bytes=256, temperature=1.0):
        self.eval()
        device = prompt_bytes.device
        B = prompt_bytes.shape[0]
        T_prompt = prompt_bytes.shape[1]

        total_len = T_prompt + max_new_bytes
        generated = torch.empty(B, total_len, device=device, dtype=torch.long)
        generated[:, :T_prompt] = prompt_bytes

        logits, states, _ = self(prompt_bytes, states=None, position_offset=0)

        pos = T_prompt
        for _ in range(max_new_bytes):
            bit_logits = logits[:, -1, :] / temperature
            sampled_bits = torch.bernoulli(torch.sigmoid(bit_logits)).long()
            next_byte = (sampled_bits * self.bit_powers).sum(-1, keepdim=True)
            generated[:, pos] = next_byte.squeeze(-1)
            logits, states, _ = self(next_byte, states=states, position_offset=pos)
            pos += 1

        return generated[:, :pos]

    def num_parameters(self, exclude_embeddings=False):
        total = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            total -= self.pos_emb.weight.numel()
        return total
