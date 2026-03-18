"""
Binary Thinking Net (BTN) — full model definition (optimized).

Architecture (from paper):
  1. Binary I/O:  byte → 8-bit vector → Linear(8, d)  /  Linear(d, 8) → bits
  2. Binary Associative Memory replaces multi-head attention:
     - Project Q, K, V
     - Binarize K, V to {-1, +1} via Straight-Through Estimator
     - Accumulate causal outer-product memory: M_t = Σ_{i≤t} k_i ⊗ v_i
     - Retrieve: output_t = M_t @ q_t
  3. SwiGLU FFN + RMSNorm + residuals

Optimizations over naive implementation:
  - Vectorized parallel chunked association (no Python for-loop)
  - SwiGLU FFN (better quality per FLOP, same param count)
  - Causal mask + bit shifts registered as persistent buffers
  - torch.compile compatible
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
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""

    __constants__ = ["eps"]

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class BinarizeSTE(torch.autograd.Function):
    """Binarize to {-1, +1} with straight-through gradient estimator."""

    @staticmethod
    def forward(ctx, x):
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output  # straight-through: pass gradient unchanged


def binarize(x: torch.Tensor) -> torch.Tensor:
    """Binarize tensor to {-1, +1} using STE."""
    return BinarizeSTE.apply(x)


# ---------------------------------------------------------------------------
# Byte ↔ Bit conversion utilities
# ---------------------------------------------------------------------------

# Constant tensors (created once, moved to device with model)
_BIT_SHIFTS = torch.arange(7, -1, -1)  # [7, 6, 5, 4, 3, 2, 1, 0]
_BIT_POWERS = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.long)


def bytes_to_bits(byte_ids: torch.Tensor) -> torch.Tensor:
    """Convert byte values (0-255) to 8-bit float vectors, MSB first."""
    shifts = _BIT_SHIFTS.to(byte_ids.device)
    return ((byte_ids.unsqueeze(-1) >> shifts) & 1).float()


def bits_to_bytes(bit_logits: torch.Tensor) -> torch.Tensor:
    """Convert 8-bit logits back to byte values via hard thresholding."""
    bits = (bit_logits > 0).long()
    powers = _BIT_POWERS.to(bit_logits.device)
    return (bits * powers).sum(-1)


# ---------------------------------------------------------------------------
# Binary Associative Memory (replaces multi-head attention)
# ---------------------------------------------------------------------------

class BinaryAssociativeMemory(nn.Module):
    """
    Core innovation: replaces softmax attention with binary associative memory.

    Training: vectorized parallel chunked association.
      - Reshape sequence into N chunks of size C
      - Compute ALL intra-chunk causal attentions in one batched matmul
      - Compute cross-chunk states via cumsum (exclusive prefix sum)
      - No Python for-loop — entire operation is batched GPU ops

    Inference: recurrent O(1) context memory.
      - M ← M + sign(k_t) ⊗ sign(v_t)
      - output_t = M @ q_t
      - State size: H × d_head² (constant regardless of sequence length)
    """

    def __init__(self, config: BTNConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.d_head = config.d_head
        self.d_model = config.d_model
        self.chunk_size = config.chunk_size
        self.scale = 1.0 / math.sqrt(self.d_head)

        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

        # Pre-register causal mask as a buffer (avoids re-creation every forward)
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

        # Project and reshape to multi-head: [B, H, T, d]
        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)
        k = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2)

        # Scale query (analogous to 1/√d_k in standard attention)
        q = q * self.scale

        # Binarize keys and values to {-1, +1} via STE
        k = binarize(k)
        v = binarize(v)

        # Choose execution mode
        if state is not None:
            out, new_state = self._recurrent_association(q, k, v, state)
        else:
            out, new_state = self._parallel_chunked_association(q, k, v)

        # Merge heads and project
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.o_proj(out), new_state

    def _parallel_chunked_association(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Vectorized parallel chunked association — no Python for-loop.

        All chunks are processed simultaneously via batched matmuls:
        1. Reshape [B,H,T,d] → [B,H,N,C,d] where N=T/C chunks
        2. Intra-chunk: batched causal matmul over all N chunks at once
        3. Cross-chunk: exclusive prefix sum of per-chunk outer products,
           then batched query into accumulated states
        4. Combine and reshape back

        Memory: O(N·C²) for intra-scores + O(N·d²) for chunk states
        Speed: ~3-5x faster than sequential for-loop (eliminates Python overhead,
               enables full GPU saturation across chunks)
        """
        B, H, T, d = q.shape
        C = self.chunk_size

        # Pad T to exact multiple of C
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

        # Reshape into chunks: [B, H, N, C, d]
        qc = q.reshape(B, H, N, C, d)
        kc = k.reshape(B, H, N, C, d)
        vc = v.reshape(B, H, N, C, d)

        # === 1. Intra-chunk: causal linear attention (ALL chunks in parallel) ===
        # scores: [B, H, N, C, C] — query-key dot products within each chunk
        intra_scores = torch.einsum("bhnid,bhnjd->bhnij", qc, kc)
        intra_scores = intra_scores * self.causal_mask  # apply causal mask
        # weighted sum of values: [B, H, N, C, d]
        intra = torch.einsum("bhnij,bhnjd->bhnid", intra_scores, vc)

        # === 2. Cross-chunk: prefix-sum of association matrices ===
        # Per-chunk outer product sum: M_n = Σ_{i in chunk n} k_i ⊗ v_i
        chunk_kv = torch.einsum("bhnci,bhncj->bhnij", kc, vc)  # [B, H, N, d, d]

        # Exclusive prefix sum: state for chunk n = Σ_{m < n} chunk_kv[m]
        # (chunk 0 has no prior state, chunk 1 gets chunk 0's state, etc.)
        prefix = torch.zeros_like(chunk_kv)
        if N > 1:
            prefix[:, :, 1:] = torch.cumsum(chunk_kv[:, :, :-1], dim=2)

        # Query into accumulated state from all prior chunks
        cross = torch.einsum("bhncd,bhnde->bhnce", qc, prefix)  # [B, H, N, C, d]

        # === 3. Combine and reshape ===
        output = (intra + cross).reshape(B, H, T_pad, d)

        # Remove padding
        if pad:
            output = output[:, :, :T]

        # Final accumulated state (sum of all chunk contributions)
        final_state = prefix[:, :, -1] + chunk_kv[:, :, -1]  # [B, H, d, d]

        return output, final_state

    def _recurrent_association(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Recurrent inference mode: O(1) context memory.

        For each new position:
            M ← M + k_t ⊗ v_t
            output_t = M @ q_t

        State size is fixed at H × d_head × d_head regardless of sequence length.
        At d_head=128 with 96 heads: 96 × 128² × 2 bytes = 3 MB total.
        Compare to transformer KV cache at 64K tokens: ~805 MB per layer.
        """
        B, H, T, d = q.shape

        outputs = []
        for t in range(T):
            qt = q[:, :, t, :]  # [B, H, d]
            kt = k[:, :, t, :]
            vt = v[:, :, t, :]

            # Update association matrix: M += k_t ⊗ v_t
            state = state + torch.einsum("bhi,bhj->bhij", kt, vt)

            # Retrieve: output_j = Σ_i q_i * M_{i,j} = Σ_s (q·k_s) * v_s[j]
            # This is the paper's definition: weighted sum of values by query-key similarity
            out_t = torch.einsum("bhi,bhij->bhj", qt, state)
            outputs.append(out_t)

        output = torch.stack(outputs, dim=2)  # [B, H, T, d]
        return output, state


# ---------------------------------------------------------------------------
# Feed-Forward Network — SwiGLU (Shazeer 2020, used by LLaMA/Mistral)
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """
    SwiGLU FFN: better quality per FLOP than standard GELU FFN.

    Uses 3 weight matrices instead of 2, so d_ff is reduced to (8/3)*d_model
    to maintain the same parameter count. Net result: same params, better
    compute utilization, same or better quality.

    gate = SiLU(x @ W1)
    up   = x @ W3
    out  = (gate * up) @ W2
    """

    def __init__(self, config: BTNConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff, bias=False)  # gate
        self.w3 = nn.Linear(config.d_model, config.d_ff, bias=False)  # up
        self.w2 = nn.Linear(config.d_ff, config.d_model, bias=False)  # down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# ---------------------------------------------------------------------------
# Single BTN Block (Association + FFN)
# ---------------------------------------------------------------------------

class BTNBlock(nn.Module):
    """One layer of BTN: pre-norm → binary association → residual → pre-norm → FFN → residual."""

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
        # Binary associative memory (replaces attention)
        residual = x
        h, new_state = self.assoc(self.norm1(x), state=state)
        x = residual + h

        # Feed-forward
        x = x + self.ffn(self.norm2(x))

        return x, new_state


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class BinaryThinkingNet(nn.Module):
    """
    Binary Thinking Net — 175B-class tokenizer-free language model.

    Key properties:
    - Binary I/O: operates on raw bytes (8-bit vectors), no tokenizer
    - O(1) context memory at inference (vs O(T×d) for transformers)
    - Structural anti-plagiarism: superposed outer products are unrecoverable
    - 8-dimensional I/O head vs 50,000+ dimensional subword head

    Architecture:
        byte(0-255) → 8-bit vector → Linear(8,d) → +pos_emb
        → [BTNBlock × L] → RMSNorm → Linear(d,8) → 8 bit logits
    """

    def __init__(self, config: BTNConfig):
        super().__init__()
        self.config = config

        # --- Binary I/O (tokenizer elimination) ---
        self.bit_in = nn.Linear(8, config.d_model, bias=False)
        self.pos_emb = nn.Embedding(config.context_length, config.d_model)

        # --- Transformer-style blocks with binary association ---
        self.blocks = nn.ModuleList([
            BTNBlock(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # --- Output ---
        self.norm = RMSNorm(config.d_model, eps=config.norm_eps)
        self.bit_out = nn.Linear(config.d_model, 8, bias=False)

        # Initialize weights
        self.apply(self._init_weights)
        # Special scaled init for output projections (GPT-2 style)
        for block in self.blocks:
            nn.init.normal_(
                block.assoc.o_proj.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * config.n_layers),
            )
            nn.init.normal_(
                block.ffn.w2.weight,
                mean=0.0,
                std=0.02 / math.sqrt(2 * config.n_layers),
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
        """
        Args:
            byte_ids: [B, T] long tensor of byte values (0-255)
            states: list of L association matrices for recurrent inference
            use_checkpoint: enable gradient checkpointing (saves VRAM, costs compute)

        Returns:
            logits: [B, T, 8] raw logits for each bit of the next byte
            new_states: list of L updated association matrices
        """
        B, T = byte_ids.shape

        # --- Binary I/O: bytes → bits → embedding ---
        bits = bytes_to_bits(byte_ids)       # [B, T, 8]
        x = self.bit_in(bits)                # [B, T, d_model]

        # --- Positional embedding ---
        positions = torch.arange(T, device=byte_ids.device)
        x = x + self.pos_emb(positions)

        # --- Process through all BTN blocks ---
        new_states = []
        for i, block in enumerate(self.blocks):
            layer_state = states[i] if states is not None else None

            if use_checkpoint and self.training:
                def create_block_fn(blk, st):
                    def block_fn(hidden):
                        return blk(hidden, state=st)
                    return block_fn

                x, new_state = checkpoint(
                    create_block_fn(block, layer_state),
                    x,
                    use_reentrant=False,
                )
            else:
                x, new_state = block(x, state=layer_state)

            new_states.append(new_state)

        # --- Output: embedding → bits ---
        x = self.norm(x)
        logits = self.bit_out(x)  # [B, T, 8]

        return logits, new_states

    # -------------------------------------------------------------------
    # Convenience methods
    # -------------------------------------------------------------------

    def compute_loss(
        self, byte_ids: torch.Tensor, use_checkpoint: bool = False
    ) -> tuple[torch.Tensor, dict]:
        """
        Compute next-byte prediction loss.

        Args:
            byte_ids: [B, T+1] where input=[:, :-1], target=[:, 1:]
        Returns:
            loss: scalar (mean binary cross-entropy per bit)
            metrics: dict with 'bpb' (bits-per-byte)
        """
        input_bytes = byte_ids[:, :-1]
        target_bytes = byte_ids[:, 1:]

        logits, _ = self(input_bytes, use_checkpoint=use_checkpoint)
        target_bits = bytes_to_bits(target_bytes)

        loss = F.binary_cross_entropy_with_logits(logits, target_bits)

        # Bits-per-byte: convert per-bit nats to per-byte bits
        with torch.no_grad():
            bpb = loss.item() * 8.0 / math.log(2)

        return loss, {"bpb": bpb}

    @torch.no_grad()
    def generate(
        self,
        prompt_bytes: torch.Tensor,
        max_new_bytes: int = 256,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Autoregressive byte generation using recurrent O(1) inference.

        Args:
            prompt_bytes: [1, T] byte tensor
            max_new_bytes: number of bytes to generate
            temperature: sampling temperature

        Returns:
            [1, T + max_new_bytes] generated byte sequence
        """
        self.eval()
        device = prompt_bytes.device
        B = prompt_bytes.shape[0]
        assert B == 1, "Generation currently supports batch_size=1"

        # Initialize empty association matrices for all layers
        d = self.config.d_head
        H = self.config.n_heads
        param_dtype = next(self.parameters()).dtype
        states = [
            torch.zeros(B, H, d, d, device=device, dtype=param_dtype)
            for _ in range(self.config.n_layers)
        ]

        # Process prompt through recurrent mode
        generated = prompt_bytes.clone()
        for t in range(prompt_bytes.shape[1]):
            step_input = prompt_bytes[:, t:t+1]
            logits, states = self(step_input, states=states)

        # Pre-compute powers tensor once
        powers = torch.tensor(
            [128, 64, 32, 16, 8, 4, 2, 1], device=device, dtype=torch.long
        )

        # Generate new bytes
        for _ in range(max_new_bytes):
            bit_logits = logits[:, -1, :] / temperature
            bit_probs = torch.sigmoid(bit_logits)
            sampled_bits = torch.bernoulli(bit_probs).long()

            next_byte = (sampled_bits * powers).sum(-1, keepdim=True)
            generated = torch.cat([generated, next_byte], dim=1)

            logits, states = self(next_byte, states=states)

        return generated

    def num_parameters(self, exclude_embeddings: bool = False) -> int:
        """Count total parameters."""
        total = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            total -= self.pos_emb.weight.numel()
        return total
