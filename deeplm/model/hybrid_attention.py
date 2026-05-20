"""
Hybrid Attention: Softmax (MLA) + Linear (Lightning Attention v2).

References:
  - MiniMax M2.7: Hybrid Lightning + Softmax Attention
  - DeepSeek V4: Multi-head Latent Attention (MLA) for all layers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import LinearAttentionConfig


class LightningAttentionV2(nn.Module):
    """Linear attention with block-based computation (MiniMax Lightning Attention v2).

    Uses intra-block softmax attention (for local precision) and inter-block
    linear attention (KV product) for O(n) overall complexity with O(nd²) per step.

    Supports incremental KV state for efficient autoregressive generation.

    Reference: MiniMax-01 Technical Report (2025)
    """

    def __init__(self, config: LinearAttentionConfig, head_dim: int):
        super().__init__()
        self.config = config
        self.head_dim = head_dim
        self.block_size = config.block_size
        self.activation_name = config.activation
        self.register_buffer("_kv_state", None, persistent=False)
        self._state_seq_len = 0

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply activation function that replaces softmax."""
        if self.activation_name == "swish":
            return x * torch.sigmoid(x)
        return F.relu(x)

    def _intra_block_softmax(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Within-block softmax attention (causal) using SDPA for speed."""
        return F.scaled_dot_product_attention(
            Q, K, V,
            attn_mask=None,
            is_causal=True,
            scale=self.head_dim ** -0.5,
        )

    def _inter_block_kv_product(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Cross-block linear attention via cumulative KV product.

        Uses a single incremental pass instead of recomputing from scratch
        per block. O(n) instead of O(n²).
        """
        bsz, num_heads, seq_len, head_dim = Q.shape
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        outputs = []

        kv_cumulative = torch.zeros(bsz, num_heads, head_dim, head_dim, device=Q.device, dtype=Q.dtype)

        for b in range(num_blocks):
            start = b * self.block_size
            end = min(start + self.block_size, seq_len)

            K_block = K[:, :, start:end]
            V_block = V[:, :, start:end]

            kv_cumulative = kv_cumulative + torch.matmul(K_block.transpose(-2, -1), V_block)

            Q_block = Q[:, :, start:end]
            out_block = torch.matmul(Q_block, kv_cumulative)
            outputs.append(out_block)

        return torch.cat(outputs, dim=2)

    def _step(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Single-token inference step using incremental KV state.

        Args:
            q: (batch, num_heads, 1, head_dim)
            k: (batch, num_heads, 1, head_dim)
            v: (batch, num_heads, 1, head_dim)

        Returns:
            output: (batch, num_heads, 1, head_dim)
        """
        if self._kv_state is None:
            bsz, num_heads, _, head_dim = q.shape
            self.register_buffer("_kv_state", torch.zeros(bsz, num_heads, head_dim, head_dim,
                                                          device=q.device, dtype=q.dtype), persistent=False)
            self._state_seq_len = 0

        # Update KV state: KV += k^T @ v
        self._kv_state = self._kv_state + torch.matmul(k.transpose(-2, -1), v)
        self._state_seq_len += 1

        # Output: q @ KV
        return torch.matmul(q, self._kv_state)

    def reset_kv_state(self):
        """Reset incremental KV state for new sequence."""
        self._kv_state = None
        self._state_seq_len = 0

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                past_kv_state=None, use_cache: bool = False) -> torch.Tensor:
        """Forward pass with intra-block softmax + inter-block linear attention.

        Args:
            Q: (batch, num_heads, seq_len, head_dim)
            K: (batch, num_heads, seq_len, head_dim)
            V: (batch, num_heads, seq_len, head_dim)
            past_kv_state: Optional previous KV state for incremental generation
            use_cache: Whether to return updated KV state

        Returns:
            output: (batch, num_heads, seq_len, head_dim)
            kv_state: Optional KV state tuple (kv_state, seq_len)
        """
        bsz, num_heads, seq_len, head_dim = Q.shape

        Q = self._activation(Q)
        K = self._activation(K)

        # Single-token inference: use incremental KV state
        if seq_len == 1:
            if past_kv_state is not None:
                self._kv_state = past_kv_state[0]
                self._state_seq_len = past_kv_state[1]
            output = self._step(Q, K, V)
            kv_state = (self._kv_state, self._state_seq_len) if use_cache else None
            return output, kv_state

        # Reset state for full sequence processing
        self.reset_kv_state()

        if seq_len <= self.block_size:
            return self._intra_block_softmax(Q, K, V), None

        intra = self._intra_block_softmax(Q, K, V)
        inter = self._inter_block_kv_product(Q, K, V)
        return intra + inter, None


class HybridAttention(nn.Module):
    """Attention that combines softmax (MLA) and linear (Lightning) attention.

    Architecture:
      - Softmax layers (e.g., layers 0, 4, 7): Use MLA only
      - Linear layers (e.g., layers 1,2,3,5,6): Use MLA + 50/50 blend with Lightning

    The blend ratio is configurable, with 0.5 being the MiniMax default.
    """

    def __init__(self, layer_idx: int, softmax_layers: list, linear_config: LinearAttentionConfig,
                 hidden_size: int, num_heads: int, head_dim: int,
                 blend_ratio: float = 0.5, **mla_kwargs):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_softmax = layer_idx in softmax_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.blend_ratio = blend_ratio  # Weight for linear attention output

        from .mla import MultiHeadLatentAttention
        from ..config import MLAConfig

        mla_kwargs['num_heads'] = num_heads
        mla_config = MLAConfig(**mla_kwargs)
        self.mla = MultiHeadLatentAttention(mla_config, hidden_size)

        if not self.use_softmax:
            # Additional projections for linear attention path
            self.linear_attn = LightningAttentionV2(linear_config, head_dim)
            self.linear_q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
            self.linear_k_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
            self.linear_v_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
            self.linear_output_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
            self.linear_output_proj._is_residual = True

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None,
                past_key_value=None, use_cache: bool = False):
        """Forward pass: MLA always, plus optional Lightning blend."""
        bsz, q_len, _ = hidden_states.size()

        # Extract MLA KV cache from combined past_key_value
        mla_past_kv = past_key_value[:2] if past_key_value is not None else None

        attn_output, mla_kv_cache = self.mla(
            hidden_states, attention_mask, mla_past_kv, use_cache
        )

        if not self.use_softmax:
            # Compute linear attention
            q = self.linear_q_proj(hidden_states)
            k = self.linear_k_proj(hidden_states)
            v = self.linear_v_proj(hidden_states)

            q = q.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            v = v.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)

            # Extract linear attention KV state from past_key_value if available
            past_linear_kv = past_key_value[2] if past_key_value is not None and len(past_key_value) > 2 else None
            linear_out, linear_kv_state = self.linear_attn(q, k, v, past_linear_kv, use_cache)
            linear_out = linear_out.transpose(1, 2).contiguous()
            linear_out = linear_out.view(bsz, q_len, self.num_heads * self.head_dim)
            linear_out = self.linear_output_proj(linear_out)

            # Blend with MLA output
            attn_output = (1 - self.blend_ratio) * attn_output + self.blend_ratio * linear_out

            # Return combined KV cache: (mla_kv_latent, mla_k_pe, linear_kv_state)
            if use_cache:
                kv_cache = (mla_kv_cache[0], mla_kv_cache[1], linear_kv_state) if mla_kv_cache is not None else (None, None, linear_kv_state)
            else:
                kv_cache = None
        else:
            kv_cache = mla_kv_cache

        return attn_output, kv_cache
