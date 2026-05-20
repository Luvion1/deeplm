"""
Transformer block with pluggable attention and MoE components.

Supports:
- Hybrid softmax/linear attention (MLA + Lightning)
- MoE with load-balanced routing (routed + shared experts)
- Hyper-Connections replacing standard residuals
- Configurable normalization (LayerNorm / RMSNorm)

Architecture:
  x → pre_norm → attention → hyper_connection/residual → post_norm → moe → residual → output
"""
import torch
import torch.nn as nn

from ..config import DeeplmConfig
from .hybrid_attention import HybridAttention
from .moe import MoELayer
from .hyper_connections import HyperConnectionsBlock


class TransformerBlock(nn.Module):
    """Decoder-only transformer block with pluggable attention and MoE.

    Args:
        config: Model configuration
        layer_idx: Layer index for hybrid attention routing
    """

    def __init__(self, config: DeeplmConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.architecture.hidden_size

        # Attention (softmax or softmax+linear hybrid)
        self.attention = HybridAttention(
            layer_idx=layer_idx,
            softmax_layers=config.hybrid_attention.softmax_layers,
            linear_config=config.hybrid_attention.linear_attention_config,
            hidden_size=config.architecture.hidden_size,
            head_dim=config.architecture.head_dim,
            q_lora_rank=config.mla.q_lora_rank,
            kv_lora_rank=config.mla.kv_lora_rank,
            qk_rope_head_dim=config.mla.qk_rope_head_dim,
            qk_nope_head_dim=config.mla.qk_nope_head_dim,
            v_head_dim=config.mla.v_head_dim,
            num_heads=config.mla.num_heads,
            kv_heads=config.mla.kv_heads,
        )

        # MoE (routed + shared experts)
        self.moe = MoELayer(config.moe, config.architecture.hidden_size)

        # Normalization layers
        self.pre_attention_norm = nn.LayerNorm(config.architecture.hidden_size)
        self.post_moe_norm = nn.LayerNorm(config.architecture.hidden_size)

        # Hyper-Connections (replaces standard residual)
        if config.hyper_connections.enabled:
            self.hyper_connection = HyperConnectionsBlock(
                config.hyper_connections, config.architecture.hidden_size
            )
        else:
            self.hyper_connection = None

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor = None,
                past_key_value=None, use_cache: bool = False):
        """
        Args:
            hidden_states: (batch, seq_len, hidden_size)
            attention_mask: Optional causal mask
            past_key_value: Optional KV cache
            use_cache: Whether to return KV cache

        Returns:
            hidden_states: Updated hidden states
            kv_cache: Optional KV cache tuple
        """
        residual = hidden_states

        hidden_states = self.pre_attention_norm(hidden_states)
        attn_output, kv_cache = self.attention(
            hidden_states, attention_mask, past_key_value, use_cache
        )

        # Hyper-connection or standard residual
        if self.hyper_connection is not None:
            hidden_states = self.hyper_connection(residual, attn_output)
        else:
            hidden_states = residual + attn_output

        # MoE/FFN with residual
        residual = hidden_states
        hidden_states = self.post_moe_norm(hidden_states)
        moe_output = self.moe(hidden_states)
        hidden_states = residual + moe_output

        return hidden_states, kv_cache
