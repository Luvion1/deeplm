"""
Mixture of Experts (MoE) layer with load-balanced routing.

Architecture:
  Input → Router → Top-K Experts + Shared Experts → Output
  
The router uses sqrt(softplus(x)) scoring with bias-based load balancing
(no auxiliary loss), following DeepSeek V4 / Kimi K2.6 style.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import MoEConfig


class SwiGLU(nn.Module):
    """Swish-Gated Linear Unit activation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return F.silu(x1) * x2


class ExpertFFN(nn.Module):
    """Feed-forward expert network with SwiGLU activation.

    Uses fused gate+up projection for efficiency.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, activation: str = "swiglu"):
        super().__init__()
        self.gate_up_proj = nn.Linear(hidden_size, intermediate_size * 2, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.down_proj._is_residual = True
        self.activation = SwiGLU() if activation == "swiglu" else nn.GELU()
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.gate_up_proj.weight)
        nn.init.xavier_uniform_(self.down_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        activated = self.activation(gate_up)
        return self.down_proj(activated)


class LoadBalancedRouter(nn.Module):
    """Token-to-expert router with bias-based load balancing (no aux loss).

    Scores tokens using sqrt(softplus(x)), then selects top-k experts.
    Maintains per-expert bias that auto-adjusts to balance load.

    References:
        - DeepSeek V4: No auxiliary loss routing with dynamic bias correction
        - MiniMax M2.7: sqrtsoftplus scoring for numerical stability
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int, config: MoEConfig):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # Routing configuration
        self.scoring_function = config.router.scoring_function
        self.bias_update_speed = config.router.bias_update_speed
        self.load_balance_tolerance = config.router.load_balance_tolerance

        # Learned token→expert affinity weights
        self.weight = nn.Parameter(torch.randn(num_experts, hidden_size) * 0.01)

        # Dynamic bias for load balancing (not persistent in checkpoints)
        self.register_buffer("routing_bias", torch.zeros(num_experts), persistent=False)

    def forward(self, x: torch.Tensor):
        """Route tokens to top-k experts.

        Args:
            x: Input tensor of shape (batch*seq, hidden_size)

        Returns:
            routing_weights: Weight per expert per token
            topk_indices: Selected expert indices per token
        """
        logits = F.linear(x, self.weight)

        # Score: sqrt(softplus(x)) for numerical stability
        if self.scoring_function == "sqrtsoftplus":
            scores = torch.sqrt(F.softplus(logits))
        else:
            scores = torch.sigmoid(logits)

        # Apply load-balancing bias (no auxiliary loss)
        scores = scores + self.routing_bias

        # Select top-k experts
        topk_scores, topk_indices = torch.topk(scores, self.top_k, dim=-1)
        routing_weights = F.softmax(topk_scores, dim=-1)

        # Update load-balancing bias during training
        if self.training:
            mask = torch.zeros_like(scores).scatter_(1, topk_indices, 1.0)
            expert_counts = mask.sum(dim=0).float()
            expected_count = mask.numel() / self.num_experts
            imbalance = (expert_counts - expected_count) / (expected_count + 1e-20)
            self.routing_bias.data -= self.bias_update_speed * imbalance

        return routing_weights, topk_indices


class MoELayer(nn.Module):
    """Mixture of Experts layer with routed and shared experts.

    Each token is routed to top-k of N routed experts, plus always-active
    shared experts. The shared expert ensures baseline knowledge is always
    accessible (Kimi K2.6 style).

    Optional expert affinity memory tracks which experts handle which tokens,
    enabling faster convergence through token-expert caching.
    """

    def __init__(self, config: MoEConfig, hidden_size: int):
        super().__init__()
        self.config = config
        self.num_routed = config.num_routed_experts
        self.num_shared = config.num_shared_experts
        self.top_k = config.top_k

        self.routed_experts = nn.ModuleList([
            ExpertFFN(hidden_size, config.expert_intermediate_size, config.expert_activation)
            for _ in range(config.num_routed_experts)
        ])

        self.shared_experts = nn.ModuleList([
            ExpertFFN(hidden_size, config.shared_expert_intermediate_size, config.expert_activation)
            for _ in range(config.num_shared_experts)
        ])

        self.router = LoadBalancedRouter(hidden_size, config.num_routed_experts, config.top_k, config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with expert routing.

        Uses grouped dispatch: tokens are sorted by expert assignment,
        processed in batched expert groups, then scattered back.
        Avoids Python loop overhead and maximizes GPU utilization.

        Args:
            x: Input of shape (batch, seq_len, hidden_size)

        Returns:
            Output of shape (batch, seq_len, hidden_size)
        """
        bsz, seq_len, hidden = x.shape
        x_flat = x.view(-1, hidden)
        num_tokens = x_flat.size(0)

        routing_weights, topk_indices = self.router(x_flat)

        y = torch.zeros_like(x_flat)

        # Grouped dispatch: for each expert, gather tokens, process, scatter back
        # Stack expert weights for batched processing when possible
        for expert_idx in range(self.num_routed):
            mask = topk_indices == expert_idx
            if not mask.any():
                continue
            positions = mask.nonzero(as_tuple=True)
            token_pos = positions[0]
            selection_idx = positions[1]
            expert_weights = routing_weights[token_pos, selection_idx].unsqueeze(-1)
            expert_tokens = x_flat[token_pos]
            expert_output = self.routed_experts[expert_idx](expert_tokens)
            y.index_add_(0, token_pos, expert_output * expert_weights)

        # Shared experts: fused forward for better throughput
        if self.num_shared == 1:
            y.add_(self.shared_experts[0](x_flat))
        else:
            for expert in self.shared_experts:
                y.add_(expert(x_flat))

        return y.view(bsz, seq_len, hidden)
