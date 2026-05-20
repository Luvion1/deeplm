"""
Hyper-Connections with Sinkhorn routing (DeepSeek V4 style).

Replaces standard residual connections with learned routing over multiple
connection types (identity, transform, gate). Uses Sinkhorn-Knopp
normalization for doubly-stochastic routing weights.

Key reference: DeepSeek V4 Technical Report (2025)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import HyperConnectionsConfig


class SinkhornRouting(nn.Module):
    """Doubly-stochastic routing via iterative Sinkhorn-Knopp normalization.

    Normalizes a logit matrix so rows and columns each sum to 1,
    producing a soft assignment matrix. This creates competition both
    across connection types and across batch positions.
    """

    def __init__(self, iterations: int = 2, temperature: float = 0.1):
        super().__init__()
        self.iterations = iterations
        self.temperature = temperature

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits / self.temperature
        for _ in range(self.iterations):
            logits = logits - logits.logsumexp(dim=-2, keepdim=True)
            logits = logits - logits.logsumexp(dim=-1, keepdim=True)
        return logits.exp()


class HyperConnection(nn.Module):
    """Adaptive connection between residual path and layer output.

    Instead of a simple residual (x + f(x)), this learns to combine:
    - identity: standard residual connection
    - transform: linear projection of layer output
    - gate: gated combination
    - skip: no connection (zero path)
    
    The mixture weights are computed via Sinkhorn routing, making them
    both input-dependent and globally normalized.
    """

    def __init__(self, config: HyperConnectionsConfig, hidden_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size

        # Connection type configuration
        self.connection_types = config.connection_types
        self.num_types = len(self.connection_types)

        # Learned projections
        self.transform_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size * 2, self.num_types, bias=False)
        self.gate_signal_proj = nn.Linear(hidden_size * 2, self.num_types, bias=False)

        # Sinkhorn router for connection weights
        self.sinkhorn = SinkhornRouting(
            iterations=config.sinkhorn_iterations,
            temperature=config.sinkhorn_temperature,
        )

        # Learnable type biases (initialized from config)
        self.type_bias = nn.Parameter(torch.tensor([
            config.initial_weights.get(ct, 0.0) for ct in self.connection_types
        ]))

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.transform_proj.weight)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.xavier_uniform_(self.gate_signal_proj.weight)

    def forward(self, hidden_states: torch.Tensor, layer_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: Residual stream (batch, seq_len, hidden)
            layer_output: output of attention/FFN sublayer (batch, seq_len, hidden)

        Returns:
            Updated hidden states: sum(weight_i * connection_i) + original hidden
        """
        bsz, seq_len, hidden = hidden_states.size()

        # Compute routing weights via Sinkhorn
        combined = torch.cat([hidden_states, layer_output], dim=-1)
        routing_logits = self.gate_proj(combined.mean(dim=1))
        routing_probs = self.sinkhorn(routing_logits)

        # Transform option
        transformed = self.transform_proj(layer_output)

        # Compute weighted combination of all connection types
        connections = []
        for i, ct in enumerate(self.connection_types):
            w = routing_probs[:, i].unsqueeze(-1).unsqueeze(-1) * self.type_bias[i]

            if ct == "identity":
                connections.append(w * layer_output)
            elif ct == "transform":
                connections.append(w * transformed)
            elif ct == "gate":
                gate_signal = torch.sigmoid(
                    self.gate_signal_proj(torch.cat([hidden_states.mean(dim=1), layer_output.mean(dim=1)], dim=-1))
                )
                connections.append(w * gate_signal[:, i].unsqueeze(-1).unsqueeze(-1) * layer_output)
            else:
                connections.append(torch.zeros_like(layer_output))

        output = sum(connections)
        return output + hidden_states


class HyperConnectionsBlock(nn.Module):
    """Hyper-Connection block with pre-normalization.

    Wraps HyperConnection with LayerNorm applied to the layer output
    before routing.
    """

    def __init__(self, config: HyperConnectionsConfig, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size

        self.hyper_connection = HyperConnection(config, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor, layer_output: torch.Tensor) -> torch.Tensor:
        layer_output = self.layer_norm(layer_output)
        return self.hyper_connection(hidden_states, layer_output)
