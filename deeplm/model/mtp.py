import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import MTPConfig
from .mla import RotaryEmbedding, apply_rotary_pos_emb


class MTPProjection(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) + self.norm(x)


class MTPLayer(nn.Module):
    def __init__(self, config: MTPConfig, hidden_size: int, vocab_size: int, rope_theta: float = 10000.0):
        super().__init__()
        self.config = config
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.mtp_depth = config.mtp_depth
        self.rope_dim = hidden_size // 4

        self.projections = nn.ModuleList([
            MTPProjection(hidden_size) for _ in range(config.mtp_depth)
        ])

        self.rotary_emb = RotaryEmbedding(self.rope_dim, theta=rope_theta)

    def forward(self, hidden_states: torch.Tensor, lm_head: nn.Linear,
                tied_embedding: nn.Embedding = None) -> list:
        predictions = []

        for depth in range(self.mtp_depth):
            proj_input = hidden_states
            proj_output = self.projections[depth](proj_input)

            seq_len = proj_output.size(1)
            cos, sin = self.rotary_emb(proj_output[..., :self.rope_dim], seq_len=seq_len)
            proj_output[..., :self.rope_dim] = apply_rotary_pos_emb(
                proj_output[..., :self.rope_dim], cos, sin
            )

            if self.config.mtp_head == "tied" and tied_embedding is not None:
                logits = F.linear(proj_output, tied_embedding.weight)
            else:
                logits = lm_head(proj_output)

            predictions.append(logits)

        return predictions


class MTPHead(nn.Module):
    def __init__(self, config: MTPConfig, hidden_size: int, vocab_size: int, rope_theta: float = 10000.0):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            MTPLayer(config, hidden_size, vocab_size, rope_theta)
            for _ in range(config.num_mtp_layers)
        ])
        self.loss_weight = config.mtp_loss_weight

    def forward(self, hidden_states: torch.Tensor, lm_head: nn.Linear,
                tied_embedding: nn.Embedding = None) -> list:
        all_predictions = []
        for layer in self.layers:
            preds = layer(hidden_states, lm_head, tied_embedding)
            all_predictions.extend(preds)
        return all_predictions

    def compute_loss(self, predictions: list, target_ids: torch.Tensor, shift: int = 1) -> torch.Tensor:
        total_loss = 0.0
        count = 0

        for i, logits in enumerate(predictions):
            shifted_target = target_ids[:, shift + i:]
            if shifted_target.size(1) <= 0:
                continue
            logits_truncated = logits[:, :shifted_target.size(1)]

            loss = F.cross_entropy(
                logits_truncated.reshape(-1, logits_truncated.size(-1)),
                shifted_target.reshape(-1),
                ignore_index=-100,
            )
            total_loss = total_loss + loss
            count += 1

        return (total_loss / count) * self.loss_weight if count > 0 else torch.tensor(0.0, device=target_ids.device)

    def compute_loss_chunked(self, hidden_states: torch.Tensor, lm_head: nn.Linear,
                              target_ids: torch.Tensor, shift: int = 1,
                              tied_embedding: nn.Embedding = None,
                              chunk_size: int = 40) -> torch.Tensor:
        """Chunked MTP loss — never materializes full (B, S, V) logits per depth."""
        bsz, nseql, _ = hidden_states.shape
        total_loss = 0.0
        total_count = 0

        for i in range(0, nseql, chunk_size):
            end = min(i + chunk_size, nseql)
            chunk_hidden = hidden_states[:, i:end, :]
            cos_chunk, sin_chunk = None, None

            for layer in self.layers:
                for depth in range(layer.mtp_depth):
                    proj_out = layer.projections[depth](chunk_hidden)

                    seq_len = proj_out.size(1)
                    if cos_chunk is None:
                        cos_chunk, sin_chunk = layer.rotary_emb(
                            proj_out[..., :layer.rope_dim], seq_len=seq_len
                        )
                    proj_out[..., :layer.rope_dim] = apply_rotary_pos_emb(
                        proj_out[..., :layer.rope_dim], cos_chunk, sin_chunk
                    )

                    if layer.config.mtp_head == "tied" and tied_embedding is not None:
                        chunk_logits = F.linear(proj_out, tied_embedding.weight)
                    else:
                        chunk_logits = lm_head(proj_out)

                    shifted_target = target_ids[:, shift + depth + i:shift + depth + end]
                    if shifted_target.size(1) <= 0:
                        continue
                    logits_trunc = chunk_logits[:, :shifted_target.size(1)]

                    closs = F.cross_entropy(
                        logits_trunc.reshape(-1, logits_trunc.size(-1)),
                        shifted_target.reshape(-1),
                        ignore_index=-100,
                        reduction='sum',
                    )
                    total_loss = total_loss + closs
                    total_count = total_count + 1

        return (total_loss / max(total_count, 1)) * self.loss_weight
