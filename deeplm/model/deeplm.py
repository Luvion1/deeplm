"""
Deeplm: Main model combining MLA attention, MoE, Hyper-Connections, and MTP.

Architecture:
  Embed → N× TransformerBlock (MLA + MoE + Hyper-Connections) → Norm → LM Head

Optional:
  - MTP (Multi-Token Prediction) for richer learning signal
  - Tied embeddings (weight sharing between embed and lm_head)
  - Gradient checkpointing for memory efficiency
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import DeeplmConfig
from .transformer_block import TransformerBlock
from .mtp import MTPHead


class DeeplmModel(nn.Module):
    """Deeplm: Decoder-only transformer with MLA, MoE, and Hyper-Connections.

    Args:
        config: Model configuration (DeeplmConfig)
    """

    def __init__(self, config: DeeplmConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.architecture.hidden_size
        self.max_seq_length = config.architecture.max_seq_length
        self.gradient_checkpointing_enabled = False

        self.embed_tokens = nn.Embedding(config.vocab_size, config.architecture.hidden_size)
        self.embed_scale = config.architecture.hidden_size ** 0.5

        self.layers = nn.ModuleList([
            TransformerBlock(config, i) for i in range(config.architecture.num_layers)
        ])

        self.norm = nn.LayerNorm(config.architecture.hidden_size)

        if config.output_heads.lm_head.type == "tied":
            self.lm_head = None
        else:
            self.lm_head = nn.Linear(
                config.architecture.hidden_size, config.vocab_size,
                bias=config.output_heads.lm_head.bias,
            )

        if config.mtp.enabled:
            self.mtp_head = MTPHead(
                config.mtp,
                config.architecture.hidden_size,
                config.vocab_size,
                rope_theta=config.architecture.rope_theta,
            )
        else:
            self.mtp_head = None

        self._apply_init_weights()
        self.register_buffer("_causal_mask", None, persistent=False)

    def _apply_init_weights(self):
        """Initialize all parameters."""
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize Linear, Embedding, and LayerNorm modules.

        Uses xavier_normal with depth-scaled std for residual projections,
        following GPT-2 / LLaMA initialization scheme.
        """
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "_is_residual"):
                std = std / (2.0 * self.config.architecture.num_layers) ** 0.5
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _make_causal_mask(self, bsz: int, seq_len: int, device: torch.device,
                          dtype: torch.dtype):
        """Create or return cached causal (lower-triangular) attention mask."""
        if self._causal_mask is not None and self._causal_mask.shape[-1] >= seq_len:
            if self._causal_mask.device != device or self._causal_mask.dtype != dtype:
                self._causal_mask = self._causal_mask.to(device=device, dtype=dtype)
            return self._causal_mask[:, :, :seq_len, :seq_len]
        mask = torch.tril(
            torch.ones(1, 1, seq_len, seq_len, device=device, dtype=dtype)
        )
        if self._causal_mask is None or self._causal_mask.shape[-1] < seq_len:
            self._causal_mask = mask
        return mask

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None,
                labels: torch.Tensor = None, past_key_values=None, use_cache: bool = False,
                output_hidden_states: bool = False, output_mtp_loss: bool = False):
        """
        Forward pass.

        Args:
            input_ids: Token indices (batch, seq_len)
            attention_mask: Optional padding mask (batch, seq_len)
            labels: Optional target tokens for loss computation
            past_key_values: Optional KV cache from previous forward passes
            use_cache: Whether to return updated KV cache
            output_hidden_states: Whether to return all layer hidden states
            output_mtp_loss: Whether to compute MTP auxiliary loss

        Returns:
            dict with keys: loss, logits, past_key_values, hidden_states, mtp_loss
        """
        bsz, seq_len = input_ids.size()
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones(bsz, seq_len, device=device)

        hidden_states = self.embed_tokens(input_ids) * self.embed_scale

        causal_mask = self._make_causal_mask(bsz, seq_len, device=device, dtype=hidden_states.dtype)
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2).to(dtype=hidden_states.dtype)
        attention_mask = attention_mask.masked_fill(causal_mask == 0, float("-inf"))

        all_hidden_states = () if output_hidden_states else None
        all_kv_caches = () if use_cache else None

        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing_enabled and self.training:
                hidden_states, kv_cache = torch.utils.checkpoint.checkpoint(
                    layer, hidden_states, attention_mask, past_kv, use_cache,
                    use_reentrant=True,
                )
            else:
                hidden_states, kv_cache = layer(
                    hidden_states, attention_mask, past_kv, use_cache
                )

            if use_cache:
                all_kv_caches = all_kv_caches + (kv_cache,)

        hidden_states = self.norm(hidden_states)

        # Multi-Token Prediction loss (chunked, no full logits)
        mtp_loss = None
        if self.mtp_head is not None and output_mtp_loss and labels is not None:
            mtp_loss = self.mtp_head.compute_loss_chunked(
                hidden_states, self.lm_head, labels,
                tied_embedding=self.embed_tokens, chunk_size=40,
            )

        # Main language modeling loss — chunked over seq dim (no full logits materialization)
        loss = None
        logits = None
        if labels is not None:
            shift_hidden = hidden_states[:, :-1, :]
            shift_labels = labels[:, 1:]
            bsz, nseql, _ = shift_hidden.shape
            total_loss = 0.0
            total_valid = 0
            chunk_size = 40
            lm_head = self.lm_head
            embed_weight = self.embed_tokens.weight if lm_head is None else None
            for i in range(0, nseql, chunk_size):
                end = min(i + chunk_size, nseql)
                chunk_hidden = shift_hidden[:, i:end, :]
                if lm_head is None:
                    chunk_logits = F.linear(chunk_hidden, embed_weight)
                else:
                    chunk_logits = lm_head(chunk_hidden)
                chunk_labels = shift_labels[:, i:end]
                chunk_loss = F.cross_entropy(
                    chunk_logits.reshape(-1, chunk_logits.size(-1)),
                    chunk_labels.reshape(-1),
                    ignore_index=-100,
                    reduction='sum',
                )
                total_loss = total_loss + chunk_loss
                total_valid = total_valid + (chunk_labels != -100).sum()
            loss = total_loss / total_valid.clamp(min=1)
            if mtp_loss is not None:
                loss = loss + mtp_loss
        else:
            # Inference: compute full logits for generation
            if self.lm_head is None:
                logits = F.linear(hidden_states, self.embed_tokens.weight)
            else:
                logits = self.lm_head(hidden_states)

        return {
            "loss": loss,
            "logits": logits,
            "past_key_values": all_kv_caches,
            "hidden_states": all_hidden_states,
            "mtp_loss": mtp_loss,
        }

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory-efficient training.

        Wraps each layer's forward in torch.utils.checkpoint.checkpoint
        using use_reentrant=True (required for MoE dynamic routing shapes).
        No forward pass duplication needed.
        """
        self.gradient_checkpointing_enabled = True

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False

    def resize_token_embeddings(self, new_num_tokens: int):
        """Resize token embeddings to match a new vocabulary size.

        Preserves existing embeddings and initializes new ones.
        Also resizes the LM head if it's untied.

        Args:
            new_num_tokens: Target vocabulary size

        Returns:
            The new embedding layer
        """
        old_embeddings = self.embed_tokens
        old_num_tokens = old_embeddings.weight.shape[0]
        embed_dim = old_embeddings.weight.shape[1]

        new_embeddings = nn.Embedding(
            new_num_tokens, embed_dim,
            padding_idx=old_embeddings.padding_idx,
        )
        new_embeddings.to(old_embeddings.weight.device, dtype=old_embeddings.weight.dtype)

        num_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_to_copy] = old_embeddings.weight.data[:num_to_copy]
        if new_num_tokens > old_num_tokens:
            nn.init.normal_(new_embeddings.weight.data[old_num_tokens:], mean=0.0, std=0.02)

        self.embed_tokens = new_embeddings
        self.config.vocab_size = new_num_tokens

        if self.lm_head is not None:
            old_lm = self.lm_head
            new_lm = nn.Linear(embed_dim, new_num_tokens, bias=False)
            new_lm.to(old_lm.weight.device, dtype=old_lm.weight.dtype)
            new_lm.weight.data[:num_to_copy] = old_lm.weight.data[:num_to_copy]
            if new_num_tokens > old_num_tokens:
                nn.init.normal_(new_lm.weight.data[old_num_tokens:], mean=0.0, std=0.02)
            self.lm_head = new_lm

        return new_embeddings

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 1024,
                 temperature: float = 0.7, top_k: int = 50, top_p: float = 0.9,
                 do_sample: bool = True, repetition_penalty: float = 1.05,
                 eos_token_id: int = None, use_cache: bool = True):
        """Autoregressive text generation.

        Supports top-k, top-p (nucleus) sampling with repetition penalty.
        """
        self.eval()
        generated = input_ids.clone()
        past_key_values = None

        with torch.inference_mode():
            for _ in range(max_new_tokens):
                if past_key_values is None:
                    current_input = (
                        generated if generated.size(1) <= self.max_seq_length
                        else generated[:, -self.max_seq_length:]
                    )
                else:
                    current_input = generated[:, -1:]

                outputs = self.forward(
                    current_input,
                    use_cache=use_cache,
                    past_key_values=past_key_values,
                )
                logits = outputs["logits"][:, -1, :]
                past_key_values = outputs["past_key_values"]

                if repetition_penalty != 1.0:
                    logits = logits.clone()
                    gen_tokens = generated[:, -self.max_seq_length:]
                    score = logits.gather(-1, gen_tokens)
                    penalized = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
                    logits.scatter_add_(-1, gen_tokens, penalized - score)

                if do_sample and temperature > 0:
                    logits = logits / temperature
                    if top_k > 0:
                        topk_values, topk_indices = torch.topk(logits, min(top_k, logits.size(-1)))
                        logits = torch.full_like(logits, float("-inf"))
                        logits.scatter_(-1, topk_indices, topk_values)
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = False
                        indices_to_remove = sorted_indices_to_remove.scatter(
                            -1, sorted_indices, sorted_indices_to_remove
                        )
                        logits[indices_to_remove] = float("-inf")
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)

                generated = torch.cat([generated, next_token], dim=1)

                if eos_token_id is not None and next_token.item() == eos_token_id:
                    break

        return generated

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())
