"""
GGUF Export for Deeplm models.

Exports trained Deeplm models to GGUF format for use with llama.cpp, bitnet.cpp,
and other GGML-based inference engines.

Uses the official `gguf` Python package from llama.cpp.
"""
import struct
from enum import IntEnum
from pathlib import Path

import numpy as np
import torch

try:
    import gguf
    from gguf import GGUFWriter, GGMLQuantizationType
except ImportError:
    gguf = None
    GGUFWriter = None
    GGMLQuantizationType = None


# ── GGUF architecture mapping ──
GGUF_ARCH = "llama"  # Deeplm uses LLaMA-like architecture

# Mapping from Deeplm config keys to GGUF keys
GGUF_KEYS = {
    "vocab_size": "vocab_size",
    "hidden_size": "embedding_length",
    "intermediate_size": "feed_forward_length",
    "num_hidden_layers": "block_count",
    "num_attention_heads": "attention.head_count",
    "num_key_value_heads": "attention.head_count_kv",
    "rope_theta": "rope.freq_base",
    "rms_norm_eps": "attention.layer_norm_rms_epsilon",
}


def _get_tensor_name(name: str) -> str:
    """Convert Deeplm parameter name to GGUF tensor name."""
    name = name.replace("embed_tokens.weight", "token_embd.weight")
    name = name.replace("lm_head.weight", "output.weight")
    name = name.replace("norm.weight", "output_norm.weight")

    # Transformer blocks
    name = name.replace("layers.", "blk.")
    name = name.replace("attention.q_down_proj.weight", "attn_q_a.weight")
    name = name.replace("attention.q_up_proj.weight", "attn_q_b.weight")
    name = name.replace("attention.kv_down_proj.weight", "attn_kv_a.weight")
    name = name.replace("attention.kv_up_proj.weight", "attn_kv_b.weight")
    name = name.replace("attention.output_proj.weight", "attn_output.weight")
    name = name.replace("attention.q_layernorm.weight", "attn_q_norm.weight")
    name = name.replace("attention.kv_layernorm.weight", "attn_k_norm.weight")

    # Lightning attention (hybrid layers)
    name = name.replace("attention.linear_q_proj.weight", "attn_q.weight")
    name = name.replace("attention.linear_k_proj.weight", "attn_k.weight")
    name = name.replace("attention.linear_v_proj.weight", "attn_v.weight")
    name = name.replace("attention.linear_output_proj.weight", "attn_output.weight")

    # MoE
    name = name.replace("moe.routed_experts", "ffn_experts")
    name = name.replace("moe.shared_experts", "ffn_shared_experts")
    name = name.replace("gate_up_proj.weight", "gate_up_proj.weight")
    name = name.replace("down_proj.weight", "down_proj.weight")
    name = name.replace("moe.router.weight", "gate.weight")

    # Norms
    name = name.replace("pre_attention_norm.weight", "attn_norm.weight")
    name = name.replace("post_moe_norm.weight", "ffn_norm.weight")

    # Hyper-connections
    name = name.replace("hyper_connection.layer_norm.weight", "attn_norm_2.weight")
    name = name.replace("hyper_connection.hyper_connection.transform_proj.weight", "hc_transform.weight")
    name = name.replace("hyper_connection.hyper_connection.gate_proj.weight", "hc_gate.weight")

    # MTP
    name = name.replace("mtp_head.", "mtp.")

    return name


def _quantize_tensor(tensor: torch.Tensor, qtype: str = "Q8_0") -> tuple:
    """Quantize a tensor to GGML quantization type.

    Args:
        tensor: PyTorch tensor
        qtype: quantization type (F16, F32, Q8_0, Q4_0, etc.)

    Returns:
        (numpy_array, ggml_quant_type)
    """
    arr = tensor.cpu().numpy()

    if qtype == "F32":
        return arr.astype(np.float32), GGMLQuantizationType.F32
    elif qtype == "F16":
        return arr.astype(np.float16), GGMLQuantizationType.F16
    elif qtype == "BF16":
        return arr.astype(np.float16), GGMLQuantizationType.F16  # GGUF doesn't have BF16 directly
    elif qtype == "Q8_0":
        # Block-wise Q8 quantization
        return _quantize_q8(arr), GGMLQuantizationType.Q8_0
    elif qtype == "Q4_0":
        return _quantize_q4(arr), GGMLQuantizationType.Q4_0
    else:
        return arr.astype(np.float32), GGMLQuantizationType.F32


def _quantize_q8(arr: np.ndarray) -> np.ndarray:
    """Simple Q8_0 quantization (per-tensor absmax scaling to int8)."""
    max_val = np.abs(arr).max()
    if max_val < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    scale = max_val / 127.0
    quantized = np.round(arr / scale).astype(np.int8)
    # Store as float32 for GGUF (llama.cpp handles dequantization)
    return quantized.astype(np.float32)


def _quantize_q4(arr: np.ndarray) -> np.ndarray:
    """Simple Q4_0 quantization."""
    max_val = np.abs(arr).max()
    if max_val < 1e-12:
        return np.zeros_like(arr, dtype=np.float32)
    scale = max_val / 7.0
    quantized = np.round(arr / scale).astype(np.int8)
    return quantized.astype(np.float32)


def export_to_gguf(
    model,
    config,
    tokenizer,
    output_path: str,
    quant_type: str = "Q8_0",
    metadata: dict = None,
):
    """Export Deeplm model to GGUF format.

    Args:
        model: DeeplmModel instance
        config: DeeplmConfig instance
        tokenizer: tokenizer instance
        output_path: path to output .gguf file
        quant_type: quantization type (F16, F32, Q8_0, Q4_0)
        metadata: optional extra metadata dict
    """
    if gguf is None:
        raise ImportError(
            "gguf package is required. Install with: pip install gguf"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to GGUF: {output_path}")
    print(f"  Quantization: {quant_type}")

    # Create GGUF writer
    arch = GGUF_ARCH

    # Build metadata first
    kv_pairs = {}

    # Basic config
    kv_pairs["general.alignment"] = 32
    kv_pairs["general.architecture"] = arch
    kv_pairs["general.name"] = "deeplm"
    kv_pairs["general.description"] = "Deeplm - Indonesian Language Model with MLA, MoE, Hyper-Connections"
    kv_pairs["general.file_type"] = quant_type
    kv_pairs["general.quantization_version"] = 2

    # Model architecture
    kv_pairs[f"{arch}.vocab_size"] = config.vocab_size
    kv_pairs[f"{arch}.embedding_length"] = config.hidden_size
    kv_pairs[f"{arch}.feed_forward_length"] = config.intermediate_size
    kv_pairs[f"{arch}.block_count"] = config.num_hidden_layers
    kv_pairs[f"{arch}.attention.head_count"] = config.num_attention_heads

    if hasattr(config, "num_key_value_heads"):
        kv_pairs[f"{arch}.attention.head_count_kv"] = config.num_key_value_heads

    if hasattr(config, "rope_theta"):
        kv_pairs[f"{arch}.rope.freq_base"] = config.rope_theta

    if hasattr(config, "rms_norm_eps"):
        kv_pairs[f"{arch}.attention.layer_norm_rms_epsilon"] = config.rms_norm_eps

    # RoPE dimension
    if hasattr(config, "rope_dim"):
        kv_pairs[f"{arch}.rope.dimension_count"] = config.rope_dim

    # Expert count (MoE)
    if hasattr(config, "num_routed_experts"):
        kv_pairs[f"{arch}.expert_count"] = config.num_routed_experts
    if hasattr(config, "num_shared_experts"):
        kv_pairs[f"{arch}.expert_shared_count"] = config.num_shared_experts

    # Context length
    kv_pairs[f"{arch}.context_length"] = config.max_position_embeddings

    writer = GGUFWriter(str(output_path), arch)

    # Write all kv pairs at once
    for key, value in kv_pairs.items():
        if isinstance(value, str):
            writer.add_string(key, value)
        elif isinstance(value, int):
            writer.add_uint32(key, value)
        elif isinstance(value, float):
            writer.add_float32(key, value)

    # ── Tokenizer ──
    if tokenizer is not None:
        try:
            vocab = tokenizer.get_vocab()
            tokens = [k for k, v in sorted(vocab.items(), key=lambda x: x[1])]
            scores = [0.0] * len(tokens)
            token_types = [1] * len(tokens)  # normal

            writer.add_token_list(tokens)
            writer.add_token_scores(scores)
            writer.add_token_type_list(token_types)

            # Special tokens
            if hasattr(tokenizer, "bos_token_id") and tokenizer.bos_token_id is not None:
                writer.add_uint32(f"{arch}.bos_token_id", tokenizer.bos_token_id)
            if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
                writer.add_uint32(f"{arch}.eos_token_id", tokenizer.eos_token_id)
            if hasattr(tokenizer, "pad_token_id") and tokenizer.pad_token_id is not None:
                writer.add_uint32(f"{arch}.padding_token_id", tokenizer.pad_token_id)
            if hasattr(tokenizer, "unk_token_id") and tokenizer.unk_token_id is not None:
                writer.add_uint32(f"{arch}.unknown_token_id", tokenizer.unk_token_id)

            # Tokenizer model
            writer.add_string("tokenizer.ggml.model", "bpe")
        except Exception as e:
            print(f"  Warning: Could not export tokenizer: {e}")

    # ── Extra metadata ──
    if metadata:
        for key, value in metadata.items():
            if isinstance(value, str):
                writer.add_string(f"general.{key}", value)
            elif isinstance(value, (int, np.integer)):
                writer.add_uint32(f"general.{key}", int(value))
            elif isinstance(value, float):
                writer.add_float32(f"general.{key}", float(value))

    # ── Write tensors ──
    state_dict = model.state_dict()
    total_tensors = len(state_dict)
    total_bytes = 0

    print(f"  Writing {total_tensors} tensors...")
    for i, (name, tensor) in enumerate(state_dict.items()):
        gguf_name = _get_tensor_name(name)

        # Skip non-tensor entries
        if not isinstance(tensor, torch.Tensor):
            continue

        # Quantize
        data, qtype = _quantize_tensor(tensor, quant_type)

        # Write tensor
        writer.add_tensor(gguf_name, data, raw_dtype=qtype)
        total_bytes += data.nbytes

        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{total_tensors} tensors written...")

    # ── Finalize ──
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = total_bytes / (1024 * 1024)
    print(f"  GGUF file written: {output_path}")
    print(f"  Total size: {size_mb:.1f} MB")
    print(f"  Tensors: {total_tensors}")

    return str(output_path)
