#!/usr/bin/env python3
"""
Save Deeplm model weights as safetensors with BitNet b1.58 quantization.

Saves to models/<model_name>/ with:
- model.safetensors — BitNet ternary quantized weights
- model_fp16.safetensors — Full precision weights (for reference)
- config.json — Model configuration
- generation_config.json — Generation defaults
- tokenizer files — Copied from tokenizer path

Usage:
    # Save with BitNet quantization
    python scripts/save_model.py --checkpoint checkpoints/final.pt --name deeplm-108m-bitnet --bitnet

    # Save full precision
    python scripts/save_model.py --checkpoint checkpoints/final.pt --name deeplm-108m-fp16

    # Save with custom config
    python scripts/save_model.py --checkpoint checkpoints/final.pt --config config.yaml --name deeplm-custom
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safetensors.torch import save_file

from deeplm.config import DeeplmConfig
from deeplm.model.deeplm import DeeplmModel
from deeplm.quantization.bitnet_quantize import ternary_quantize, apply_bitnet_quantization


def load_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            return checkpoint["model"]
        else:
            return checkpoint
    return checkpoint


def load_config(config_path: str) -> DeeplmConfig:
    """Load model config."""
    if config_path and Path(config_path).exists():
        return DeeplmConfig.from_yaml(config_path)
    return DeeplmConfig()


def save_safetensors(state_dict: dict, output_path: str, metadata: dict = None):
    """Save state dict as safetensors."""
    # Convert all tensors to contiguous
    clean_dict = {}
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            clean_dict[k] = v.contiguous()

    save_file(clean_dict, output_path, metadata=metadata)
    size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    return size_mb


def save_config(config: DeeplmConfig, output_dir: str):
    """Save model config as JSON."""
    config_dict = {
        "architectures": ["DeeplmModel"],
        "model_type": "deeplm",
        "vocab_size": config.tokenizer.vocab_size,
        "hidden_size": config.architecture.hidden_size,
        "intermediate_size": config.architecture.intermediate_size,
        "num_hidden_layers": config.architecture.num_layers,
        "num_attention_heads": config.architecture.num_attention_heads,
        "num_key_value_heads": config.architecture.num_key_value_heads,
        "max_position_embeddings": config.max_seq_length,
        "rms_norm_eps": 1e-6,
        "rope_theta": config.architecture.rope_theta,
        "rope_dim": config.mla.qk_rope_head_dim,
        "tie_word_embeddings": True,
        # MoE config
        "num_routed_experts": config.moe.num_routed_experts,
        "num_shared_experts": config.moe.num_shared_experts,
        "expert_topk": config.moe.top_k,
        # MLA config
        "q_lora_rank": config.mla.q_lora_rank,
        "kv_lora_rank": config.mla.kv_lora_rank,
        "qk_rope_head_dim": config.mla.qk_rope_head_dim,
        "qk_nope_head_dim": config.mla.qk_nope_head_dim,
        "v_head_dim": config.mla.v_head_dim,
        # MTP config
        "mtp_depth": config.mtp.mtp_depth,
        "mtp_num_layers": config.mtp.num_mtp_layers,
        # BitNet config
        "bitnet_quantized": False,
        "bitnet_scale": "absmean",
    }

    with open(Path(output_dir) / "config.json", "w") as f:
        json.dump(config_dict, f, indent=2)


def save_generation_config(output_dir: str):
    """Save generation config."""
    gen_config = {
        "max_new_tokens": 512,
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "pad_token_id": 0,
        "eos_token_id": 2,
        "bos_token_id": 1,
    }
    with open(Path(output_dir) / "generation_config.json", "w") as f:
        json.dump(gen_config, f, indent=2)


def copy_tokenizer(tokenizer_path: str, output_dir: str):
    """Copy tokenizer files to output directory."""
    if not Path(tokenizer_path).exists():
        return False

    tokenizer_files = ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]
    copied = 0

    for fname in tokenizer_files:
        src = Path(tokenizer_path) / fname
        if not src.exists():
            src = Path(tokenizer_path).parent / fname
        if src.exists():
            shutil.copy2(src, Path(output_dir) / fname)
            copied += 1

    return copied > 0


def main():
    parser = argparse.ArgumentParser(description="Save Deeplm model as safetensors")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--tokenizer", default="tokenizer/", help="Path to tokenizer directory")
    parser.add_argument("--output-dir", default="models", help="Output directory (default: models/)")
    parser.add_argument("--name", default="deeplm", help="Model name (creates models/<name>/)")
    parser.add_argument("--bitnet", action="store_true", help="Apply BitNet b1.58 ternary quantization")
    parser.add_argument("--bitnet-scale", default="absmean", choices=["absmean", "absmax"],
                        help="BitNet scaling method")
    parser.add_argument("--save-fp16", action="store_true",
                        help="Also save full FP16 weights alongside BitNet")
    parser.add_argument("--device", default="cpu", help="Device for loading model")
    args = parser.parse_args()

    # ── Setup output directory ──
    output_dir = Path(args.output_dir) / args.name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # ── Load config ──
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    print(f"  vocab_size: {config.tokenizer.vocab_size:,}")
    print(f"  hidden_size: {config.architecture.hidden_size}")
    print(f"  layers: {config.architecture.num_layers}")
    print(f"  heads: {config.architecture.num_attention_heads}")

    # ── Create model ──
    print("Creating model...")
    model = DeeplmModel(config)

    # ── Load checkpoint ──
    print(f"Loading checkpoint from {args.checkpoint}...")
    state_dict = load_checkpoint(args.checkpoint, args.device)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    model.eval()
    model.to(args.device)

    # ── Save FP16 weights (optional) ──
    if args.save_fp16:
        print("\nSaving FP16 weights...")
        fp16_dict = {k: v.half().cpu() for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}
        size = save_safetensors(fp16_dict, str(output_dir / "model_fp16.safetensors"),
                                metadata={"format": "pt", "dtype": "fp16"})
        print(f"  model_fp16.safetensors: {size:.1f} MB")

    # ── Apply BitNet quantization ──
    if args.bitnet:
        print(f"\nApplying BitNet b1.58 ternary quantization (scale={args.bitnet_scale})...")
        stats = apply_bitnet_quantization(model, scale=args.bitnet_scale, verbose=True)
        print(f"  Quantized: {stats['quantized']}/{stats['total_linear']} Linear layers")
        if stats['errors']:
            print(f"  Errors: {stats['errors']}")

        # Save BitNet quantized weights
        print("\nSaving BitNet quantized weights...")
        bitnet_dict = {k: v.half().cpu() for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}
        size = save_safetensors(bitnet_dict, str(output_dir / "model.safetensors"),
                                metadata={
                                    "format": "pt",
                                    "dtype": "fp16",
                                    "bitnet": "true",
                                    "bitnet_scale": args.bitnet_scale,
                                    "quantized_layers": str(stats['quantized']),
                                    "total_linear": str(stats['total_linear']),
                                })
        print(f"  model.safetensors: {size:.1f} MB")

        # Update config.json with BitNet info
        config_dict_path = output_dir / "config.json"
        if config_dict_path.exists():
            with open(config_dict_path) as f:
                cfg = json.load(f)
            cfg["bitnet_quantized"] = True
            cfg["bitnet_scale"] = args.bitnet_scale
            with open(config_dict_path, "w") as f:
                json.dump(cfg, f, indent=2)
    else:
        # Save full precision as model.safetensors
        print("\nSaving model weights (full precision)...")
        fp_dict = {k: v.half().cpu() for k, v in model.state_dict().items() if isinstance(v, torch.Tensor)}
        size = save_safetensors(fp_dict, str(output_dir / "model.safetensors"),
                                metadata={"format": "pt", "dtype": "fp16"})
        print(f"  model.safetensors: {size:.1f} MB")

    # ── Save config and generation config ──
    print("\nSaving config files...")
    save_config(config, str(output_dir))
    save_generation_config(str(output_dir))
    print("  config.json")
    print("  generation_config.json")

    # ── Copy tokenizer ──
    print("\nCopying tokenizer...")
    if copy_tokenizer(args.tokenizer, str(output_dir)):
        print(f"  Tokenizer files copied from {args.tokenizer}")
    else:
        print(f"  Warning: No tokenizer files found at {args.tokenizer}")

    # ── Summary ──
    print(f"\n{'='*50}")
    print(f"Model saved to: {output_dir}")
    print(f"Files:")
    for f in sorted(output_dir.iterdir()):
        size = f.stat().st_size / (1024 * 1024) if f.is_file() else 0
        print(f"  {f.name}: {size:.1f} MB" if f.is_file() else f"  {f.name}/")
    print(f"{'='*50}")

    # ── Verify ──
    print("\nVerifying saved model...")
    try:
        from safetensors import safe_open
        with safe_open(str(output_dir / "model.safetensors"), framework="pt", device="cpu") as f:
            keys = f.keys()
            print(f"  Tensors: {len(keys)}")
            # Show first few keys
            for k in list(keys)[:5]:
                tensor = f.get_tensor(k)
                print(f"    {k}: {list(tensor.shape)} dtype={tensor.dtype}")
            if len(keys) > 5:
                print(f"    ... and {len(keys) - 5} more tensors")
    except Exception as e:
        print(f"  Verification error: {e}")


if __name__ == "__main__":
    main()
