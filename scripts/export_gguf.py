#!/usr/bin/env python3
"""
Export Deeplm model to GGUF format with optional BitNet b1.58 quantization.

Usage:
    # Export to F16 GGUF
    python scripts/export_gguf.py --checkpoint checkpoints/final.pt --output deeplm-f16.gguf

    # Export to Q8_0 GGUF (smaller)
    python scripts/export_gguf.py --checkpoint checkpoints/final.pt --output deeplm-q8.gguf --quant Q8_0

    # Export with BitNet ternary quantization
    python scripts/export_gguf.py --checkpoint checkpoints/final.pt --output deeplm-bitnet.gguf --quant Q8_0 --bitnet

    # Export with custom config
    python scripts/export_gguf.py --checkpoint checkpoints/final.pt --config config.yaml --output deeplm.gguf
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
import yaml

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deeplm.config import DeeplmConfig
from deeplm.model.deeplm import DeeplmModel
from deeplm.quantization import apply_bitnet_quantization, export_to_gguf


def load_tokenizer(tokenizer_path: str):
    """Load tokenizer from path."""
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(tokenizer_path)
        return tokenizer
    except Exception as e:
        print(f"Warning: Could not load tokenizer from {tokenizer_path}: {e}")
        return None


def load_checkpoint(checkpoint_path: str, device: str = "cpu"):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    return state_dict


def load_config(config_path: str) -> DeeplmConfig:
    """Load model config."""
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            cfg_dict = yaml.safe_load(f)

        # Extract model config
        model_cfg = cfg_dict.get("model", cfg_dict)
        return DeeplmConfig(**{k: v for k, v in model_cfg.items() if hasattr(DeeplmConfig, k) or k in DeeplmConfig.__annotations__})

    # Default config
    return DeeplmConfig()


def main():
    parser = argparse.ArgumentParser(description="Export Deeplm to GGUF format")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--tokenizer", default="tokenizer/tokenizer.json", help="Path to tokenizer.json")
    parser.add_argument("--output", default="deeplm.gguf", help="Output GGUF file path")
    parser.add_argument("--quant", default="F16", choices=["F16", "F32", "Q8_0", "Q4_0"],
                        help="Quantization type (default: F16)")
    parser.add_argument("--bitnet", action="store_true",
                        help="Apply BitNet b1.58 ternary quantization before export")
    parser.add_argument("--bitnet-scale", default="absmean", choices=["absmean", "absmax"],
                        help="BitNet scaling method")
    parser.add_argument("--device", default="cpu", help="Device for loading model")
    parser.add_argument("--name", default="deeplm", help="Model name for GGUF metadata")
    parser.add_argument("--description", default="Deeplm - Indonesian Language Model",
                        help="Model description for GGUF metadata")
    args = parser.parse_args()

    # ── Load config ──
    print(f"Loading config from {args.config}...")
    config = load_config(args.config)
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  layers: {config.num_hidden_layers}")
    print(f"  heads: {config.num_attention_heads}")

    # ── Create model ──
    print("Creating model...")
    model = DeeplmModel(config)

    # ── Load checkpoint ──
    print(f"Loading checkpoint from {args.checkpoint}...")
    state_dict = load_checkpoint(args.checkpoint, args.device)

    # Handle strict=False for compatibility
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  Missing keys: {len(missing)}")
    if unexpected:
        print(f"  Unexpected keys: {len(unexpected)}")

    model.eval()
    model.to(args.device)

    # ── Apply BitNet quantization ──
    if args.bitnet:
        print(f"Applying BitNet b1.58 ternary quantization (scale={args.bitnet_scale})...")
        stats = apply_bitnet_quantization(model, scale=args.bitnet_scale, verbose=True)
        print(f"  Quantized: {stats['quantized']}/{stats['total_linear']} Linear layers")
        if stats['errors']:
            print(f"  Errors: {stats['errors']}")

    # ── Load tokenizer ──
    tokenizer = None
    if Path(args.tokenizer).exists():
        print(f"Loading tokenizer from {args.tokenizer}...")
        tokenizer = load_tokenizer(args.tokenizer)
    else:
        print(f"Warning: Tokenizer not found at {args.tokenizer}")

    # ── Export to GGUF ──
    metadata = {
        "name": args.name,
        "description": args.description,
        "bitnet_quantized": str(args.bitnet),
    }

    print(f"\nExporting to GGUF: {args.output}")
    print(f"  Quantization: {args.quant}")
    print(f"  BitNet: {args.bitnet}")

    export_to_gguf(
        model=model,
        config=config,
        tokenizer=tokenizer,
        output_path=args.output,
        quant_type=args.quant,
        metadata=metadata,
    )

    print(f"\nExport complete!")
    print(f"  File: {args.output}")
    size_mb = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"  Size: {size_mb:.1f} MB")

    # ── Verify ──
    try:
        import gguf
        reader = gguf.GGUFReader(args.output, "r")
        print(f"\nVerification:")
        print(f"  Tensors: {len(reader.tensors)}")
        print(f"  Architecture: {reader.architecture}")
        for key, val in reader.fields.items():
            if hasattr(val, 'parts') and val.parts:
                print(f"  {key}: {val.parts[0].data}")
    except Exception as e:
        print(f"\nVerification skipped: {e}")


if __name__ == "__main__":
    main()
