#!/usr/bin/env python3
"""
Upload Deeplm model to HuggingFace Hub.

Uploads:
- Model weights (from /vol/checkpoints/final.pt or best.pt)
- All model code (deeplm/)
- Config (config.yaml)
- Tokenizer (if available)
- Model card (README.md)

Usage:
    python scripts/upload_to_hf.py --repo-id samcheng0/deeplm-98m --token hf_xxx
"""
import argparse
import json
import os
from pathlib import Path

from huggingface_hub import HfApi, login, create_repo


def create_model_card(repo_id: str, params: int, vocab_size: int, **kwargs) -> str:
    """Generate model card README."""
    return f"""---
language: id
license: apache-2.0
library_name: transformers
tags:
  - indonesian
  - language-model
  - moe
  - mla
  - hyper-connections
  - lightning-attention
  - multi-token-prediction
base_model: "none"
---

# Deeplm-98M

Indonesian language model with novel architecture combining MLA, MoE, Hyper-Connections, Hybrid Attention, and Multi-Token Prediction.

## Architecture

| Component | Detail |
|-----------|--------|
| **Total Parameters** | {params:,} |
| **Vocabulary** | {vocab_size:,} (BBPE) |
| **Layers** | 8 Transformer blocks |
| **Hidden Size** | 384 |
| **Attention** | MLA (Multi-head Latent Attention) |
| **FFN** | MoE (4 routed + 1 shared experts, top-k=2) |
| **Residual** | Hyper-Connections with Sinkhorn routing |
| **Linear Attention** | LightningAttentionV2 (5/8 layers) |
| **Prediction** | MTP (Multi-Token Prediction, depth=2) |
| **Embeddings** | Tied (shared between input/output) |

## Key Innovations

### 1. Multi-head Latent Attention (MLA)
Compressed KV cache via low-rank latent space (~10x memory savings vs MHA).
- Q: hidden → q_lora_rank(192) → num_heads × head_dim
- KV: hidden → kv_lora_rank(128) + rope_dim(32) → decompressed to full heads
- Decoupled RoPE on small portion of Q/K

### 2. Mixture of Experts (MoE)
- 4 routed experts + 1 shared expert (always active)
- Top-k=2 routing with sqrt(softplus) scoring
- Bias-based load balancing (no auxiliary loss)
- Grouped dispatch via sorting for efficient processing

### 3. Hyper-Connections
Replaces standard residuals with learned routing over 4 connection types:
- Identity, Transform, Gate, Skip
- Sinkhorn-Knopp normalization for doubly-stochastic weights
- Input-dependent routing with type biases

### 4. Hybrid Attention
- 3 softmax layers (0, 4, 7): Standard MLA
- 5 linear layers (1, 2, 3, 5, 6): MLA + LightningAttentionV2 blend
- LightningAttentionV2: O(n) complexity with incremental KV state

### 5. Multi-Token Prediction (MTP)
- 2 prediction depths per MTP layer
- RoPE positional encoding (reduced dim for efficiency)
- Skip connections in projections
- Tied LM head for parameter sharing

## Training

- **Dataset**: GSM8K (7.5K) + CEFR CEP (3.48M) interleaved
- **Tokenizer**: 128K BBPE (KBBI + Corpus-Indonesia + WordNet)
- **Optimizer**: AdamW with cosine LR schedule
- **Batch Size**: 2 × 16 grad_accum = 32 effective
- **Sequence Length**: 2048
- **Learning Rate**: 8e-4 with 3% warmup

## Usage

```python
import torch
from deeplm.config import DeeplmConfig
from deeplm.model.deeplm import DeeplmModel

# Load model
config = DeeplmConfig()
model = DeeplmModel(config)
model.load_state_dict(torch.load("model.pt", map_location="cpu"), strict=False)
model.eval()

# Generate
input_ids = torch.tensor([[1, 2, 3]])  # tokenized input
output = model.generate(input_ids, max_new_tokens=128, do_sample=True, temperature=0.7)
```

## Files

- `model.pt` — Model weights
- `config.yaml` — Model configuration
- `tokenizer.json` — Tokenizer (128K vocab)
- `deeplm/` — Full model code
"""


def upload_model(repo_id: str, token: str = None, model_path: str = None,
                 tokenizer_path: str = None, config_yaml: str = None):
    """Upload model and all files to HuggingFace."""
    if token:
        login(token)

    api = HfApi()

    # Create repo if not exists
    try:
        create_repo(repo_id, repo_type="model", exist_ok=True)
        print(f"✓ Repo {repo_id} exists/created")
    except Exception as e:
        print(f"✗ Failed to create repo: {e}")
        return

    # Upload model weights
    if model_path and Path(model_path).exists():
        print(f"Uploading model weights: {model_path}")
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo="model.pt",
            repo_id=repo_id,
        )
        print("  ✓ model.pt")
    else:
        print("  ⚠ No model weights found (skipping)")

    # Upload optimizer state (for resume training)
    opt_path = Path(model_path).parent / f"opt-{Path(model_path).stem.split('-')[-1].replace('.pt','')}.pt" if model_path else None
    for p in ["/root/deeplm/opt.pt", "opt.pt"]:
        if Path(p).exists():
            opt_path = Path(p)
            break

    if opt_path and opt_path.exists():
        print(f"Uploading optimizer state: {opt_path}")
        api.upload_file(
            path_or_fileobj=str(opt_path),
            path_in_repo="optimizer.pt",
            repo_id=repo_id,
        )
        print("  ✓ optimizer.pt")
    else:
        print("  ⚠ No optimizer state found (skipping)")

    # Upload tokenizer
    if tokenizer_path and Path(tokenizer_path).exists():
        print(f"Uploading tokenizer: {tokenizer_path}")
        api.upload_file(
            path_or_fileobj=str(tokenizer_path),
            path_in_repo="tokenizer.json",
            repo_id=repo_id,
        )
        print("  ✓ tokenizer.json")
    else:
        # Try default paths
        for p in ["/vol/tokenizer/tokenizer.json", "tokenizer/tokenizer.json"]:
            if Path(p).exists():
                api.upload_file(
                    path_or_fileobj=str(p),
                    path_in_repo="tokenizer.json",
                    repo_id=repo_id,
                )
                print(f"  ✓ tokenizer.json (from {p})")
                break

    # Upload config YAML
    if config_yaml and Path(config_yaml).exists():
        api.upload_file(
            path_or_fileobj=str(config_yaml),
            path_in_repo="config.yaml",
            repo_id=repo_id,
        )
        print("  ✓ config.yaml")

    # Upload model code
    print("Uploading model code...")
    code_dirs = [
        "deeplm/model",
        "deeplm/training",
        "deeplm/data",
        "deeplm/inference",
        "deeplm/self_evolution",
    ]
    for d in code_dirs:
        if Path(d).exists():
            for f in Path(d).rglob("*.py"):
                if "__pycache__" not in str(f):
                    repo_path = str(f)
                    api.upload_file(
                        path_or_fileobj=str(f),
                        path_in_repo=repo_path,
                        repo_id=repo_id,
                    )
                    print(f"  ✓ {repo_path}")

    # Upload config.py
    if Path("deeplm/config.py").exists():
        api.upload_file(
            path_or_fileobj="deeplm/config.py",
            path_in_repo="deeplm/config.py",
            repo_id=repo_id,
        )
        print("  ✓ deeplm/config.py")

    # Upload __init__.py
    if Path("deeplm/__init__.py").exists():
        api.upload_file(
            path_or_fileobj="deeplm/__init__.py",
            path_in_repo="deeplm/__init__.py",
            repo_id=repo_id,
        )
        print("  ✓ deeplm/__init__.py")

    # Create and upload model card
    card = create_model_card(repo_id, params=98_237_216, vocab_size=128_000)
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
    )
    print("  ✓ README.md")

    print(f"\n✓ Upload complete! View at: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="samcheng0/deeplm-98m")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--model-path", default=None,
                        help="Path to model.pt (default: check /vol/checkpoints/)")
    parser.add_argument("--tokenizer-path", default=None,
                        help="Path to tokenizer.json")
    parser.add_argument("--config-yaml", default="configs/train_kbi.yaml")
    args = parser.parse_args()

    # Auto-detect model path
    if not args.model_path:
        for p in [
            "/vol/checkpoints/final.pt",
            "/vol/checkpoints/best.pt",
            "model.pt",
        ]:
            if Path(p).exists():
                args.model_path = p
                break

    upload_model(**vars(args))
