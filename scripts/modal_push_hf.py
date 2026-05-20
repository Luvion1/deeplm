"""
Modal ↔ HuggingFace: Sinkronisasi bidirectional.
  - Source code: selalu pull dari GitHub
  - Model weights: cari di volume dulu, fallback download dari HF
  - Upload: push semua aset ke HF
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import modal
from huggingface_hub import HfApi, login, create_repo, snapshot_download

VOL = "/vol"
volume = modal.Volume.from_name("deeplm-checkpoints", create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel")
    .run_commands("apt-get update -qq && apt-get install -y -qq git")
    .pip_install("huggingface-hub", "tokenizers", "PyYAML")
)

_hf_token = os.environ.get("HF_TOKEN", "")
_hf_secrets = [modal.Secret.from_dict({"HF_TOKEN": _hf_token})] if _hf_token else []

app = modal.App("deeplm-push-hf", image=image)

MODEL_CARD = """---
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
  - self-evolution
  - autotuner
  - deeplm
base_model: "none"
---

# Deeplm-105M

Indonesian language model with novel architecture combining MLA, MoE, Hyper-Connections, Hybrid Attention, Multi-Token Prediction, and Self-Evolution Framework with autonomous AutoTuner.

## Architecture

| Component | Detail |
|-----------|--------|
| **Total Parameters** | {params:,} (~{params_m:.0f}M) |
| **Vocabulary** | {vocab_size:,} (BBPE) |
| **Layers** | 10 Transformer blocks |
| **Hidden Size** | 512 |
| **Feed-Forward** | 2048 (SwiGLU, 4× hidden) |
| **Attention Heads** | 8 query heads, 1 KV head (MQA) |
| **Head Dim** | 128 (64 RoPE + 64 NoPE) |
| **Max Seq Length** | 4096 |
| **RoPE Theta** | 50,000 |
| **Attention** | MLA (Multi-head Latent Attention) |
| **FFN** | MoE (4 routed + 1 shared experts, top-k=2) |
| **Residual** | Hyper-Connections with Sinkhorn routing |
| **Hybrid Attention** | 3 softmax + 7 Lightning layers |
| **Prediction** | MTP (Multi-Token Prediction, depth=2, 2 MTP layers) |
| **Self-Evolution** | Autonomous research loop (100+ rounds) |
| **Embeddings** | Tied (shared between input/output) |
| **AutoTuner** | Adaptive energy-based optimizer scheduler |
| **Dtype** | float32 (Hyper-Connections stability) |

## Key Innovations

### 1. Multi-head Latent Attention (MLA) — *DeepSeek V4 / Kimi K2.6*
- Q compressed: hidden → q_lora_rank(192) → Layernorm → q_up(8 × 128)
- KV compressed: hidden → [kv_latent(64) + k_rope(64)] → kv_up → [k_nope(64) + v(128)] × 8 heads
- Entire KV cache per token: just 128 dims (64 latent + 64 rope) — **~8× smaller** than standard MHA
- Decoupled RoPE applied only to 64-dim k_pe, content path stays RoPE-free
- Absorption trick pre-computes W_UK @ W_UV for faster inference
- MQA-style: KV decomposed once, expanded to all query heads

### 2. Mixture of Experts (MoE) — *DeepSeek V4 / Kimi K2.6*
- 4 routed experts + 1 shared expert (always active, Kimi K2.6 style)
- Top-k=2 routing: each token activates only 2 experts
- **sqrt(softplus(x))** scoring for numerical stability (DeepSeek V4)
- **Bias-based load balancing** (no auxiliary loss, no gradient interference)
- Per-expert routing bias auto-updates to balance token assignments
- SwiGLU activation in every expert (fused gate+up projection)
- Expert affinity memory tracks token-expert history

### 3. Hyper-Connections with Sinkhorn Routing — *DeepSeek V4*
- Replaces standard residual connections with learned routing
- 4 connection types: **identity**, **transform**, **gate**, **skip**
- Sinkhorn-Knopp normalization (2 iterations) for doubly-stochastic weights
- Input-dependent routing via gating network
- Type-specific learnable biases initialized per config
- Pre-LayerNorm on layer output before routing

### 4. Hybrid Attention — *MiniMax M2.7*
- 3 softmax layers (indices 0, 4, 8): Standard MLA with full causal attention
- 7 linear layers (1, 2, 3, 5, 6, 7, 9): MLA + LightningAttentionV2 **50/50 blend**
- LightningAttentionV2: O(n) complexity with intra-block softmax + inter-block KV product
- Incremental KV state for efficient autoregressive generation
- ReLU/Swish activation replaces softmax in linear path

### 5. Multi-Token Prediction (MTP) — *DeepSeek V4*
- 2 MTP layers, each predicting 2 tokens ahead (mtp_depth=2)
- Projection block: Linear → LayerNorm → GELU → Linear + residual skip
- RoPE positional encoding on reduced dim (hidden/4) for efficiency
- Tied LM head shares parameters with main embedding layer
- Chunked computation (chunk_size=40) to avoid full (B, S, V) logits
- Loss weight: 0.3 × cross-entropy of future token predictions

### 6. Self-Evolution Framework — *MiniMax M2.7 / Deeplm*
- Autonomous 8-phase research loop: hypothesis → design → execute → analyze → diagnose → fix → evaluate → decide
- 100+ autonomous optimization rounds per training cycle
- 3 feedback chain episodes for meta-learning

### 7. AutoTuner — *Deeplm custom*
- Energy-based adaptive hyperparameter controller
- Phase-aware dynamics (warmup → exploration → balanced → exploitation)
- Online dynamics model: causal (Δloss/Δlr, Δloss/Δwd) sensitivity tracking
- Multi-timescale loss EMAs (short=0.9, med=0.98, long=0.995)
- Gradient noise scale monitoring
- Cosine similarity for gradient direction tracking
- Layer health monitoring per gradient group
- Failure-aware rollback with revive mechanism
- Strategic planner: multi-step scheduled adjustments with plan accuracy tracking
- Trajectory predictor: loss curve fitting with convergence estimation

## Training

| Config | Value |
|--------|-------|
| **Dataset** | Wikipedia-id (Indonesian, 5M samples) + KBBI dictionary |
| **Tokenizer** | 32K BBPE |
| **Optimizer** | SGD Nesterov (momentum=0.9, weight_decay=0.1) |
| **LR Schedule** | Cosine (warmup 3%) |
| **Base LR** | 3e-4 |
| **Effective Batch** | 32 (8 × 4 grad_accum) |
| **Sequence Length** | 2048 |
| **Max Grad Norm** | 1.0 |
| **Steps** | 8,000 (latest checkpoint) |
| **GPU** | A10G (24GB) |
| **Dtype** | float32 |

## Training Algorithms

| Algorithm | Status | Description |
|-----------|--------|-------------|
| Curriculum Learning | Active | 4-tier easy→hard progression by text length |
| Dynamic Sampling | Active | Adaptive category mix based on per-category loss |
| Difficulty Scheduling | Active | 4 phases: Token Learning → Syntax → Reasoning → Expert |
| MoE Balancing | Active | Bias-based load-balanced routing |
| AutoTuner | Active | AI adaptive hyperparameter control |
| MTP | Active | Auxiliary multi-token prediction loss |
| Curriculum Scheduling | Inactive | Loss-based adaptive difficulty |
| Reflection Training | Inactive | High-loss example replay |
| Synthetic Evolution | Inactive | Model-generated training data |

## AutoTuner State (Step 8000)

| Metric | Value |
|--------|-------|
| **Phase** | Balanced |
| **LR Multiplier** | 0.74× |
| **Grad Norm Multiplier** | 1.0× |
| **Best Loss** | 4.01 |
| **Best Step** | 2,139 |
| **Plateau** | 840 steps |
| **Confidence** | 0.67 |
| **Adjustments Made** | 118 |
| **Revive Attempts** | 2 |
| **Diagnosis** | Plateauing |
| **Plan Strategy** | Explore & Escape |

## Training Curves

![Training Curves](training_curves.png)

*6-panel training visualization: main loss, MTP loss, learning rate, gradient norm, early convergence zoom, and main vs MTP loss correlation.*

## Files

| File | Description |
|------|-------------|
| `model.pt` | Model weights (~105M params, 419MB) |
| `optimizer.pt` | SGD Nesterov optimizer state |
| `best.pt` | Best checkpoint by eval loss |
| `training_state.json` | Full training state including AutoTuner state |
| `tokenizer.json` | BBPE tokenizer (32K vocab) |
| `tokenizer_config.json` | Tokenizer configuration |
| `config.yaml` | Model configuration (DeeplmConfig defaults) |
| `training_curves.png` | Training curves visualization (6 panels) |
| `deeplm/` | Full model source code (model, training, data, inference, self-evolution) |

## Architecture Details

```
Input (token ids)
    ↓
Embed (512d, scaled √512)
    ↓
┌─────────────────────────────────────┐
│  ×10 Transformer Blocks:           │
│    LayerNorm → Hybrid Attention    │
│      MLA (all layers)              │
│      + Lightning (7 layers, 50%)   │
│    Hyper-Connection (Sinkhorn)     │
│    LayerNorm → MoE                 │
│      4 routed experts (top-k=2)    │
│      + 1 shared expert             │
│    Standard residual               │
└─────────────────────────────────────┘
    ↓
LayerNorm (final)
    ↓
LM Head (tied with Embed, no bias)
    ↓
Output logits (32K vocab)

MTP Head (parallel):
  Hidden → 2× MTP layers
    depth-1: predict +1 token
    depth-2: predict +2 tokens
  Loss: cross-entropy × 0.3
```

## Usage

```python
import torch
from deeplm.config import DeeplmConfig
from deeplm.model.deeplm import DeeplmModel

config = DeeplmConfig()
model = DeeplmModel(config)
model.load_state_dict(torch.load("model.pt", map_location="cpu"), strict=False)
model.eval()

input_ids = torch.tensor([[1, 2, 3]])
output = model.generate(
    input_ids,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
)
print(output)
```
"""


def _upload_dir(api, repo_id, local_dir, repo_subdir=""):
    for f in Path(local_dir).rglob("*.py"):
        if "__pycache__" in str(f):
            continue
        rel = f.relative_to(local_dir)
        repo_path = f"{repo_subdir}/{rel}" if repo_subdir else str(rel)
        api.upload_file(path_or_fileobj=str(f), path_in_repo=repo_path, repo_id=repo_id)


def _ensure_file(api, vol_path: Path, repo_path: str, hf_repo: str, label: str):
    """Upload from volume; fallback download dari HF."""
    if vol_path.exists():
        sz = vol_path.stat().st_size
        if sz > 50 * 1024 * 1024:
            print(f"Uploading {label} ({sz / 1e9:.1f}GB)...")
        api.upload_file(path_or_fileobj=str(vol_path), path_in_repo=repo_path, repo_id=hf_repo)
        print(f"  \u2713 {label} (from volume)")
        return True

    # Fallback: download from HF to volume, then upload
    print(f"  \u26a0 {label} not on volume — downloading from HF...")
    try:
        dl = snapshot_download(repo_id=hf_repo, allow_patterns=repo_path, local_dir=vol_path.parent)
        src = Path(dl) / repo_path
        if src.exists():
            vol_path.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(src, vol_path)
            api.upload_file(path_or_fileobj=str(vol_path), path_in_repo=repo_path, repo_id=hf_repo)
            print(f"  \u2713 {label} (from HF fallback)")
            return True
    except Exception as e:
        print(f"  \u2717 HF fallback failed: {e}")
    return False


@app.function(
    gpu=None,
    cpu=2,
    memory=4096,
    timeout=1200,
    volumes={VOL: volume},
    secrets=_hf_secrets,
)
def push_to_hf(
    repo_id: str = "samcheng0/deeplm-108m",
    ckpt_step: int = 8000,
    gh_repo: str = "Luvion1/deeplm",
):
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: HF_TOKEN not set")
        return

    login(token)
    api = HfApi()
    create_repo(repo_id, repo_type="model", exist_ok=True)
    print(f"\u2713 Repo {repo_id} ready\n")

    ckpt_dir = Path(VOL) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 1-4. Model, optimizer, state, best — dengan fallback HF
    _ensure_file(api, ckpt_dir / f"ckpt-{ckpt_step}.pt", "model.pt", repo_id, "model.pt")
    _ensure_file(api, ckpt_dir / f"opt-{ckpt_step}.pt", "optimizer.pt", repo_id, "optimizer.pt")
    _ensure_file(api, ckpt_dir / f"state-{ckpt_step}.json", "training_state.json", repo_id, "training_state.json")
    _ensure_file(api, ckpt_dir / "best.pt", "best.pt", repo_id, "best.pt")

    # 5. Tokenizer
    tok_dir = Path(VOL) / "tokenizer"
    tok_dir.mkdir(parents=True, exist_ok=True)
    _ensure_file(api, tok_dir / "tokenizer.json", "tokenizer.json", repo_id, "tokenizer.json")
    _ensure_file(api, tok_dir / "tokenizer_config.json", "tokenizer_config.json", repo_id, "tokenizer_config.json")

    # 6. Source code — selalu fresh dari GitHub
    print("\nCloning source code from GitHub...")
    repo_dir = Path(VOL) / "deeplm"
    if not repo_dir.exists():
        url = f"https://github.com/{gh_repo}.git"
        subprocess.run(["git", "clone", "--depth=1", url, str(repo_dir)], check=True, capture_output=True)
    else:
        subprocess.run(["git", "-C", str(repo_dir), "pull", "--ff-only"], capture_output=True)

    for subdir in ["deeplm/model", "deeplm/training", "deeplm/data",
                   "deeplm/inference", "deeplm/self_evolution"]:
        p = repo_dir / subdir
        if p.exists():
            _upload_dir(api, repo_id, p, subdir)
    for fname in ["deeplm/config.py", "deeplm/__init__.py"]:
        fp = repo_dir / fname
        if fp.exists():
            api.upload_file(path_or_fileobj=str(fp), path_in_repo=fname, repo_id=repo_id)
            print(f"  \u2713 {fname}")
    print("  \u2713 source code")

    # 7. Config YAML dari DeeplmConfig defaults
    sys.path.insert(0, str(repo_dir))
    from deeplm.config import DeeplmConfig
    import dataclasses, yaml as _yaml
    cfg = DeeplmConfig()
    cfg_dict = dataclasses.asdict(cfg)
    cfg_dict['model_name'] = 'Deeplm'
    cfg_dict['version'] = '1.0.0'
    api.upload_file(
        path_or_fileobj=_yaml.dump(cfg_dict, default_flow_style=False, sort_keys=False, allow_unicode=True).encode(),
        path_in_repo="config.yaml",
        repo_id=repo_id,
    )
    print("  \u2713 config.yaml (from DeeplmConfig defaults)")

    # 8. Model card
    vocab = 32_000
    params = 104_747_048
    card = MODEL_CARD.format(params=params, params_m=params / 1_000_000, vocab_size=vocab)
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=repo_id)
    print("  \u2713 README.md (model card)")

    print(f"\n\u2713 Done! https://huggingface.co/{repo_id}")


@app.local_entrypoint()
def main(repo_id: str = "samcheng0/deeplm-108m", ckpt_step: int = 8000):
    push_to_hf.remote(repo_id=repo_id, ckpt_step=ckpt_step)

    # Upload local assets
    from huggingface_hub import HfApi, login
    token = os.environ.get("HF_TOKEN", "")
    if token:
        login(token)
        api = HfApi()
        curves = Path(__file__).resolve().parent.parent / "deeplm_output" / "training_curves.png"
        if curves.exists():
            api.upload_file(path_or_fileobj=str(curves), path_in_repo="training_curves.png", repo_id=repo_id)
            print(f"  \u2713 training_curves.png (from local)")
        print(f"\n\u2713 Final: https://huggingface.co/{repo_id}")
