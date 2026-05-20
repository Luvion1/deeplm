"""
Deeplm Training — Modal Cloud — Wikipedia-id Pipeline.

Dataset: indonesian-nlp/wikipedia-id + Jackrong/GLM-5.1-Reasoning-1M-Cleaned
Source: https://huggingface.co/datasets/indonesian-nlp/wikipedia-id
        https://huggingface.co/datasets/Jackrong/GLM-5.1-Reasoning-1M-Cleaned

Pipeline:
   1. Load from HuggingFace (2 sources)
   2. Dedup (exact)
   3. Length filter (min/max characters)
   4. Quality filter (character ratio, language purity/relaxed for EN)
   5. Repetition filter (n-gram repetition threshold)
   6. Combine → tokenize → train

Active Algorithms (set enabled=True/False in ALGO dict):
  ✔ Curriculum Learning       — easy→hard progression by text length
  ✔ Dynamic Sampling          — adjust mix based on per-category loss
  ✔ Difficulty Scheduling     — 4 phases: tokens→syntax→reasoning→expert
  ✔ MTP (Multi-Token Prediction) — auxiliary loss for richer signal
  ✔ MoE Balancing             — load-balanced routing with bias correction
  ✔ AutoTuner                 — AI yang otomatis mengatur cara belajar sendiri
  ✘ Loss-Based Curriculum     — adaptive difficulty with momentum
  ✘ Reflection Training       — store & re-train on high-loss examples
  ✘ Multi-Objective Training  — LM + MTP + reflection losses with dynamic weights
  ✘ Synthetic Evolution Loop  — generate synthetic data from model predictions
  ✘ Memory Algorithms         — experience replay with gradient-based scoring
  ✘ Tool Routing Intelligence — classify & route to 5 expert categories

Model: Deeplm (~105M params) — NLP Reasoning Focus
  Architecture: Decoder-only Transformer with MLA + MoE + Hyper-Connections + MTP
  MLA: Multi-head Latent Attention (DeepSeek V4 style)
  MoE: Mixture of Experts (4 routed + 1 shared)
  Hyper-Connections: Sinkhorn routing over residual connections
  MTP: Multi-Token Prediction (2 layers)
  Hybrid Attention: Softmax + Lightning blend

Features:
- Proper checkpoint: model + optimizer + step + loss
- Smart resume: auto-detect latest, load optimizer state
- Gradient clipping, loss EMA
- Real-time logs ke Modal dashboard
- GPU: A10G (24GB)
"""
import gc
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import deque
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import modal

VOL = "/vol"
volume = modal.Volume.from_name("deeplm-checkpoints", create_if_missing=True)

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.1-cudnn9-devel")
    .run_commands("apt-get update -qq && apt-get install -y -qq git wget xz-utils")
    .pip_install("datasets>=4.0.0", "tokenizers", "tqdm", "huggingface-hub", "einops", "seacrowd", "PyYAML")  # BUILD_ID=v3
)

app = modal.App("deeplm-train", image=image)

_CODE_VERSION = "ee12bdb"

# Pass HF_TOKEN from local env to container
_hf = os.environ.get("HF_TOKEN", "")
func_secrets = [modal.Secret.from_dict({"HF_TOKEN": _hf})] if _hf else []

# ── Algorithm Configuration (global, editable) ─────────────────────
ALGO = {
    "curriculum": True,
    "dynamic_sampling": True,
    "difficulty_scheduling": True,
    "mtp": True,
    "moe_balancing": True,
    "auto_tuner": True,
    "reflection": True,
    "synthetic_evolution": False,
    "memory_algorithms": True,
    "tool_routing": True,
    "loss_based_curriculum": True,
    "multi_objective": True,
}


@app.function(cpu=2, memory=4096, timeout=120)
def debug_glm():
    """Quick debug: inspect GLM dataset structure on Modal."""
    from datasets import load_dataset
    print("Loading GLM dataset...", flush=True)
    ds = load_dataset("Jackrong/GLM-5.1-Reasoning-1M-Cleaned", name="main", split="train", streaming=True)
    for i, item in enumerate(ds):
        if i >= 3:
            break
        print(f"\n[item {i}] type={type(item).__name__} is_dict={isinstance(item, dict)}", flush=True)
        if isinstance(item, dict):
            for k, v in item.items():
                if isinstance(v, str):
                    print(f"  {k}: str len={len(v)}", flush=True)
                elif isinstance(v, list):
                    print(f"  {k}: list len={len(v)}", flush=True)
                    if v:
                        print(f"    [0] type={type(v[0]).__name__} keys={list(v[0].keys()) if isinstance(v[0], dict) else 'N/A'}", flush=True)
                elif isinstance(v, dict):
                    print(f"  {k}: dict keys={list(v.keys())}", flush=True)
                else:
                    print(f"  {k}: type={type(v).__name__} val={v}", flush=True)
        else:
            print(f"  str={str(item)[:200]}", flush=True)
    print("\nDone!", flush=True)

@app.function(gpu="A10G", cpu=8, memory=32768, timeout=3600*24, volumes={VOL: volume}, secrets=func_secrets)
def train(
    max_steps: int = 50000,
    batch_size: int = 8,
    grad_accum: int = 4,
    max_seq_length: int = 2048,
    lr: float = 3e-4,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.1,
    max_grad_norm: float = 1.0,
    save_steps: int = 5000,
    log_steps: int = 50,
    eval_steps: int = 5000,
    compile: bool = True,
    lr_scheduler: str = "cosine",
    bf16: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
    min_length: int = 50,
    max_length: int = 8000,
    max_repetition_ratio: float = 0.4,
    min_char_ratio: float = 0.25,
    hf_repo: str = "samcheng0/deeplm-108m",
    force_rebuild: bool = True,
):

    print(f"  Code version: {_CODE_VERSION} | force_rebuild={force_rebuild}", flush=True)

    # ── Setup path ──────────────────────────────────────────────────
    repo = Path(VOL) / "deeplm"
    if not repo.exists():
        print("Cloning deeplm repo...", flush=True)
        subprocess.run(["git", "clone", "--depth=1",
            "https://github.com/Luvion1/deeplm.git", str(repo)], check=True)
    else:
        print("Pulling latest code from GitHub...", flush=True)
        result = subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"],
                                capture_output=True, text=True)
        if result.returncode != 0:
            print("  Pull failed, re-cloning...", flush=True)
            subprocess.run(["rm", "-rf", str(repo)], check=True)
            subprocess.run(["git", "clone", "--depth=1",
                "https://github.com/Luvion1/deeplm.git", str(repo)], check=True)
    sys.path = [p for p in sys.path if "deeplm" not in p]
    sys.path.insert(0, str(repo))
    os.environ["PYTHONUNBUFFERED"] = "1"

    import torch
    torch.set_float32_matmul_precision("high")

    if seed is not None:
        import random, numpy as np
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    from torch.optim import SGD
    from torch.optim.lr_scheduler import LambdaLR
    from torch.utils.data import DataLoader
    from tokenizers import Tokenizer
    from deeplm.config import DeeplmConfig
    from deeplm.model.deeplm import DeeplmModel
    from deeplm.training.auto_tuner import AutoTuner

    # ── Logging ─────────────────────────────────────────────────────
    log_dir = Path(VOL) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_txt = log_dir / "train.log"

    class _C:
        END = "\033[0m"; BOLD = "\033[1m"; DIM = "\033[2m"
        GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"
        CYAN = "\033[96m"; MAGENTA = "\033[95m"
        _gpu_cache = "N/A"

    _ansi_pat = re.compile(r"\033\[[0-9;]*m")
    _last_gpu_log = [0.0]

    def _gpu_str():
        now = time.time()
        if now - _last_gpu_log[0] > 1:
            mem = torch.cuda.memory_allocated() / 1e9
            tot = torch.cuda.get_device_properties(0).total_memory / 1e9
            _last_gpu_log[0] = now
            _C._gpu_cache = f"{mem:.1f}/{tot:.0f}GB ({mem/tot*100:.0f}%)"
        return _C._gpu_cache

    def log(msg, end="\n", color=""):
        ts = datetime.now().strftime("%H:%M:%S")
        prefix = f"{_C.DIM}{ts}{_C.END} {_C.DIM}{_gpu_str()}{_C.END} "
        colored = f"{prefix}{color}{msg}{_C.END}" if color else f"{prefix}{msg}"
        print(colored, end=end, flush=True)
        with open(log_txt, "a") as f:
            f.write(_ansi_pat.sub("", f"{ts} {_gpu_str()} {msg}") + ("\n" if end == "\n" else end))

    # ── Tokenizer ───────────────────────────────────────────────────
    tok_path = Path(VOL) / "tokenizer"
    if not (tok_path / "tokenizer.json").exists():
        log(f"Tokenizer not found at /vol/tokenizer — downloading from HF...")
        tok_path.mkdir(parents=True, exist_ok=True)
        try:
            from huggingface_hub import hf_hub_download
            for fname in ["tokenizer.json", "tokenizer_config.json"]:
                hf_hub_download(repo_id=hf_repo, filename=fname, local_dir=str(tok_path))
            log(f"  Downloaded tokenizer from {hf_repo}")
        except Exception as e:
            log(f"{_C.RED}ERROR: Cannot fetch tokenizer from HF: {e}{_C.END}")
            return
    tokenizer = Tokenizer.from_file(str(tok_path / "tokenizer.json"))
    log(f"Tokenizer: {tokenizer.get_vocab_size():,}")

    # ── Device & Model ──────────────────────────────────────────────
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0).replace("NVIDIA ", "")
    gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    log(f"GPU: {gpu_name} ({gpu_gb:.1f}GB)")

    config = DeeplmConfig()
    model = DeeplmModel(config)
    model.to(device)
    model.gradient_checkpointing_enable()
    total_params = model.num_parameters()

    # ── Model info ──────────────────────────────────────────────────
    log(f"{_C.BOLD}{'='*65}{_C.END}")
    log(f"{_C.BOLD}│  MODEL: Deeplm — NLP Reasoning{_C.END}")
    log(f"{_C.BOLD}{'='*65}{_C.END}")

    param_counts = {}
    for name, param in model.named_parameters():
        part = name.split(".")[0] if "." in name else name
        param_counts[part] = param_counts.get(part, 0) + param.numel()

    log(f"{_C.BOLD}Total Parameters: {total_params:,}{_C.END}")
    log(f"{_C.DIM}{'─'*65}{_C.END}")

    arch = config.architecture
    for label, val in [
        ("Architecture", "Decoder-only Transformer"),
        ("Hidden Size", f"{arch.hidden_size:,}"),
        ("Num Layers", str(arch.num_layers)),
        ("Intermediate", f"{arch.intermediate_size:,}"),
        ("Attention Heads", str(arch.num_attention_heads)),
        ("Head Dim", str(arch.head_dim)),
        ("RoPE Head Dim", str(arch.rope_head_dim)),
        ("Max Seq Length", f"{arch.max_seq_length:,}"),
        ("RoPE Theta", f"{arch.rope_theta:,.0f}"),
    ]:
        log(f"  {label:20s}: {val}")
    log(f"{_C.DIM}{'─'*65}{_C.END}")

    mla = config.mla
    kv_cache_size = mla.kv_lora_rank + mla.qk_rope_head_dim
    log(f"  {_C.CYAN}MLA{_C.END}")
    log(f"    Q LoRA: {mla.q_lora_rank} | KV LoRA: {mla.kv_lora_rank}")
    log(f"    KV Cache/Token: {kv_cache_size} dims | Compression: {arch.hidden_size/max(kv_cache_size,1):.1f}x")
    log(f"{_C.DIM}{'─'*65}{_C.END}")

    moe = config.moe
    log(f"  {_C.MAGENTA}MoE{_C.END}")
    log(f"    Routed: {moe.num_routed_experts} | Shared: {moe.num_shared_experts} | Top-K: {moe.top_k}")
    log(f"    Router: {moe.router.scoring_function} + bias load balancing")
    log(f"{_C.DIM}{'─'*65}{_C.END}")

    hc = config.hyper_connections
    log(f"  {_C.GREEN}Hyper-Connections{_C.END}")
    log(f"    Types: {', '.join(hc.connection_types)}")
    log(f"{_C.DIM}{'─'*65}{_C.END}")

    mtp = config.mtp
    log(f"  {_C.YELLOW}MTP{_C.END}")
    log(f"    Layers: {mtp.num_mtp_layers} | Depth: {mtp.mtp_depth} | Weight: {mtp.mtp_loss_weight}")
    log(f"{_C.DIM}{'─'*65}{_C.END}")

    ha = config.hybrid_attention
    log(f"  {_C.CYAN}Hybrid Attention{_C.END}")
    log(f"    Softmax: {ha.softmax_layers} | Linear: {ha.linear_layers}")
    log(f"{_C.DIM}{'─'*65}{_C.END}")

    log(f"  {_C.BOLD}Params:{_C.END}")
    for part, count in sorted(param_counts.items(), key=lambda x: -x[1]):
        pct = count / max(total_params, 1) * 100
        log(f"    {part:25s}: {count:>12,} ({pct:5.1f}%)")
    log(f"{_C.DIM}{'─'*65}{_C.END}")

    log(f"  {_C.BOLD}Algorithms:{_C.END}")
    for name, enabled in ALGO.items():
        log(f"    {'✔' if enabled else '✘'} {name}")
    log(f"{_C.BOLD}{'='*65}{_C.END}")

    if compile:
        torch._dynamo.config.suppress_errors = True
        log("Compiling model...")
        model = torch.compile(model, mode="default")
        log("  Compiled")

    # ── Checkpoints ─────────────────────────────────────────────────
    ckpt_dir = Path(VOL) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    start_step = 0
    best_loss = float("inf")
    ckpt_files = sorted(ckpt_dir.glob("ckpt-*.pt"))

    auto_tuner_state = None  # BUG 77 FIX: restore AutoTuner state on resume

    if ckpt_files:
        latest_step = max(int(f.stem.split("-")[1]) for f in ckpt_files)
        start_step = latest_step
        log(f"Resume: step {latest_step}")
        sd = torch.load(ckpt_dir / f"ckpt-{latest_step}.pt", map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=False)
        log(f"  Model: {len(sd)}/{len(model.state_dict())} keys loaded")
        del sd
        state_file = ckpt_dir / f"state-{latest_step}.json"
        if state_file.exists():
            with open(state_file) as f:
                state_data = json.load(f)
                best_loss = state_data.get("best_loss", float("inf"))
                auto_tuner_state = state_data.get("auto_tuner")
    else:
        log("No local checkpoints — trying HF fallback...")
        try:
            from huggingface_hub import hf_hub_download
            for fname in ["model.pt", "optimizer.pt", "training_state.json"]:
                try:
                    hf_hub_download(repo_id=hf_repo, filename=fname, local_dir=str(ckpt_dir))
                except Exception:
                    pass
            ckpt_src = ckpt_dir / "model.pt"
            opt_src = ckpt_dir / "optimizer.pt"
            state_src = ckpt_dir / "training_state.json"
            if ckpt_src.exists():
                step = 0
                if state_src.exists():
                    with open(state_src) as f:
                        sd = json.load(f)
                        step = sd.get("step", 0)
                        best_loss = sd.get("best_loss", float("inf"))
                        auto_tuner_state = sd.get("auto_tuner")
                if step == 0:
                    step = 8000
                ckpt_src.rename(ckpt_dir / f"ckpt-{step}.pt")
                if opt_src.exists():
                    opt_src.rename(ckpt_dir / f"opt-{step}.pt")
                if state_src.exists():
                    state_src.rename(ckpt_dir / f"state-{step}.json")
                start_step = step
                log(f"  HF fallback: resumed from step {step}")
                ckpt_files = sorted(ckpt_dir.glob("ckpt-*.pt"))
                if ckpt_files:
                    sd = torch.load(ckpt_dir / f"ckpt-{step}.pt", map_location=device, weights_only=True)
                    model.load_state_dict(sd, strict=False)
                    log(f"  Model loaded from HF fallback")
                    del sd
            else:
                log("  No checkpoints on HF either — fresh start")
        except Exception as e:
            log(f"  HF fallback failed: {e} — fresh start")

    # ── Multi-Source Dataset Pipeline ─────────────────────────────────────
    log(f"{_C.BOLD}{'='*65}{_C.END}")
    log(f"{_C.BOLD}│  DATASET PIPELINE (3 sources){_C.END}")
    log(f"{_C.BOLD}{'='*65}{_C.END}")

    data_dir = Path(VOL) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    combined_file = data_dir / "corpus.jsonl"

    DATASETS = [
        {"name": "indonesian-nlp/wikipedia-id", "split": "train", "lang": "id", "max_samples": 5_000_000},
        {"name": "Jackrong/GLM-5.1-Reasoning-1M-Cleaned", "config": "main", "split": "train", "lang": "en", "max_samples": 500_000},
        {"name": "wikimedia/wikipedia", "config": "20231101.en", "split": "train", "lang": "en", "max_samples": 1_000_000},
    ]

    from deeplm.training.data_pipeline import (
        StrictFilter, TokenCache, BucketDataset, WeightedBucketSampler,
        build_pipeline,
    )

    log(f"  [PIPELINE] force_rebuild={force_rebuild} combined_exists={combined_file.exists()}")
    if force_rebuild and combined_file.exists():
        combined_file.unlink()
        log(f"  Force rebuild: removed old corpus")
    if not combined_file.exists():
        from datasets import load_dataset
        all_corpus_lines = []

        for ds_cfg in DATASETS:
            ds_name = ds_cfg["name"]
            ds_split = ds_cfg["split"]
            ds_lang = ds_cfg["lang"]
            ds_max = ds_cfg["max_samples"]
            ds_config = ds_cfg.get("config", None)
            log(f"{_C.BOLD}── {ds_name} ({ds_lang}, max {ds_max:,}) ──{_C.END}")

            # [1] Load
            log("  [1/5] Loading...")
            try:
                kwargs = {"split": ds_split, "streaming": True}
                if ds_config:
                    kwargs["name"] = ds_config
                ds = load_dataset(ds_name, **kwargs)
            except Exception as e:
                log(f"  {_C.RED}ERROR loading {ds_name}: {e}{_C.END}")
                log(f"  → Corpus total: {len(all_corpus_lines):,}\n")
                continue
            lines = []
            debug_null, debug_conv, debug_input = 0, 0, 0
            _item_count = 0
            for i, item in enumerate(ds):
                if i >= ds_max:
                    break
                _item_count += 1
                if i == 0 and ds_name != "indonesian-nlp/wikipedia-id":
                    log(f"  [DBG] item type={type(item).__name__} is_dict={isinstance(item, dict)} keys={sorted(item.keys()) if hasattr(item,'keys') else 'N/A'}")
                    if isinstance(item, dict):
                        for _k in item:
                            _v = item[_k]
                            log(f"  [DBG]   {_k}: type={type(_v).__name__} val={str(_v)[:120]}")
                    else:
                        log(f"  [DBG]   str={str(item)[:200]}")

                try:
                    text = ""
                    if isinstance(item, dict) and "conversations" in item:
                        convs = item["conversations"]
                        if isinstance(convs, list):
                            vals = [c.get("value", "") for c in convs if isinstance(c, dict)]
                            text = "\n".join(v for v in vals if v)
                        if text:
                            debug_conv += 1
                    if not text and isinstance(item, dict):
                        inp = item.get("input") or ""
                        out = item.get("output") or ""
                        if inp and out:
                            text = f"{inp}\n{out}"
                            debug_input += 1
                        else:
                            text = (item.get("text") if isinstance(item, dict) else "") or ""
                            if isinstance(item, dict):
                                text = text or item.get("content", "") or item.get("question", "") or ""
                    if not text:
                        debug_null += 1
                    if isinstance(text, list):
                        text = " ".join(str(t) for t in text)
                    text = text.strip()
                    if text:
                        lines.append(text)
                except Exception as e:
                    debug_null += 1
                if (i + 1) % 200_000 == 0:
                    log(f"    Read {i+1:,}...")
            log(f"  Total: {len(lines):,} lines (conv={debug_conv}, inp={debug_input}, null={debug_null}) seen={_item_count}")
            del ds; gc.collect()

            if not lines:
                log(f"  → Corpus total: {len(all_corpus_lines):,}\n")
                continue

            # [2] Dedup (exact)
            log("  [2/5] Dedup...")
            seen = set()
            deduped = []
            for line in lines:
                if line not in seen:
                    seen.add(line)
                    deduped.append(line)
            log(f"  Removed {len(lines)-len(deduped):,} dupes → {len(deduped):,} unique")
            del seen, lines; gc.collect()

            # [3] Strict multi-stage filter (replaces old length + quality + rep)
            log(f"  [3/5] StrictFilter ({ds_lang})...")
            filter_cfg = {
                "min_length": min_length, "max_length": max_length,
                "min_char_ratio": min_char_ratio,
                "max_repetition_ratio": max_repetition_ratio,
                "min_lang_score": 0.001,  # tiny — catches pure noise
                "min_words": 10,
            }
            sf = StrictFilter(filter_cfg)
            filtered = sf.filter(deduped, lang=ds_lang)
            log(f"  {sf.summary()}")
            del deduped; gc.collect()

            all_corpus_lines.extend(filtered)
            log(f"  → Corpus total: {len(all_corpus_lines):,}\n")
            del filtered; gc.collect()

        # [5] Save combined corpus
        log(f"{_C.BOLD}── Saving combined corpus ({len(all_corpus_lines):,} lines) ──{_C.END}")
        with open(combined_file, "w", encoding="utf-8") as f:
            for line in all_corpus_lines:
                f.write(json.dumps({"text": line}, ensure_ascii=False) + "\n")
        final_count = len(all_corpus_lines)
        del all_corpus_lines; gc.collect()
        log(f"  Saved to {combined_file}")
    else:
        log("  Using pre-processed corpus")
        final_count = sum(1 for _ in open(combined_file))
        log(f"  {final_count:,} lines from {len(DATASETS)} sources")

    log(f"{_C.DIM}{'─'*65}{_C.END}")

    # ── Load texts ──────────────────────────────────────────────────
    log("  Loading texts from combined corpus...")
    all_texts = []
    with open(combined_file, encoding="utf-8") as f:
        for line in f:
            all_texts.append(json.loads(line)["text"])
    log(f"  {len(all_texts):,} texts loaded")

    # ── Categorize ──
    log("  Categorizing...")
    categories = {"short": [], "medium": [], "long": [], "very_long": []}
    for text in all_texts:
        ln = len(text)
        if ln < 100:
            categories["short"].append(text)
        elif ln < 500:
            categories["medium"].append(text)
        elif ln < 2000:
            categories["long"].append(text)
        else:
            categories["very_long"].append(text)

    for k, v in categories.items():
        log(f"    {k:12s}: {len(v):>10,}")
    total_size = sum(len(v) for v in categories.values())
    log(f"    {'TOTAL':12s}: {total_size:>10,}")

    if total_size == 0:
        log(f"{_C.RED}ERROR: No valid samples. Check filter settings.{_C.END}")
        return

    # ── Train/eval split by stratified category ──
    log("  Splitting train/eval (stratified by category)...")
    rng = torch.Generator().manual_seed(seed)
    train_texts, eval_texts = [], []
    for cat_name, cat_list in categories.items():
        n_cat = len(cat_list)
        n_eval = max(1, min(500, n_cat // 10))
        perm = torch.randperm(n_cat, generator=rng).tolist()
        eval_texts.extend([cat_list[i] for i in perm[:n_eval]])
        train_texts.extend([cat_list[i] for i in perm[n_eval:]])
    # Cap eval at 5000
    if len(eval_texts) > 5000:
        eval_texts = eval_texts[:5000]
    log(f"  Train: {len(train_texts):,} | Eval: {len(eval_texts):,}")

    # ── Sort train by length (curriculum baseline) ──
    log("  Sorting train by length (easy → hard)...")
    train_texts.sort(key=len)

    bos = tokenizer.token_to_id("<|begin_of_sentence|>")
    eos = tokenizer.token_to_id("<|end_of_sentence|>")
    pad = tokenizer.token_to_id("<|pad|>")

    # ── Token cache + bucket dataset ──
    log("  Building BucketDataset with TokenCache...")
    token_cache = TokenCache(
        cache_dir=str(data_dir / "tokencache"),
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        bos=bos, eos=eos, pad=pad,
    )
    bucket_size = batch_size * 8  # 64 for batch 8
    train_dataset = BucketDataset(train_texts, token_cache, bucket_size=bucket_size)
    log(f"  {len(train_dataset):,} samples in {len(train_dataset.get_buckets()):,} buckets")

    # Pre-tokenize (optional — makes first epoch faster)
    log("  Pre-tokenizing (background)...")
    n_tok = train_dataset.pre_tokenize()
    log(f"  Pre-tokenized {n_tok:,} texts into {token_cache.cache_dir}")

    # ── Eval dataset (simple, no bucket) ──
    eval_dataset = BucketDataset(eval_texts, token_cache, bucket_size=bucket_size)

    # ── Sampler with category weights (from DynamicSampler) ──
    # Build per-bucket category mapping
    bucket_categories = {}
    for flat_idx in range(len(train_dataset)):
        info = train_dataset._text_map.get(flat_idx)
        if info is not None:
            # Map through original text to find category
            t = info["text"]
            ln = len(t)
            cat = "short" if ln < 100 else ("medium" if ln < 500 else ("long" if ln < 2000 else "very_long"))
            bucket_categories.setdefault(info["bucket"], cat)

    bucket_counts = train_dataset.get_buckets()

    _nw = min(num_workers or 8, os.cpu_count() or 1)
    # Use weighted sampler if dynamic_sampler exists (set below), else uniform bucket sampler
    _use_weighted = ALGO["dynamic_sampling"]
    if _use_weighted:
        # All buckets equal weight initially — DynamicSampler.update_mix adjusts during training
        cat_weights = {c: 1.0 for c in ["short", "medium", "long", "very_long"]}
        sampler_obj = WeightedBucketSampler(
            bucket_counts=bucket_counts,
            bucket_ids={i: info["bucket"] for i, info in train_dataset._text_map.items()},
            bucket_categories=bucket_categories,
            category_weights=cat_weights,
            epoch_size=len(train_dataset),
            seed=seed,
        )
    else:
        sampler_obj = None  # default sequential

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler_obj,
        num_workers=_nw,
        pin_memory=_nw > 0,
        drop_last=True,
        prefetch_factor=4,
        persistent_workers=_nw > 0,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size,
        num_workers=0, pin_memory=False,
    )
    log(f"  ✓ Pipeline: BucketDataset({bucket_size}) + TokenCache + {'WeightedSampler' if _use_weighted else 'Sequential'} | Tok/s pre-tokenized")

    # ── Optimizer ───────────────────────────────────────────────────
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    total = max_steps
    warmup = int(total * warmup_ratio)

    def lr_fn(s):
        if s < warmup:
            return s / max(1, warmup)
        p = min((s - warmup) / max(1, total - warmup), 1.0)
        if lr_scheduler == "linear":
            return max(1e-5 / lr, 1.0 - p)
        if lr_scheduler == "constant":
            return 1.0
        return max(1e-5 / lr, 0.5 * (1.0 + math.cos(math.pi * p)))

    scheduler = LambdaLR(optimizer, lr_fn)

    opt_file = ckpt_dir / f"opt-{start_step}.pt" if start_step > 0 else None
    if opt_file and opt_file.exists():
        try:
            optimizer.load_state_dict(torch.load(opt_file, map_location="cpu", weights_only=True))
            for _ in range(start_step):
                scheduler.step()
            log("  Optimizer resumed")
        except (ValueError, RuntimeError) as e:
            log(f"  Optimizer mismatch: {e}")

    # ── Algorithms ──────────────────────────────────────────────────
    log(f"{_C.BOLD}Initializing algorithms...{_C.END}")

    # 1. Curriculum
    curriculum = None
    if ALGO["curriculum"]:
        class CurriculumScheduler:
            def __init__(self, steps, tiers=4):
                self.steps = steps; self.tiers = tiers
                self.size = steps // tiers
                self.names = ["easy", "medium", "hard", "expert"]
            def get_tier(self, step):
                idx = min(step // self.size, self.tiers - 1)
                return self.names[idx], idx
        curriculum = CurriculumScheduler(total)
        log(f"  ✔ Curriculum: {curriculum.tiers} tiers")

    # 2. Dynamic Sampling
    dynamic_sampler = None
    if ALGO["dynamic_sampling"]:
        class DynamicSampler:
            def __init__(self, cats, base, alpha=0.05):
                self.cats = list(cats.keys())
                self.base = base.copy(); self.mix = base.copy()
                self.ema = {c: 0.0 for c in self.cats}
                self.alpha = alpha
            def record_loss(self, cat, loss):
                self.ema[cat] = self.alpha * loss + (1 - self.alpha) * self.ema[cat]
            def update_mix(self, step, interval=500):
                if step % interval != 0:
                    return
                tot = sum(self.ema.values()) + 1e-8
                w = {c: self.ema[c] / tot for c in self.cats}
                for c in self.cats:
                    self.mix[c] = 0.6 * self.base[c] + 0.4 * w[c]
                s = sum(self.mix.values())
                for c in self.mix:
                    self.mix[c] /= s
            def get_weights(self):
                return self.mix.copy()
            def stats(self):
                return {"mix": {c: f"{v:.3f}" for c, v in self.mix.items()},
                        "ema": {c: f"{v:.3f}" for c, v in self.ema.items()}}

        cat_sizes = {k: len(v) for k, v in categories.items()}
        base_mix = {k: v / total_size for k, v in cat_sizes.items()}
        dynamic_sampler = DynamicSampler(categories, base_mix)
        log(f"  ✔ Dynamic Sampling: {len(categories)} categories")

    # 3. Difficulty Scheduling
    difficulty_sched = None
    if ALGO["difficulty_scheduling"]:
        class DifficultyScheduler:
            def __init__(self, steps, tw=0.05):
                self.steps = steps; self.tw = tw
                self.phases = [
                    (0.0, 0.25, "Token Learning", {"short": 0.6, "medium": 0.4}),
                    (0.25, 0.50, "Syntax", {"medium": 0.5, "long": 0.3, "short": 0.2}),
                    (0.50, 0.75, "Reasoning", {"long": 0.5, "very_long": 0.3, "medium": 0.2}),
                    (0.75, 1.0, "Expert", {"very_long": 0.5, "long": 0.4, "medium": 0.1}),
                ]
            def get_phase(self, step):
                prog = step / max(self.steps, 1)
                for i, (s, e, name, focus) in enumerate(self.phases):
                    if s <= prog < e:
                        if prog < s + self.tw and i > 0:
                            pf = self.phases[i-1][3]
                            b = (prog - s) / self.tw
                            bl = {}
                            for c in set(list(pf.keys()) + list(focus.keys())):
                                bl[c] = (1-b)*pf.get(c,0) + b*focus.get(c,0)
                            return f"{self.phases[i-1][2]} → {name}", bl, b
                        if prog > e - self.tw and i < len(self.phases)-1:
                            nf = self.phases[i+1][3]
                            b = (prog - (e - self.tw)) / self.tw
                            bl = {}
                            for c in set(list(focus.keys()) + list(nf.keys())):
                                bl[c] = (1-b)*focus.get(c,0) + b*nf.get(c,0)
                            return f"{name} → {self.phases[i+1][2]}", bl, b
                        return name, focus, (prog - s) / max(e - s, 0.01)
                return self.phases[-1][0], self.phases[-1][1], 1.0
        difficulty_sched = DifficultyScheduler(total)
        log(f"  ✔ Difficulty: 4 phases")

    # 4-5. MTP + MoE (built-in)
    if ALGO["mtp"]:
        log(f"  ✔ MTP: {config.mtp.num_mtp_layers} layers, weight={config.mtp.mtp_loss_weight}")
    if ALGO["moe_balancing"]:
        log(f"  ✔ MoE Balancing: bias-based")

    # 6-11. Disabled algorithms (stubs)
    loss_curriculum = None
    reflection_memory = None
    multi_obj = None
    synthetic_evo = None
    exp_memory = None
    tool_router = None

    if ALGO["loss_based_curriculum"]:
        class LossBasedCurriculum:
            def __init__(self, target=5.0, mom=0.9):
                self.target = target; self.mom = mom
                self.difficulty = 0.5; self.prev = 0.5
                self.hist = deque(maxlen=200)
            def update(self, loss):
                self.hist.append(loss)
                if len(self.hist) < 10: return
                avg = sum(list(self.hist)[-20:]) / min(20, len(self.hist))
                if avg < self.target:
                    td = min(1.0, 0.5 + (self.target - avg) / self.target)
                elif avg > self.target * 1.5:
                    td = max(0.0, 0.5 - (avg - self.target*1.5) / self.target)
                else:
                    td = 0.5
                self.difficulty = self.mom * self.prev + (1 - self.mom) * td
                self.prev = self.difficulty
        loss_curriculum = LossBasedCurriculum()
        log(f"  ✔ Loss-Based Curriculum")

    if ALGO["reflection"]:
        class ReflectionMemory:
            def __init__(self, max=5000):
                self.max = max; self.memory = []; self.count = 0
            def add(self, text, loss, step, threshold=7.0):
                if loss > threshold:
                    self.memory.append((text, loss, step, 0))
                    if len(self.memory) > self.max:
                        self.memory.sort(key=lambda x: -x[1])
                        self.memory = self.memory[:self.max]
            def stats(self):
                if not self.memory: return {"size": 0}
                return {"size": len(self.memory),
                        "avg_loss": sum(l for _,l,_,_ in self.memory)/len(self.memory)}
        reflection_memory = ReflectionMemory()
        log(f"  ✔ Reflection")

    if ALGO["multi_objective"]:
        class MultiObjectiveTrainer:
            def __init__(self):
                self.lm_w = 1.0; self.mtp_w = 0.3; self.ref_w = 0.2
                self.hist = deque(maxlen=100)
            def update_weights(self, step, lm, mtp):
                self.hist.append(lm)
                if step > 0 and step % 500 == 0:
                    if lm < 5.0: self.mtp_w = min(0.5, self.mtp_w + 0.05)
                    if reflection_memory and reflection_memory.memory:
                        self.ref_w = min(0.3, self.ref_w + 0.02)
                    self.lm_w = max(0.5, 1.0 - (self.mtp_w-0.3) - (self.ref_w-0.2))
            def total_loss(self, lm, mtp=None, ref=None):
                t = self.lm_w * lm
                if mtp is not None: t += self.mtp_w * mtp
                if ref is not None: t += self.ref_w * ref
                return t
        multi_obj = MultiObjectiveTrainer()
        log(f"  ✔ Multi-Objective")

    if ALGO["synthetic_evolution"]:
        class SyntheticEvolution:
            def __init__(self, model, tokenizer, device, max_seq):
                self.model = model; self.tokenizer = tokenizer
                self.device = device; self.max_seq = max_seq
                self.texts = []; self.count = 0
                self.seeds = ["Judul: Bahasa Indonesia adalah", "def solve(", "Question: Berapakah"]
            def generate_from_seed(self, seed=None, max_new=128, temp=0.8):
                if seed is None: import random; seed = random.choice(self.seeds)
                try:
                    enc = self.tokenizer.encode(seed)
                    ids = torch.tensor([[self.tokenizer.token_to_id("<|begin_of_sentence|>")] + enc.ids[:256]],
                                       dtype=torch.long, device=self.device)
                    with torch.no_grad():
                        out = self.model.generate(ids, max_new_tokens=max_new, temperature=temp, top_k=40, top_p=0.9, do_sample=True)
                    gen = self.tokenizer.decode(out[0].tolist(), skip_special_tokens=True)
                    if len(gen) > 50:
                        self.texts.append(gen); self.count += 1; return gen
                except Exception: pass
                return None
            def stats(self):
                return {"generated": self.count, "stored": len(self.texts)}
        synthetic_evo = SyntheticEvolution(model, tokenizer, device, max_seq_length)
        log(f"  ✔ Synthetic Evolution")

    if ALGO["memory_algorithms"]:
        class ExperienceMemory:
            def __init__(self, cap=10000):
                self.cap = cap; self.exp = []; self.scores = {}
            def store(self, tid, loss, gn, step, cat="unknown"):
                imp = loss * (1.0 + min(gn, 10.0) / 10.0)
                self.exp.append((tid, loss, gn, step, cat))
                self.scores[tid] = imp
                if len(self.exp) > self.cap:
                    self.exp.sort(key=lambda x: -self.scores.get(x[0], 0))
                    self.exp = self.exp[:self.cap]
            def stats(self):
                if not self.exp: return {"stored": 0}
                return {"stored": len(self.exp),
                        "avg_loss": sum(e[1] for e in self.exp)/len(self.exp)}
        exp_memory = ExperienceMemory()
        log(f"  ✔ Memory Algorithms")

    if ALGO["tool_routing"]:
        class ToolRouter:
            def __init__(self):
                self.stats_dict = {"code":0,"math":0,"formal":0,"creative":0,"dialog":0}
                self.perf = {k: [] for k in self.stats_dict}
            def classify(self, text):
                lo = text.lower()
                scores = {
                    "code": sum(1 for w in ["def ","class ","import ","```"] if w in lo),
                    "math": sum(1 for w in ["=","+","-","*","rumus","hasil"] if w in lo),
                    "formal": sum(1 for w in ["yang","adalah","merupakan","dalam"] if w in lo),
                    "creative": sum(1 for w in ["cerita","novel","puisi"] if w in lo),
                    "dialog": sum(1 for w in ["user:","assistant:","?"] if w in lo),
                }
                return max(scores, key=scores.get), max(scores.values()) / max(sum(scores.values()), 1e-8)
            def record_performance(self, cat, loss):
                self.perf[cat].append(loss)
                if len(self.perf[cat]) > 100: self.perf[cat] = self.perf[cat][-100:]
                self.stats_dict[cat] += 1
            def stats(self):
                return {"routing": self.stats_dict,
                        "performance": {k: f"{sum(v)/max(len(v),1):.3f}" for k,v in self.perf.items()}}
        tool_router = ToolRouter()
        log(f"  ✔ Tool Routing")

    # 12. AutoTuner
    auto_tuner = None
    if ALGO["auto_tuner"]:
        warmup_steps = int(total * warmup_ratio)
        auto_tuner = AutoTuner(lr, warmup_steps, total, max_grad_norm)
        if auto_tuner_state:
            auto_tuner.restore_state(auto_tuner_state)
            log(f"  Restored AutoTuner: LR×{auto_tuner.lr_mult:.2f}, GradClip×{auto_tuner.gn_mult:.2f}, best={auto_tuner.best:.4f}")
        log(f"  ✔ AutoTuner: self-adaptive learning")

    log(f"{_C.DIM}{'─'*70}{_C.END}")

    eff = batch_size * grad_accum
    log(f"{_C.BOLD}┌{'─'*63}┐{_C.END}")
    log(f"{_C.BOLD}│{_C.END} {_C.CYAN}Deeplm{_C.END} — A10G — {_C.BOLD}{total_params:,}{_C.END} params — NLP Reasoning{' ' * max(0, 20 - len(f'{total_params:,}'))}{_C.BOLD}│{_C.END}")
    log(f"{_C.BOLD}│{_C.END} {'':<61} {_C.BOLD}│{_C.END}")
    log(f"{_C.BOLD}│{_C.END} {total:,} steps │ batch {batch_size}×{grad_accum}={eff} eff │ lr {lr} │ wd {weight_decay}")
    log(f"{_C.BOLD}│{_C.END} Dataset: Wikipedia-id ({final_count:,} lines)")
    log(f"{_C.BOLD}│{_C.END} Pipeline: dedup → length → quality → repetition")
    log(f"{_C.BOLD}│{_C.END} 10 layers | 512d | 8 heads | 4 MoE | rope 50k")
    log(f"{_C.BOLD}│{_C.END} MLA + Hybrid + HyperConn + MTP | SGD+momentum")
    log(f"{_C.BOLD}│{_C.END} Eval: every {eval_steps} steps | {'BF16' if bf16 else 'FP32'} | {lr_scheduler}")
    log(f"{_C.BOLD}│{_C.END} Logs: https://modal.com/apps/lunarchipter/main")
    log(f"{_C.BOLD}└{'─'*63}┘{_C.END}")
    log(f"{_C.DIM}{'─'*70}{_C.END}")
    hdr = f"  {_C.BOLD}{'Step':>7} | {'Loss':>9} | {'μLoss':>9} | {'LR':>10} | {'Grad':>7} | {'GPU':>12} | {'Tok/s':>8} | {'ETA':>6}{_C.END}"
    log(hdr)
    log(f"{_C.DIM}{'─'*70}{_C.END}")

    # ── Category tracking via length (not index, since dataset is sorted) ──
    def get_cat_from_length(text_len):
        if text_len < 100:
            return "short"
        elif text_len < 500:
            return "medium"
        elif text_len < 2000:
            return "long"
        else:
            return "very_long"

    # ── Training Loop ───────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()
    step = start_step
    t0 = time.time()
    accum = 0.0
    ema_loss = None
    losses = deque(maxlen=100)
    micro_step = 0
    sample_idx = 0
    tokens_processed = 0  # BUG 101 FIX: track actual non-pad tokens

    for epoch in range(999):
        first_batch = True
        for batch in train_loader:
            if first_batch:
                if difficulty_sched is not None:
                    pn, fc, _ = difficulty_sched.get_phase(step)
                    log(f"  {_C.BOLD}Phase: {pn}{_C.END} ({', '.join(fc)})")
                first_batch = False

            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            tokens_processed += attention_mask.sum().item()

            with torch.amp.autocast("cuda", dtype=torch.bfloat16) if bf16 else nullcontext():
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_mtp_loss=True)
                lm_loss = out["loss"] / grad_accum
                mtp_loss_val = out.get("mtp_loss")

            # Loss computation
            if multi_obj is not None:
                total_loss = multi_obj.lm_w * lm_loss
                if mtp_loss_val is not None:
                    total_loss = total_loss + multi_obj.mtp_w * mtp_loss_val / grad_accum
            else:
                total_loss = lm_loss
                if mtp_loss_val is not None and ALGO["mtp"]:
                    total_loss = total_loss + mtp_loss_val / grad_accum
            total_loss.backward()

            micro_step += 1
            accum += lm_loss.item()
            losses.append(lm_loss.item())

            # Algorithm hooks (per-batch)
            if dynamic_sampler is not None:
                # Use actual text length from attention mask, not index
                seq_len = attention_mask.sum().item() / batch_size
                cat = get_cat_from_length(seq_len)
                dynamic_sampler.record_loss(cat, lm_loss.item())
            sample_idx += batch_size

            if reflection_memory is not None and lm_loss.item() > 7.0:
                bt = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)[:200]
                reflection_memory.add(bt, lm_loss.item(), step)

            if exp_memory is not None:
                exp_memory.store(sample_idx, lm_loss.item(), 0.0, step)

            if tool_router is not None:
                bt = tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True)[:200]
                cat, _ = tool_router.classify(bt)
                tool_router.record_performance(cat, lm_loss.item())

            # Gradient step
            if micro_step % grad_accum == 0:
                eff_gn = max_grad_norm
                if auto_tuner is not None:
                    eff_gn = auto_tuner.effective_gn()

                # BUG 92 FIX: compute pre-clip grad norm for AutoTuner
                pre_clip_gn = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
                pre_clip_val = pre_clip_gn.item() if torch.is_tensor(pre_clip_gn) else float(pre_clip_gn)
                grad_norm_val = pre_clip_val
                # Capture unclipped gradients for AutoTuner analysis (every 10 steps)
                if auto_tuner is not None and step % 10 == 0:
                    auto_tuner.capture_gradients(model)
                # Now clip for real
                if eff_gn < float("inf"):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), eff_gn)
                optimizer.step()
                # Capture parameter updates (after step, before zero_grad, every 10 steps)
                if auto_tuner is not None and step % 10 == 0:
                    auto_tuner.capture_update(model)
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                ema_loss = losses[-1] if ema_loss is None else 0.9 * ema_loss + 0.1 * losses[-1]

                # AutoTuner (raw loss, not ema_loss — avoids double-EMA lag)
                if auto_tuner is not None:
                    auto_tuner.record(step, losses[-1], grad_norm_val)
                    adj = auto_tuner.get_adjustments(step)
                    # Apply AutoTuner multipliers to optimizer (after get_adjustments so lr_mult is fresh)
                    for pg in optimizer.param_groups:
                        pg["lr"] = scheduler.get_last_lr()[0] * auto_tuner.lr_mult
                        pg["weight_decay"] = weight_decay * auto_tuner.wd_mult
                    # BUG 68 FIX: always sync momentum, not just on adj
                    em = auto_tuner.effective_mom()
                    for pg in optimizer.param_groups:
                        pg["momentum"] = em
                    if adj:
                        log(f"  {_C.YELLOW}⟳ AutoTuner: {adj.get('action', 'adj')} | "
                            f"State: {auto_tuner.state} | "
                            f"LR×{auto_tuner.lr_mult:.2f} | "
                            f"GradClip×{auto_tuner.gn_mult:.2f}{_C.END}")

                # Other algorithm updates
                if loss_curriculum is not None:
                    loss_curriculum.update(ema_loss)
                if multi_obj is not None:
                    multi_obj.update_weights(step, ema_loss, mtp_loss_val.item() if mtp_loss_val is not None else None)
                if dynamic_sampler is not None:
                    dynamic_sampler.update_mix(step, interval=500)
                    # Sync weights to WeightedBucketSampler at same interval
                    if _use_weighted and sampler_obj is not None and step % 500 == 0:
                        sampler_obj.category_weights = dynamic_sampler.get_weights()
                if synthetic_evo is not None and step % 2000 == 0 and step > 0:
                    for seed in ["Judul: Bahasa Indonesia", "def solve(", "Question: Berapakah"]:
                        synthetic_evo.generate_from_seed(seed, max_new_tokens=64)
                    if synthetic_evo.texts:
                        log(f"  {_C.MAGENTA}⟳{_C.END} Synthetic: {synthetic_evo.count} generated")

                # Logging
                if step % log_steps == 0:
                    avg_loss = accum / log_steps
                    lr_now = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    remain = total - step
                    eta = remain * elapsed / max(1, step - start_step) if step > start_step else 0
                    tok_s = tokens_processed / max(1, elapsed)

                    ct = curriculum.get_tier(step)[0] if curriculum is not None else "N/A"
                    dp = difficulty_sched.get_phase(step)[0] if difficulty_sched is not None else "N/A"

                    if auto_tuner is not None:
                        elr = auto_tuner.effective_lr(lr_now)
                        lr_disp = f"{elr:>10.2e}"
                    else:
                        lr_disp = f"{lr_now:>10.2e}"

                    log(f"  {step:>7,} | {avg_loss:>9.4f} | {ema_loss:>9.4f} | {lr_disp} | {grad_norm_val:>7.4f} | {_gpu_str():>12} | {tok_s:>8,.0f} | {eta:>6.0f}s", color=_C.GREEN)

                    if step % (log_steps * 5) == 0:
                        auto_state = f"{auto_tuner.phase}/{auto_tuner.state}" if auto_tuner is not None else "N/A"
                        info = f"    Tier: {ct} | Phase: {dp} | TState: {auto_state}"
                        if auto_tuner is not None:
                            cos = f"cos={auto_tuner.cos_sim_ema:.2f}" if auto_tuner.cos_sim_init else "cos=N/A"
                            ur = f"ur={auto_tuner.update_ratio_ema:.5f}" if auto_tuner.ur_init else "ur=N/A"
                            ns = f"noise={auto_tuner.gn_noise_ema:.2f}" if auto_tuner.gn_noise_init else "noise=N/A"
                            info += f" | LR×{auto_tuner.lr_mult:.2f} clip×{auto_tuner.gn_mult:.2f} | {cos} {ur} {ns}"
                            # v8: Strategic Training Director
                            if auto_tuner.diagnosis != "unknown":
                                info += f"\n    ⟐ Diag: {auto_tuner.diagnosis} | Plan: {auto_tuner.plan.get('strategy', 'none')} | Acc: {auto_tuner.plan_accuracy:.2f}"
                                if auto_tuner.model_needs:
                                    needs = ", ".join(auto_tuner.model_needs[:3])
                                    info += f"\n    ⟐ Needs: {needs}"
                            if auto_tuner.traj_fit:
                                info += f" | Traj: slope={auto_tuner.traj_fit['slope']:.3f} r²={auto_tuner.traj_fit['r2']:.2f}"
                            if auto_tuner.goals.get("eta_to_target"):
                                info += f" | ETA: step {auto_tuner.goals['eta_to_target']:,}"
                            milestones_reached = sum(1 for _, r in auto_tuner.goals["milestones"] if r is not None)
                            info += f" | Milestones: {milestones_reached}/{len(auto_tuner.goals['milestones'])}"
                        if dynamic_sampler is not None:
                            info += f" | Mix: {dynamic_sampler.get_weights()}"
                        if reflection_memory is not None:
                            info += f" | Reflection: {len(reflection_memory.memory)}"
                        if synthetic_evo is not None:
                            info += f" | Synthetic: {synthetic_evo.count}"
                        if exp_memory is not None:
                            info += f" | Memory: {exp_memory.stats()['stored']}"
                        if tool_router is not None:
                            info += f" | Routing: {tool_router.stats()['routing']}"
                        log(info)
                    accum = 0.0

                # Eval
                if step % eval_steps == 0:
                    model.eval()
                    ev_loss = 0.0; ev_count = 0
                    with torch.no_grad():
                        for eb in eval_loader:
                            ei, ea, el = [x.to(device) for x in eb]
                            eo = model(input_ids=ei, attention_mask=ea, labels=el, output_mtp_loss=True)
                            ev_loss += eo["loss"].item()
                            ev_count += 1
                    ev_avg = ev_loss / max(1, ev_count)
                    if auto_tuner is not None:
                        auto_tuner.capture_eval(step, ev_avg)
                    model.train()
                    bs = f" {_C.GREEN}★{_C.END}" if ev_avg < best_loss else ""
                    log(f"  {_C.CYAN}▸{_C.END} Eval: loss={ev_avg:.4f}{bs}")

                    if auto_tuner is not None:
                        ts = auto_tuner.stats()
                        log(f"    AutoTuner: {ts['phase']}/{ts['state']} | LR×{ts['lr_mult']} | clip×{ts['gn_mult']} | "
                            f"cos={ts['cos_sim']} | noise={ts['gn_noise']} | ur={ts['upd_ratio']} | "
                            f"{ts['adj']} adj | {ts['anomalies']} anomalies")

                    if ev_avg < best_loss:
                        best_loss = ev_avg
                        torch.save(model.state_dict(), ckpt_dir / "best.pt")
                        log(f"  {_C.GREEN}★ Best!{_C.END} loss={best_loss:.4f}")

                # Save
                if step % save_steps == 0:
                    torch.save(model.state_dict(), ckpt_dir / f"ckpt-{step}.pt")
                    torch.save(optimizer.state_dict(), ckpt_dir / f"opt-{step}.pt")
                    sd = {"step": step, "best_loss": best_loss, "timestamp": datetime.now().isoformat()}
                    if curriculum is not None: sd["curriculum"] = curriculum.get_tier(step)[0]
                    if dynamic_sampler is not None: sd["mix"] = dynamic_sampler.get_weights()
                    if reflection_memory is not None: sd["reflection"] = len(reflection_memory.memory)
                    if synthetic_evo is not None: sd["synthetic"] = synthetic_evo.count
                    if exp_memory is not None: sd["memory"] = exp_memory.stats()
                    if tool_router is not None: sd["routing"] = tool_router.stats()
                    if auto_tuner is not None: sd["auto_tuner"] = auto_tuner.save_state()
                    with open(ckpt_dir / f"state-{step}.json", "w") as f:
                        json.dump(sd, f)
                    for pat in ["ckpt-*.pt", "opt-*.pt", "state-*.json"]:
                        files = sorted(ckpt_dir.glob(pat))
                        for f in files[:-5]:
                            f.unlink()
                    log(f"  {_C.GREEN}✓{_C.END} Saved ckpt-{step}")

                if step >= total:
                    break
        if step >= total:
            break

    # ── Final ───────────────────────────────────────────────────────
    total_time = time.time() - t0
    final_loss = ema_loss if ema_loss is not None else 0.0
    torch.save(model.state_dict(), ckpt_dir / "final.pt")

    log(f"\n{_C.BOLD}┌{'─'*63}┐{_C.END}")
    log(f"{_C.BOLD}│{_C.END} {_C.GREEN}✓ Training Complete!{_C.END}")
    log(f"{_C.BOLD}│{_C.END} {step:,} steps — {total_time/3600:.1f}h — {(step/total_time*3600):,.0f} steps/hr")
    log(f"{_C.BOLD}│{_C.END} Final loss: {final_loss:.4f} | Best eval: {best_loss:.4f}")
    log(f"{_C.BOLD}│{_C.END} Throughput: {tokens_processed/total_time:,.0f} tok/s")
    if curriculum is not None:
        log(f"{_C.BOLD}│{_C.END} Curriculum: {curriculum.get_tier(step)[0]}")
    if dynamic_sampler is not None:
        log(f"{_C.BOLD}│{_C.END} Dynamic Mix: {dynamic_sampler.get_weights()}")
    if auto_tuner is not None:
        ts = auto_tuner.stats()
        log(f"{_C.BOLD}│{_C.END} AutoTuner: {ts['state']} | LR×{ts['lr_mult']} | {ts['adj']} adj")
        log(f"{_C.BOLD}│{_C.END}   Best: {ts['best']} | Recoveries: {ts['recovery']} | DivRed: {ts['div_red']} | DegRed: {ts['deg_red']}")
    log(f"{_C.BOLD}└{'─'*63}┘{_C.END}")

    # ── Push to HuggingFace ───────────────────────────────────────────
    hf = os.environ.get("HF_TOKEN")
    if hf:
        try:
            from huggingface_hub import HfApi, login, create_repo
            import yaml, dataclasses
            login(hf)
            api = HfApi()
            create_repo(hf_repo, repo_type="model", exist_ok=True)

            # Model weights
            api.upload_file(path_or_fileobj=str(ckpt_dir / f"ckpt-{step}.pt"),
                            path_in_repo="model.pt", repo_id=hf_repo)
            log(f"  ✓ model.pt uploaded to {hf_repo}")

            # Optimizer
            opt_file = ckpt_dir / f"opt-{step}.pt"
            if opt_file.exists():
                api.upload_file(path_or_fileobj=str(opt_file),
                                path_in_repo="optimizer.pt", repo_id=hf_repo)
                log(f"  ✓ optimizer.pt")

            # State
            state_file = ckpt_dir / f"state-{step}.json"
            if state_file.exists():
                api.upload_file(path_or_fileobj=str(state_file),
                                path_in_repo="training_state.json", repo_id=hf_repo)
                log(f"  ✓ training_state.json")

            # Best
            best_file = ckpt_dir / "best.pt"
            if best_file.exists():
                api.upload_file(path_or_fileobj=str(best_file),
                                path_in_repo="best.pt", repo_id=hf_repo)
                log(f"  ✓ best.pt")

            # Tokenizer
            tok_dir = Path(VOL) / "tokenizer"
            for fname in ["tokenizer.json", "tokenizer_config.json"]:
                fp = tok_dir / fname
                if fp.exists():
                    api.upload_file(path_or_fileobj=str(fp), path_in_repo=fname, repo_id=hf_repo)
                    log(f"  ✓ {fname}")

            # Config YAML from DeeplmConfig defaults
            sys.path.insert(0, str(repo))
            from deeplm.config import DeeplmConfig
            cfg = DeeplmConfig()
            cfg_dict = dataclasses.asdict(cfg)
            cfg_dict['model_name'] = 'Deeplm'
            cfg_dict['version'] = '1.0.0'
            api.upload_file(
                path_or_fileobj=yaml.dump(cfg_dict, default_flow_style=False,
                                          sort_keys=False, allow_unicode=True).encode(),
                path_in_repo="config.yaml", repo_id=hf_repo,
            )
            log(f"  ✓ config.yaml")

            log(f"{_C.GREEN}✓ All assets pushed to https://huggingface.co/{hf_repo}{_C.END}")
        except Exception as e:
            log(f"{_C.RED}HF push failed: {e}{_C.END}")

    return {"steps": step, "time": round(total_time), "best_loss": round(best_loss, 4)}


@app.function(volumes={VOL: volume})
def tail_logs(n: int = 20):
    log_file = Path(VOL) / "logs" / "train.log"
    if not log_file.exists():
        print("No logs yet.")
        return
    with open(log_file) as f:
        lines = f.readlines()
    for line in lines[-n:]:
        print(line, end="")


@app.function(volumes={VOL: volume})
def watch_logs():
    import time
    log_file = Path(VOL) / "logs" / "train.log"
    pos = 0
    while True:
        if log_file.exists():
            with open(log_file) as f:
                f.seek(pos)
                new = f.read()
                if new:
                    print(new, end="", flush=True)
                    pos = f.tell()
        time.sleep(2)


@app.function(volumes={VOL: volume}, secrets=func_secrets, timeout=3600)
def upload_to_hf(repo_id: str = "samcheng0/deeplm-108m", step: int = None):
    import os
    from huggingface_hub import HfApi, login
    from pathlib import Path

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN not set")
        return

    login(hf_token)
    api = HfApi()
    ckpt_dir = Path(VOL) / "checkpoints"
    repo = Path(VOL) / "deeplm"

    if repo.exists():
        print("Pulling latest code...")
        result = subprocess.run(["git", "-C", str(repo), "pull", "--ff-only"],
                                capture_output=True, text=True)
        if result.returncode != 0:
            subprocess.run(["rm", "-rf", str(repo)], check=True)
            subprocess.run(["git", "clone", "--depth=1",
                "https://github.com/Luvion1/deeplm.git", str(repo)], check=True)

    if step:
        ckpt = ckpt_dir / f"ckpt-{step}.pt"
        opt = ckpt_dir / f"opt-{step}.pt"
    else:
        ckpts = list(ckpt_dir.glob("ckpt-*.pt"))
        if ckpts:
            ckpt = max(ckpts, key=lambda p: int(p.stem.split("-")[1]))
            s = int(ckpt.stem.split("-")[1])
            opt = ckpt_dir / f"opt-{s}.pt"
        else:
            ckpt = ckpt_dir / "best.pt" if (ckpt_dir / "best.pt").exists() else None
            opt = None

    print(f"Uploading to {repo_id}...")
    if ckpt and ckpt.exists():
        print(f"  Uploading {ckpt.name} ({ckpt.stat().st_size/1e9:.1f}GB)")
        api.upload_file(path_or_fileobj=str(ckpt), path_in_repo="model.pt", repo_id=repo_id)
        print("  ✓ model.pt")
    if opt and opt.exists():
        print(f"  Uploading {opt.name} ({opt.stat().st_size/1e9:.1f}GB)")
        api.upload_file(path_or_fileobj=str(opt), path_in_repo="optimizer.pt", repo_id=repo_id)
        print("  ✓ optimizer.pt")
    tok = Path(VOL) / "tokenizer" / "tokenizer.json"
    if tok.exists():
        api.upload_file(path_or_fileobj=str(tok), path_in_repo="tokenizer.json", repo_id=repo_id)
        print("  ✓ tokenizer.json")
    if repo.exists():
        for f in repo.rglob("*.py"):
            if "__pycache__" not in str(f):
                rel = f.relative_to(repo.parent)
                api.upload_file(path_or_fileobj=str(f), path_in_repo=str(rel), repo_id=repo_id)
        print("  ✓ deeplm/ (all code)")

    # Config YAML from DeeplmConfig defaults
    import sys
    sys.path.insert(0, str(repo))
    import yaml, dataclasses
    from deeplm.config import DeeplmConfig
    cfg = DeeplmConfig()
    cfg_dict = dataclasses.asdict(cfg)
    cfg_dict['model_name'] = 'Deeplm'
    cfg_dict['version'] = '1.0.0'
    api.upload_file(
        path_or_fileobj=yaml.dump(cfg_dict, default_flow_style=False, sort_keys=False, allow_unicode=True).encode(),
        path_in_repo="config.yaml", repo_id=repo_id,
    )
    print("  ✓ config.yaml")

    print(f"\n✓ Done! https://huggingface.co/{repo_id}")


@app.local_entrypoint()
def main(
    max_steps: int = 50000,
    total_steps: int = None,
    batch_size: int = 8,
    grad_accum: int = 4,
    max_seq_length: int = 2048,
    lr: float = 3e-4,
    warmup_steps: int = None,
    warmup_ratio: float = None,
    weight_decay: float = None,
    max_grad_norm: float = None,
    logging_steps: int = None,
    save_steps: int = None,
    compile: bool = True,
    mode: str = "train",
    tail: int = 20,
    step: int = None,
    repo_id: str = "samcheng0/deeplm-108m",
    lr_scheduler: str = "cosine",
    bf16: bool = True,
    pin_memory: bool = True,
    num_workers: int = 0,
    seed: int = 42,
    min_length: int = 50,
    max_length: int = 8000,
    max_repetition_ratio: float = 0.4,
    min_char_ratio: float = 0.25,
):
    if total_steps is not None:
        max_steps = total_steps
    if logging_steps is not None:
        log_steps = logging_steps
    else:
        log_steps = 10
    if save_steps is not None:
        save_steps = save_steps
    else:
        save_steps = 5000
    if warmup_steps is not None:
        warmup_ratio = warmup_steps / max_steps if max_steps > 0 else 0.03
    elif warmup_ratio is None:
        warmup_ratio = 0.03

    if mode == "train":
        train.spawn(
            max_steps=max_steps, batch_size=batch_size, grad_accum=grad_accum,
            max_seq_length=max_seq_length, lr=lr, warmup_ratio=warmup_ratio,
            weight_decay=weight_decay if weight_decay is not None else 0.1,
            max_grad_norm=max_grad_norm if max_grad_norm is not None else 1.0,
            save_steps=save_steps, log_steps=log_steps, compile=compile,
            lr_scheduler=lr_scheduler, bf16=bf16, num_workers=num_workers,
            pin_memory=pin_memory, seed=seed,
            min_length=min_length, max_length=max_length,
            max_repetition_ratio=max_repetition_ratio, min_char_ratio=min_char_ratio,
            hf_repo=repo_id,
        )
    elif mode == "tail":
        tail_logs.remote(n=tail)
    elif mode == "watch":
        watch_logs.remote()
    elif mode == "upload":
        upload_to_hf.remote(repo_id=repo_id, step=step)
