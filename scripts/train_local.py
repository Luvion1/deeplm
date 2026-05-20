"""
Local training script for Deeplm — optimized for single-machine training.

Features:
- SGD + momentum + nesterov optimizer
- Warmup + cosine LR scheduling
- Gradient accumulation for large effective batch sizes
- Memory-mapped KBBI dataset (minimal RAM usage)
- torch.compile support for GPU
- Automatic checkpointing with best-loss tracking
- JSONL metrics logging
- Resume from checkpoint
- Detailed colored logging with ETA, EMA loss, GPU memory

Usage:
    # Quick test (CPU)
    python scripts/train_local.py --num-rows 1000 --max-steps 50 --batch-size 2

    # Full training (GPU)
    python scripts/train_local.py --max-steps 20000 --batch-size 8 --grad-accum 4 --compile

    # Resume training
    python scripts/train_local.py --resume-from deeplm_output/checkpoints/step-5000
"""
import argparse
import gc
import json
import math
import os
import re
import sys
import time
from collections import deque
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deeplm.config import DeeplmConfig
from deeplm.model.deeplm import DeeplmModel
from deeplm.data.kbi_dataset import MappedKBBIDataset
from deeplm.data.dataset_registry import SemanticCategorizer, DatasetRegistry, CategorizedDataset
from deeplm.training.curriculum_router import CurriculumRouter


# ── ANSI Colors ─────────────────────────────────────────────────────
class C:
    END = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    _gpu_cache = ""

_ansi_pat = re.compile(r'\033\[[0-9;]*m')


# ── Logger ──────────────────────────────────────────────────────────
class DetailedLogger:
    """Detailed training logger with timestamps, GPU memory, and file output."""

    def __init__(self, output_dir: str, total_steps: int, device: torch.device):
        os.makedirs(output_dir, exist_ok=True)
        self.log_file = os.path.join(output_dir, "train.log")
        self.metrics_path = os.path.join(output_dir, "metrics.jsonl")
        self.total_steps = total_steps
        self.device = device
        self._last_gpu_log = 0.0

    def _gpu_str(self):
        if self.device.type != "cuda":
            return "CPU"
        now = time.time()
        if now - self._last_gpu_log > 1:
            gpu_mem = torch.cuda.memory_allocated() / 1e9
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            gpu_pct = gpu_mem / gpu_total * 100
            self._last_gpu_log = now
            C._gpu_cache = f"{gpu_mem:.1f}/{gpu_total:.0f}GB ({gpu_pct:.0f}%)"
        return C._gpu_cache

    def log(self, msg, color=""):
        ts = datetime.now().strftime("%H:%M:%S")
        prefix = f"{C.DIM}{ts}{C.END} {C.DIM}{self._gpu_str()}{C.END} "
        colored = f"{prefix}{color}{msg}{C.END}" if color else f"{prefix}{msg}"
        print(colored, flush=True)
        clean = _ansi_pat.sub("", f"{ts} {self._gpu_str()} {msg}")
        with open(self.log_file, "a") as f:
            f.write(clean + "\n")

    def log_metrics(self, metrics: Dict):
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")


# ── Device ──────────────────────────────────────────────────────────
def setup_device(device_str: str = "auto") -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ── Optimizer ───────────────────────────────────────────────────────
def build_optimizer(model: nn.Module, lr: float, momentum: float = 0.9,
                    weight_decay: float = 0.1, nesterov: bool = True) -> SGD:
    return SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )


# ── Scheduler ───────────────────────────────────────────────────────
def build_scheduler(optimizer: SGD, total_steps: int, warmup_ratio: float,
                    min_lr_ratio: float = 1e-2) -> LambdaLR:
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_fn)


# ── Checkpoint Manager ──────────────────────────────────────────────
class CheckpointManager:
    def __init__(self, output_dir: str, save_total_limit: int = 3):
        self.ckpt_dir = os.path.join(output_dir, "checkpoints")
        self.save_total_limit = save_total_limit
        self.checkpoints = []
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def save(self, model, optimizer, scheduler, global_step, loss, config, is_best=False):
        ckpt_path = os.path.join(self.ckpt_dir, f"step-{global_step}")
        os.makedirs(ckpt_path, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(ckpt_path, "model.pt"))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))
        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))

        with open(os.path.join(ckpt_path, "state.json"), "w") as f:
            json.dump({
                "global_step": global_step,
                "loss": loss,
                "is_best": is_best,
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)

        with open(os.path.join(ckpt_path, "config.json"), "w") as f:
            json.dump({
                "model_name": config.model_name,
                "vocab_size": config.vocab_size,
                "hidden_size": config.architecture.hidden_size,
                "num_layers": config.architecture.num_layers,
            }, f, indent=2)

        self.checkpoints.append((global_step, ckpt_path))
        if len(self.checkpoints) > self.save_total_limit:
            _, old_path = self.checkpoints.pop(0)
            import shutil
            shutil.rmtree(old_path, ignore_errors=True)

        return ckpt_path

    def load(self, ckpt_path, model, optimizer, scheduler=None):
        model_path = os.path.join(ckpt_path, "model.pt")
        opt_path = os.path.join(ckpt_path, "optimizer.pt")
        sched_path = os.path.join(ckpt_path, "scheduler.pt")
        state_path = os.path.join(ckpt_path, "state.json")

        if os.path.exists(model_path):
            sd = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(sd, strict=False)
            del sd

        if os.path.exists(opt_path):
            try:
                opt_sd = torch.load(opt_path, map_location="cpu", weights_only=True)
            except Exception:
                opt_sd = torch.load(opt_path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(opt_sd)
            del opt_sd

        if scheduler and os.path.exists(sched_path):
            scheduler.load_state_dict(torch.load(sched_path, map_location="cpu", weights_only=True))

        state = {}
        if os.path.exists(state_path):
            with open(state_path) as f:
                state = json.load(f)

        return state.get("global_step", 0), state.get("loss", float("inf"))


# ── Tokenizer ───────────────────────────────────────────────────────
def load_tokenizer(path: str) -> Tokenizer:
    tokenizer_file = os.path.join(path, "tokenizer.json")
    if os.path.exists(tokenizer_file):
        tokenizer = Tokenizer.from_file(tokenizer_file)
        return tokenizer

    print(f"Tokenizer not found at {path}. Training new tokenizer...")
    from scripts.train_tokenizer import train_tokenizer
    return train_tokenizer(128000, path)


# ── Training ────────────────────────────────────────────────────────
def train(args):
    device = setup_device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.set_float32_matmul_precision('high')

    # Tokenizer
    tokenizer = load_tokenizer(args.tokenizer_path)

    # Model
    config = DeeplmConfig()
    model = DeeplmModel(config)

    vocab_size = tokenizer.get_vocab_size()
    if vocab_size != config.vocab_size:
        model.resize_token_embeddings(vocab_size)

    # Compile (GPU only)
    if device.type == "cuda" and args.compile:
        torch._dynamo.config.suppress_errors = True
        model = torch.compile(model, mode="default", dynamic=True)

    model.to(device)
    total_params = model.num_parameters()

    # Dataset
    dataset = MappedKBBIDataset(
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        cache_dir=args.cache_dir,
        num_rows=args.num_rows,
    )

    # Dataloader
    use_cuda = device.type == "cuda"
    num_workers = 0 if not use_cuda else min(4, os.cpu_count() or 1)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_cuda,
        drop_last=True,
    )

    # Optimizer & scheduler
    total_steps = args.max_steps if args.max_steps > 0 else len(dataloader) * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    optimizer = build_optimizer(model, args.lr, args.momentum, args.weight_decay)
    scheduler = build_scheduler(optimizer, total_steps, args.warmup_ratio)

    # AutoTuner + CurriculumRouter
    auto_tuner = None
    curriculum_router = None
    if args.use_auto_tuner:
        from deeplm.training.auto_tuner import AutoTuner
        warmup_steps_at = int(total_steps * args.warmup_ratio)
        auto_tuner = AutoTuner(args.lr, warmup_steps_at, total_steps, args.max_grad_norm)
        curriculum_router = CurriculumRouter()
        auto_tuner._curriculum_router = curriculum_router

    # Logger
    logger = DetailedLogger(args.output_dir, total_steps, device)
    ckpt_manager = CheckpointManager(args.output_dir, save_total_limit=args.save_total_limit)

    if auto_tuner is not None:
        logger.log(f"AutoTuner: enabled (warmup={warmup_steps_at:,}, total={total_steps:,})")
        logger.log(f"CurriculumRouter: enabled (8 categories)")

    # Resume
    start_step = 0
    best_loss = float("inf")
    if args.resume_from:
        start_step, best_loss = ckpt_manager.load(args.resume_from, model, optimizer, scheduler)
        # Fast-forward scheduler
        for _ in range(start_step):
            scheduler.step()
        logger.log(f"Resumed from step {start_step:,}, best_loss={best_loss:.4f}", color=C.CYAN)

    effective_bsz = args.batch_size * args.grad_accum

    # ── GPU info ────────────────────────────────────────────────────
    gpu_name = ""
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0).replace("NVIDIA ", "")
        gpu_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.log(f"GPU: {gpu_name} ({gpu_total:.1f}GB)")
    else:
        logger.log(f"Device: CPU ({os.cpu_count()} cores)")

    logger.log(f"Tokenizer: {vocab_size:,}")
    logger.log(f"Model: {total_params:,} params")
    logger.log(f"Dataset: {len(dataset):,} samples, seq_len={args.max_seq_length}")

    if use_cuda and args.compile:
        logger.log("torch.compile: enabled")

    # ── Training header ─────────────────────────────────────────────
    lr_sched = "cosine"
    dtype_str = args.dtype.upper()
    logger.log(f"{C.BOLD}┌{'─'*63}┐{C.END}")
    logger.log(f"{C.BOLD}│{C.END} {C.CYAN}Deeplm{C.END} — {gpu_name or 'CPU'} — {C.BOLD}{total_params:,}{C.END} params{' ' * max(0, 35 - len(f'{total_params:,}'))}{C.BOLD}│{C.END}")
    logger.log(f"{C.BOLD}│{C.END} {'':<61} {C.BOLD}│{C.END}")
    logger.log(f"{C.BOLD}│{C.END} {total_steps:,} steps │ batch {args.batch_size}×{args.grad_accum}={effective_bsz} eff │ lr {args.lr} │ wd {args.weight_decay}")
    logger.log(f"{C.BOLD}│{C.END} Dataset: KBBI ({len(dataset):,} samples) | {dtype_str} | {lr_sched} sched | SGD+momentum")
    logger.log(f"{C.BOLD}│{C.END} Logs: {args.output_dir}/metrics.jsonl")
    logger.log(f"{C.BOLD}└{'─'*63}┘{C.END}")
    logger.log(f"{C.DIM}{'─'*70}{C.END}")
    hdr = f"  {C.BOLD}{'Step':>7} | {'Loss':>9} | {'μLoss':>9} | {'LR':>10} | {'Grad':>7} | {'Mem':>12} | {'Tok/s':>8} | {'ETA':>6}{C.END}"
    logger.log(hdr)
    logger.log(f"{C.DIM}{'─'*70}{C.END}")

    # ── Training Loop ───────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()
    step = start_step
    t0 = time.time()
    accum = 0.0
    ema_loss = None
    losses = deque(maxlen=100)
    micro_step = 0
    first_batch = True

    for epoch in range(args.epochs):
        logger.log(f"Epoch {epoch + 1}/{args.epochs} ({len(dataloader)} batches)")

        for batch_idx, batch in enumerate(dataloader):
            if first_batch:
                first_batch = False

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            output = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                output_mtp_loss=config.mtp.enabled,
            )

            loss = output["loss"] / args.grad_accum
            loss.backward()
            micro_step += 1
            accum += loss.item()
            losses.append(loss.item())

            if micro_step % args.grad_accum == 0:
                # AutoTuner: capture gradients and get control
                current_control = None
                grad_norm_clip = args.max_grad_norm
                if auto_tuner is not None:
                    auto_tuner.capture_gradients(model)
                    current_control = auto_tuner.get_training_control(step)
                    grad_norm_clip = args.max_grad_norm * current_control.hyperparams.gn_mult

                    # Apply LR multiplier
                    eff_lr = args.lr * current_control.hyperparams.lr_mult
                    for group in optimizer.param_groups:
                        group["lr"] = eff_lr

                    # Apply optimizer betas
                    for group in optimizer.param_groups:
                        if "betas" in group:
                            group["betas"] = (current_control.optimizer.beta1, current_control.optimizer.beta2)

                    # Update curriculum router
                    if curriculum_router is not None:
                        curriculum_router.set_phase(current_control.phase, step)
                        if current_control.dataset.category_weights:
                            curriculum_router.set_manual_weights(current_control.dataset.category_weights)

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_clip)
                grad_norm_val = grad_norm.item() if torch.is_tensor(grad_norm) else float(grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                # AutoTuner: capture update and record
                if auto_tuner is not None:
                    auto_tuner.capture_update(model)
                    auto_tuner.record(step, losses[-1], grad_norm_val)

                # EMA loss
                ema_loss = losses[-1] if ema_loss is None else 0.9 * ema_loss + 0.1 * losses[-1]

                # ── Logging ──────────────────────────────────────────────
                if step % args.log_every == 0:
                    avg_loss = accum / args.log_every
                    lr_now = optimizer.param_groups[0]["lr"]
                    elapsed = time.time() - t0
                    steps_remain = total_steps - step
                    eta = steps_remain * elapsed / max(1, step - start_step) if step > start_step else 0
                    tok_s = (step * args.batch_size * args.grad_accum * args.max_seq_length) / max(1, elapsed)

                    mem_str = f"{torch.cuda.memory_allocated()/1e9:.1f}GB" if use_cuda else "CPU"

                    log_line = (
                        f"  {step:>7,} | {avg_loss:>9.4f} | {ema_loss:>9.4f} | {lr_now:>10.2e} | "
                        f"{grad_norm_val:>7.4f} | {mem_str:>12} | {tok_s:>8,.0f} | {eta:>6.0f}s"
                    )
                    if auto_tuner is not None:
                        log_line += f" | {auto_tuner.phase} LR×{auto_tuner.lr_mult:.2f}"
                        if current_control:
                            log_line += f" | {current_control.phase}"

                    logger.log(log_line, color=C.GREEN)

                    logger.log_metrics({
                        "step": step,
                        "epoch": epoch + 1,
                        "loss": avg_loss,
                        "ema_loss": ema_loss,
                        "lr": lr_now,
                        "grad_norm": grad_norm_val,
                        "tokens_per_sec": tok_s,
                        "eta_seconds": eta,
                        "timestamp": datetime.now().isoformat(),
                    })
                    if output.get("mtp_loss") is not None:
                        logger.log_metrics({"mtp_loss": output["mtp_loss"].item()})

                    accum = 0.0

                # ── Checkpoint ───────────────────────────────────────────
                if step % args.save_every == 0:
                    current_loss = output["loss"].item() * args.grad_accum
                    is_best = current_loss < best_loss
                    if is_best:
                        best_loss = current_loss
                    ckpt_path = ckpt_manager.save(
                        model, optimizer, scheduler, step,
                        current_loss, config, is_best=is_best,
                    )
                    logger.log(f"Checkpoint saved: {ckpt_path} (best={is_best})", color=C.CYAN)

                if args.max_steps > 0 and step >= args.max_steps:
                    break

        # End of epoch cleanup
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if args.max_steps > 0 and step >= args.max_steps:
            break

    # ── Final save ──────────────────────────────────────────────────
    total_time = time.time() - t0
    final_path = os.path.join(args.output_dir, "final.pt")
    torch.save(model.state_dict(), final_path)

    summary = {
        "model_name": config.model_name,
        "total_params": total_params,
        "total_steps": step,
        "total_epochs": epoch + 1,
        "best_loss": best_loss,
        "total_time_seconds": total_time,
        "throughput_steps_per_sec": step / max(total_time, 1),
        "completed_at": datetime.now().isoformat(),
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    logger.log(f"{C.DIM}{'─'*70}{C.END}")
    logger.log(f"{C.BOLD}Training complete!{C.END}")
    logger.log(f"  Steps: {step:,} in {total_time:.0f}s ({total_time/60:.1f} min)")
    logger.log(f"  Best loss: {best_loss:.4f}")
    logger.log(f"  Model: {final_path}")
    logger.log(f"{C.DIM}{'─'*70}{C.END}")


# ── CLI ─────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Train Deeplm locally")

    parser.add_argument("--tokenizer-path", type=str, default="tokenizer/")
    parser.add_argument("--cache-dir", type=str, default="data_cache/")
    parser.add_argument("--num-rows", type=int, default=None)
    parser.add_argument("--max-seq-length", type=int, default=2048)

    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=6e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--nesterov", action="store_true", default=True)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])

    parser.add_argument("--output-dir", type=str, default="deeplm_output")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--resume-from", type=str, default=None)

    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--use-auto-tuner", action="store_true", help="Enable AutoTuner for adaptive training control")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
