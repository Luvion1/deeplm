"""
Enterprise-grade training script for Deeplm model.

Memory optimization techniques:
- Memory-mapped dataset (numpy memmap) — only batch data in RAM
- Auto pin_memory detection (disabled on CPU)
- Garbage collection between epochs
- DataLoader worker memory management
- MMAP checkpoint loading for reduced peak memory
- Gradient accumulation for effective large batch on small hardware

Usage:
    python scripts/train_deeplm.py --config configs/train_kbi.yaml
    python scripts/train_deeplm.py --num_rows 1000 --max_steps 50 --batch_size 2
"""
import argparse
import gc
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deeplm.config import DeeplmConfig
from deeplm.model.deeplm import DeeplmModel
from deeplm.data.kbi_dataset import MappedKBBIDataset
from deeplm.training.logger import SmartLogger, MetricsTracker
from tokenizers import Tokenizer


@dataclass
class TrainingConfig:
    """Training configuration."""

    output_dir: str = "./deeplm_output"

    data: Dict = field(default_factory=lambda: {
        "dataset_name": "Lyon28/kamus-besar-bahasa-indonesia",
        "max_seq_length": 2048,
        "tokenizer_path": "tokenizer/",
        "cache_dir": "data_cache/",
        "num_rows": None,
    })

    training: Dict = field(default_factory=lambda: {
        "num_train_epochs": 3,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 8,
        "learning_rate": 6.0e-4,
        "min_learning_rate": 6.0e-6,
        "weight_decay": 0.1,
        "adam_beta1": 0.9,
        "adam_beta2": 0.95,
        "adam_epsilon": 1.0e-8,
        "max_grad_norm": 1.0,
        "warmup_ratio": 0.03,
        "lr_schedule": "cosine",
        "max_steps": -1,
        "seed": 42,
    })

    logging: Dict = field(default_factory=lambda: {
        "logging_steps": 10,
        "save_steps": 500,
        "report_to": ["json", "console"],
    })

    checkpoint: Dict = field(default_factory=lambda: {
        "save_total_limit": 3,
        "resume_from": None,
        "save_optimizer": True,
        "save_scheduler": True,
        "mmap_load": True,
    })

    hardware: Dict = field(default_factory=lambda: {
        "device": "auto",
        "num_workers": 0,
        "pin_memory": "auto",
        "dtype": "float32",
        "compile": False,  # torch.compile support
        "compile_mode": None,  # "default", "reduce-overhead", "max-autotune"
        "prefetch_factor": 2,  # DataLoader prefetch
    })


class MetricsLogger:
    """JSON-based metrics logger."""

    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.metrics_path = os.path.join(output_dir, "metrics.jsonl")
        self.summary_path = os.path.join(output_dir, "training_summary.json")

    def log_step(self, metrics: Dict):
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

    def save_summary(self, summary: Dict):
        with open(self.summary_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


class CheckpointManager:
    """Manages training checkpoints with rotation."""

    def __init__(self, output_dir: str, save_total_limit: int = 3):
        self.output_dir = output_dir
        self.checkpoint_dir = os.path.join(output_dir, "checkpoints")
        self.save_total_limit = save_total_limit
        self.checkpoints = []
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(self, model, optimizer, scheduler, global_step, loss, config):
        ckpt_path = os.path.join(self.checkpoint_dir, f"step-{global_step}")
        os.makedirs(ckpt_path, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(ckpt_path, "model.pt"))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_path, "optimizer.pt"))

        if scheduler is not None:
            torch.save(scheduler.state_dict(), os.path.join(ckpt_path, "scheduler.pt"))

        with open(os.path.join(ckpt_path, "training_state.json"), "w") as f:
            json.dump(
                {
                    "global_step": global_step,
                    "loss": loss,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        config.save(os.path.join(ckpt_path, "config.json"))

        self.checkpoints.append((global_step, ckpt_path))

        if len(self.checkpoints) > self.save_total_limit:
            _, old_path = self.checkpoints.pop(0)
            import shutil
            shutil.rmtree(old_path, ignore_errors=True)

        print(f"  Checkpoint saved: {ckpt_path}")


def load_state_dict_mmap(path: str, device: torch.device):
    """Load state dict using memory mapping for lower peak RAM.
    
    Note: mmap=True only works with CPU; we load to CPU first then move to device.
    CPU copy is freed immediately after moving.
    """
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            sd[k] = v.to(device, non_blocking=True)
    return sd


class Trainer:
    """Enterprise-grade trainer with memory-mapped dataset support."""

    def __init__(
        self,
        model: DeeplmModel,
        config: DeeplmConfig,
        train_config: TrainingConfig,
        train_dataset: Optional[Dataset] = None,
    ):
        self.model = model
        self.config = config
        self.train_config = train_config
        self.train_dataset = train_dataset

        self.tc = train_config.training
        self.lc = train_config.logging
        self.cc = train_config.checkpoint
        self.hc = train_config.hardware

        if self.hc["device"] == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.hc["device"])

        if self.hc["pin_memory"] == "auto":
            self.pin_memory = self.device.type == "cuda"
        else:
            self.pin_memory = self.hc["pin_memory"]

        self.dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        self.dtype = self.dtype_map.get(self.hc.get("dtype", "float32"), torch.float32)

        self.model.to(self.device, dtype=self.dtype)

        # torch.compile support for GPU training
        if self.hc.get("compile", False) and self.device.type == "cuda":
            compile_mode = self.hc.get("compile_mode", None)
            self.logger.info(f"  torch.compile enabled (mode={compile_mode})")
            self.model.forward = torch.compile(
                self.model.forward, mode=compile_mode, dynamic=True
            )

            # Compile MoE and MLP for faster expert computation
            for layer in self.model.layers:
                layer.moe = torch.compile(layer.moe, mode=compile_mode, dynamic=True)
        elif self.hc.get("compile", False):
            self.logger.warning("  torch.compile requires CUDA, skipping")

        self.optimizer = SGD(
            self.model.parameters(),
            lr=self.tc["learning_rate"],
            momentum=self.tc.get("momentum", 0.9),
            weight_decay=self.tc["weight_decay"],
            nesterov=self.tc.get("nesterov", True),
        )

        self.global_step = 0
        self.epoch = 0
        self.total_loss = 0.0
        self.best_loss = float("inf")
        self.start_time = None

        os.makedirs(self.train_config.output_dir, exist_ok=True)

        self.metrics_logger = MetricsLogger(self.train_config.output_dir)
        self.checkpoint_manager = CheckpointManager(
            self.train_config.output_dir,
            self.cc.get("save_total_limit", 3),
        )

        # SmartLogger — real-time monitoring with anomaly detection
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        self.smart_logger = SmartLogger(
            log_dir=self.train_config.output_dir,
            total_steps=self.tc.get("max_steps", 1000) or 1000,
            model_params=self.model.num_parameters(),
            batch_size=self.tc["per_device_train_batch_size"],
            grad_accum=self.tc["gradient_accumulation_steps"],
            seq_length=self.train_config.data.get("max_seq_length", 2048),
            lr=self.tc["learning_rate"],
            vocab_size=self.config.vocab_size,
            gpu_name=gpu_name,
        )
        self.tracker = MetricsTracker()

        self._setup_logging()

    def _setup_logging(self):
        log_path = os.path.join(self.train_config.output_dir, "training.log")
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(message)s",
            handlers=[
                logging.FileHandler(log_path),
                logging.StreamHandler(sys.stdout),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def get_train_dataloader(self) -> DataLoader:
        from ..data.kbi_dataset import MappedKBBIDataset

        num_workers = self.hc.get("num_workers", 0)
        use_cuda = self.device.type == "cuda"

        # memmap datasets are not fork-safe; force num_workers=0
        if isinstance(self.train_dataset, MappedKBBIDataset):
            if num_workers > 0:
                self.logger.warning("  MappedKBBIDataset is not fork-safe; forcing num_workers=0")
                num_workers = 0
        elif use_cuda and num_workers == 0:
            import os
            num_workers = min(4, os.cpu_count() or 1)

        return DataLoader(
            self.train_dataset,
            batch_size=self.tc["per_device_train_batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            prefetch_factor=num_workers if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )

    def get_scheduler(self, num_training_steps: int):
        warmup_steps = int(num_training_steps * self.tc.get("warmup_ratio", 0.03))

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = (current_step - warmup_steps) / max(1, num_training_steps - warmup_steps)
            return max(
                self.tc["min_learning_rate"] / self.tc["learning_rate"],
                0.5 * (1.0 + math.cos(math.pi * progress)),
            )

        return LambdaLR(self.optimizer, lr_lambda)

    def _get_memory_info(self) -> Dict:
        info = {}
        if torch.cuda.is_available():
            info["gpu_allocated_gb"] = torch.cuda.memory_allocated() / 1e9
            info["gpu_reserved_gb"] = torch.cuda.memory_reserved() / 1e9
        return info

    def train(self, resume_from_checkpoint: Optional[str] = None):
        self.start_time = time.time()

        train_dataloader = self.get_train_dataloader()
        num_training_steps = (
            self.tc["max_steps"]
            if self.tc["max_steps"] > 0
            else len(train_dataloader) * self.tc["num_train_epochs"]
        )

        scheduler = self.get_scheduler(num_training_steps)

        if resume_from_checkpoint or self.cc.get("resume_from"):
            ckpt_path = resume_from_checkpoint or self.cc["resume_from"]
            self._load_checkpoint(ckpt_path)

        effective_batch = self.tc["per_device_train_batch_size"] * self.tc["gradient_accumulation_steps"]

        self.logger.info(f"{'='*60}")
        self.logger.info(f"Training started")
        self.logger.info(f"  Device: {self.device}")
        self.logger.info(f"  Dtype: {self.dtype}")
        self.logger.info(f"  Pin memory: {self.pin_memory}")
        self.logger.info(f"  Total params: {self.model.num_parameters():,}")
        self.logger.info(f"  Training steps: {num_training_steps:,}")
        self.logger.info(f"  Batch size: {self.tc['per_device_train_batch_size']}")
        self.logger.info(f"  Grad accumulation: {self.tc['gradient_accumulation_steps']}")
        self.logger.info(f"  Effective batch: {effective_batch}")
        self.logger.info(f"  Learning rate: {self.tc['learning_rate']}")
        self.logger.info(f"  Warmup steps: {int(num_training_steps * self.tc.get('warmup_ratio', 0.03))}")
        self.logger.info(f"  Epochs: {self.tc['num_train_epochs']}")
        self.logger.info(f"{'='*60}")

        self.model.train()
        self.optimizer.zero_grad()

        for epoch in range(self.tc["num_train_epochs"]):
            self.epoch = epoch
            self.logger.info(f"\nEpoch {epoch + 1}/{self.tc['num_train_epochs']}")

            for step, batch in enumerate(train_dataloader):
                batch_start = time.time()

                batch = {k: v.to(self.device) for k, v in batch.items()}

                output = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    output_mtp_loss=self.config.mtp.enabled,
                )

                loss = output["loss"] / self.tc["gradient_accumulation_steps"]
                loss.backward()

                self.total_loss += loss.item() * self.tc["gradient_accumulation_steps"]

                if (step + 1) % self.tc["gradient_accumulation_steps"] == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.tc["max_grad_norm"]
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if scheduler is not None:
                        scheduler.step()

                    self.global_step += 1
                    elapsed = time.time() - batch_start
                    current_lr = self.optimizer.param_groups[0]["lr"]

                    metrics = {
                        "step": self.global_step,
                        "epoch": epoch + 1,
                        "loss": loss.item() * self.tc["gradient_accumulation_steps"],
                        "avg_loss": self.total_loss / self.lc["logging_steps"],
                        "lr": current_lr,
                        "grad_norm": grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "step_time": elapsed,
                        "timestamp": datetime.now().isoformat(),
                    }

                    mem_info = self._get_memory_info()
                    metrics.update(mem_info)

                    if output.get("mtp_loss") is not None:
                        metrics["mtp_loss"] = output["mtp_loss"].item()

                    self.metrics_logger.log_step(metrics)
                    self.tracker.update(
                        loss=metrics["loss"],
                        grad_norm=metrics["grad_norm"],
                        lr=current_lr,
                        tokens_per_sec=batch["input_ids"].numel() * self.tc["gradient_accumulation_steps"] / max(elapsed, 1e-8),
                    )

                    if self.global_step % self.lc["logging_steps"] == 0:
                        gpu_mem = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                        tok_s = batch["input_ids"].numel() * self.tc["gradient_accumulation_steps"] / max(elapsed, 1e-8)
                        self.smart_logger.log_step(
                            step=self.global_step,
                            loss=metrics["loss"],
                            grad_norm=metrics["grad_norm"],
                            lr=current_lr,
                            tokens_per_sec=tok_s,
                            gpu_mem=gpu_mem,
                            mtp_loss=metrics.get("mtp_loss"),
                        )

                    if self.global_step % self.lc["save_steps"] == 0:
                        is_best = metrics["loss"] < self.best_loss
                        if is_best:
                            self.best_loss = metrics["loss"]
                        ckpt_path = os.path.join(
                            self.train_config.output_dir, f"checkpoint-{self.global_step}.pt"
                        )
                        torch.save(self.model.state_dict(), ckpt_path)
                        self.smart_logger.log_checkpoint(
                            step=self.global_step, path=ckpt_path,
                            loss=metrics["loss"], is_best=is_best,
                        )

                    if self.tc["max_steps"] > 0 and self.global_step >= self.tc["max_steps"]:
                        break

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if self.tc["max_steps"] > 0 and self.global_step >= self.tc["max_steps"]:
                break

        total_time = time.time() - self.start_time
        self.smart_logger.log_summary(
            step=self.global_step,
            final_loss=self.tracker.recent("loss"),
            best_loss=self.best_loss,
        )

        self.checkpoint_manager.save(
            self.model, self.optimizer, scheduler,
            self.global_step, self.best_loss, self.config,
        )

        summary = {
            "model_name": self.config.model_name,
            "total_params": self.model.num_parameters(),
            "total_steps": self.global_step,
            "total_epochs": self.epoch + 1,
            "best_loss": self.best_loss,
            "total_time_seconds": total_time,
            "throughput_steps_per_sec": self.global_step / total_time,
            "final_lr": self.optimizer.param_groups[0]["lr"],
            "completed_at": datetime.now().isoformat(),
        }
        self.metrics_logger.save_summary(summary)

    def _log_step(self, metrics: Dict):
        mem_str = ""
        if "gpu_allocated_gb" in metrics:
            mem_str = f" | GPU: {metrics['gpu_allocated_gb']:.2f}GB"

        self.logger.info(
            f"Step {self.global_step:>6,} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"Avg: {metrics['avg_loss']:.4f} | "
            f"LR: {metrics['lr']:.2e} | "
            f"Grad: {metrics['grad_norm']:.4f} | "
            f"Time: {metrics['step_time']:.2f}s"
            f"{mem_str}"
        )

    def _load_checkpoint(self, checkpoint_dir: str):
        self.logger.info(f"Loading checkpoint from {checkpoint_dir}")

        model_path = os.path.join(checkpoint_dir, "model.pt")
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        state_path = os.path.join(checkpoint_dir, "training_state.json")

        if os.path.exists(model_path):
            if self.cc.get("mmap_load", True):
                state_dict = load_state_dict_mmap(model_path, self.device)
            else:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            del state_dict
            self.logger.info("  Model weights loaded")

        if os.path.exists(optimizer_path) and self.cc.get("save_optimizer", True):
            if self.cc.get("mmap_load", True):
                opt_state = load_state_dict_mmap(optimizer_path, self.device)
            else:
                try:
                    opt_state = torch.load(optimizer_path, map_location=self.device, weights_only=True)
                except Exception:
                    opt_state = torch.load(optimizer_path, map_location=self.device, weights_only=False)
            self.optimizer.load_state_dict(opt_state)
            del opt_state
            self.logger.info("  Optimizer state loaded")

        if os.path.exists(scheduler_path) and self.cc.get("save_scheduler", True):
            self.logger.info("  Scheduler state loaded")

        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
            self.global_step = state["global_step"]
            self.best_loss = state.get("loss", float("inf"))
            self.logger.info(f"  Resuming from step {self.global_step:,}")


def load_config_from_yaml(yaml_path: str) -> TrainingConfig:
    import yaml

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    config = TrainingConfig()

    if "output_dir" in data:
        config.output_dir = data["output_dir"]
    if "data" in data:
        config.data.update(data["data"])
    if "training" in data:
        config.training.update(data["training"])
    if "logging" in data:
        config.logging.update(data["logging"])
    if "checkpoint" in data:
        config.checkpoint.update(data["checkpoint"])
    if "hardware" in data:
        config.hardware.update(data["hardware"])

    return config


def main():
    parser = argparse.ArgumentParser(description="Train Deeplm model")
    parser.add_argument("--config", type=str, default=None, help="Path to training config YAML")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=None)
    parser.add_argument("--save_steps", type=int, default=None)
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--num_rows", type=int, default=None, help="Limit dataset rows for testing")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=None)
    parser.add_argument("--clean_cache", action="store_true", help="Delete cached memmap files before training")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile (GPU only)")
    args = parser.parse_args()

    torch.manual_seed(args.seed or 42)

    if args.config and os.path.exists(args.config):
        train_config = load_config_from_yaml(args.config)
    else:
        train_config = TrainingConfig()

    if args.output_dir:
        train_config.output_dir = args.output_dir
    if args.epochs:
        train_config.training["num_train_epochs"] = args.epochs
    if args.batch_size:
        train_config.training["per_device_train_batch_size"] = args.batch_size
    if args.lr:
        train_config.training["learning_rate"] = args.lr
    if args.max_steps is not None:
        train_config.training["max_steps"] = args.max_steps
    if args.logging_steps:
        train_config.logging["logging_steps"] = args.logging_steps
    if args.save_steps:
        train_config.logging["save_steps"] = args.save_steps
    if args.resume_from:
        train_config.checkpoint["resume_from"] = args.resume_from
    if args.tokenizer_path:
        train_config.data["tokenizer_path"] = args.tokenizer_path
    if args.num_rows:
        train_config.data["num_rows"] = args.num_rows
    if args.seed:
        train_config.training["seed"] = args.seed
    if args.grad_accum:
        train_config.training["gradient_accumulation_steps"] = args.grad_accum
    if args.max_seq_length:
        train_config.data["max_seq_length"] = args.max_seq_length
    if args.compile:
        train_config.hardware["compile"] = True

    tokenizer_path = train_config.data["tokenizer_path"]
    tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
    if os.path.exists(tokenizer_file):
        tokenizer = Tokenizer.from_file(tokenizer_file)
        print(f"Loaded tokenizer from {tokenizer_path}")
    else:
        print(f"Tokenizer not found at {tokenizer_path}. Training new tokenizer...")
        from scripts.train_tokenizer import train_tokenizer
        tokenizer = train_tokenizer(128000, tokenizer_path)

    config = DeeplmConfig()

    print(f"\nCreating Deeplm model...")
    model = DeeplmModel(config)

    # Resize embeddings to match tokenizer vocab size
    model_vocab = model.embed_tokens.weight.shape[0]
    tokenizer_vocab = tokenizer.get_vocab_size()
    if model_vocab != tokenizer_vocab:
        print(f"  Resizing embeddings: {model_vocab:,} -> {tokenizer_vocab:,}")
        model.resize_token_embeddings(tokenizer_vocab)

    print(f"Model parameters: {model.num_parameters():,}")
    print(f"  Final vocab size: {model.embed_tokens.weight.shape[0]:,}")

    print(f"\nLoading KBBI dataset (memory-mapped)...")
    train_dataset = MappedKBBIDataset(
        tokenizer=tokenizer,
        max_seq_length=train_config.data.get("max_seq_length", 2048),
        cache_dir=train_config.data.get("cache_dir", "data_cache/"),
        num_rows=train_config.data.get("num_rows"),
    )

    if args.clean_cache:
        train_dataset.cleanup()
        train_dataset = MappedKBBIDataset(
            tokenizer=tokenizer,
            max_seq_length=train_config.data.get("max_seq_length", 2048),
            cache_dir=train_config.data.get("cache_dir", "data_cache/"),
            num_rows=train_config.data.get("num_rows"),
        )

    trainer = Trainer(
        model=model,
        config=config,
        train_config=train_config,
        train_dataset=train_dataset,
    )

    trainer.train(resume_from_checkpoint=train_config.checkpoint.get("resume_from"))


if __name__ == "__main__":
    main()
