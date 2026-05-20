"""
Trainer for Deeplm — full-featured training loop.

Features:
- Gradient accumulation with correct step counting
- Warmup + cosine LR scheduling
- Gradient checkpointing support
- Checkpoint save/load with optimizer state
- Evaluation loop
- Smart logging integration
"""
import os
import math
import json
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, IterableDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ..config import DeeplmConfig
from ..model.deeplm import DeeplmModel
from .control import TrainingControl


@dataclass
class TrainingArgs:
    output_dir: str = "./deeplm_output"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 6.0e-4
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.03
    lr_schedule: str = "cosine"
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 1000
    max_steps: int = -1
    gradient_checkpointing: bool = True
    fp16: bool = False
    bf16: bool = False
    seed: int = 42
    eval_dataset: Optional[Dataset] = None
    max_eval_samples: int = 1000
    use_auto_tuner: bool = False


class Trainer:
    """Full-featured trainer for Deeplm."""

    def __init__(self, model: DeeplmModel, config: DeeplmConfig,
                 train_dataset: Optional[Dataset] = None,
                 args: Optional[TrainingArgs] = None,
                 eval_dataset: Optional[Dataset] = None,
                 data_collator: Optional[Callable] = None):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset or args.eval_dataset if args else None
        self.args = args or TrainingArgs()
        self.data_collator = data_collator or self._default_collate_fn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        # Separate weight decay for norm/bias parameters (AdamW best practice)
        no_decay = {"bias", "LayerNorm", "layer_norm", "norm"}
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]

        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
        )

        self.global_step = 0
        self.epoch = 0
        self.total_loss = 0.0
        self.best_eval_loss = float("inf")
        self.scaler = torch.cuda.amp.GradScaler() if self.args.fp16 else None
        self.auto_tuner = None
        self.curriculum_router = None
        self.current_control = None

    def _default_collate_fn(self, batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def get_train_dataloader(self) -> DataLoader:
        shuffle = not isinstance(self.train_dataset, IterableDataset)
        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=shuffle,
            num_workers=0,
            collate_fn=self.data_collator,
            drop_last=True,
        )

    def get_eval_dataloader(self) -> DataLoader:
        if self.eval_dataset is None:
            return None
        return DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_train_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=self.data_collator,
        )

    def get_scheduler(self, num_training_steps: int):
        warmup_steps = int(num_training_steps * self.args.warmup_ratio)

        def lr_fn(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            if self.args.lr_schedule == "cosine":
                progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
            elif self.args.lr_schedule == "linear":
                progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
                return max(0.0, 1.0 - progress)
            return 1.0

        return LambdaLR(self.optimizer, lr_fn)

    def train(self, resume_from_checkpoint: Optional[str] = None,
              log_callback: Optional[Callable] = None):
        args = self.args
        os.makedirs(args.output_dir, exist_ok=True)

        train_dataloader = self.get_train_dataloader()
        eval_dataloader = self.get_eval_dataloader()

        if args.max_steps > 0:
            num_training_steps = args.max_steps
        else:
            num_training_steps = len(train_dataloader) * args.num_train_epochs

        scheduler = self.get_scheduler(num_training_steps)

        if args.use_auto_tuner:
            from .auto_tuner import AutoTuner
            from .curriculum_router import CurriculumRouter
            warmup_steps = int(num_training_steps * args.warmup_ratio)
            self.auto_tuner = AutoTuner(args.learning_rate, warmup_steps, num_training_steps, args.max_grad_norm)
            self.curriculum_router = CurriculumRouter()
            self.auto_tuner._curriculum_router = self.curriculum_router
            if resume_from_checkpoint:
                state_path = os.path.join(resume_from_checkpoint, "training_state.json")
                if os.path.exists(state_path):
                    with open(state_path) as f:
                        state = json.load(f)
                        if "auto_tuner_state" in state:
                            self.auto_tuner.restore_state(state["auto_tuner_state"])
            print("  AutoTuner: enabled")
            print("  CurriculumRouter: enabled")

        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint, scheduler)

        self.model.train()
        self.optimizer.zero_grad()

        micro_step = 0
        t_start = time.time()

        for epoch in range(args.num_train_epochs):
            self.epoch = epoch
            for step, batch in enumerate(train_dataloader):
                batch = {k: v.to(self.device) for k, v in batch.items()}

                if self.scaler is not None:
                    with torch.cuda.amp.autocast():
                        output = self.model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                            output_mtp_loss=self.config.mtp.enabled,
                        )
                        loss = output["loss"] / args.gradient_accumulation_steps
                    self.scaler.scale(loss).backward()
                else:
                    output = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                        output_mtp_loss=self.config.mtp.enabled,
                    )
                    loss = output["loss"] / args.gradient_accumulation_steps
                    loss.backward()

                self.total_loss += loss.item() * args.gradient_accumulation_steps
                micro_step += 1

                if micro_step % args.gradient_accumulation_steps == 0:
                    at = self.auto_tuner
                    if at is not None:
                        adj = at.get_adjustments(self.global_step)
                        at.capture_gradients(self.model)
                        control = at.get_training_control(self.global_step)
                        self.apply_training_control(control)
                        grad_norm_clip = self._get_grad_norm_limit()
                    else:
                        grad_norm_clip = args.max_grad_norm
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_norm_clip)
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_norm_clip)
                        self.optimizer.step()
                    if at is not None:
                        at.capture_update(self.model)
                    self.optimizer.zero_grad()
                    scheduler.step()
                    self.global_step += 1

                    # Record category loss if router available
                    if self.curriculum_router is not None and self.current_control is not None:
                        for cat in self.current_control.dataset.category_weights:
                            cat_loss = self.current_control.dataset.category_weights.get(cat, 0)
                            if cat_loss > 0:
                                self.curriculum_router.record_loss(cat, avg_loss, self.global_step)

                    # Logging
                    if self.global_step % args.logging_steps == 0:
                        avg_loss = self.total_loss / (args.logging_steps * args.gradient_accumulation_steps)
                        lr = self.optimizer.param_groups[0]["lr"]
                        grad_norm = self._get_grad_norm()
                        elapsed = time.time() - t_start
                        tok_per_sec = self._compute_tok_per_sec(batch, elapsed)

                        if at is not None:
                            at.record(self.global_step, avg_loss, grad_norm)

                        log_entry = {
                            "step": self.global_step,
                            "loss": avg_loss,
                            "lr": lr,
                            "grad_norm": grad_norm,
                            "tok_per_sec": tok_per_sec,
                            "elapsed": elapsed,
                        }
                        if at is not None:
                            log_entry["auto_tuner"] = at.stats()
                            log_entry["control"] = self.current_control.to_dict() if self.current_control else {}
                        if log_callback:
                            log_callback(log_entry)
                        else:
                            line = (f"Step {self.global_step} | Loss: {avg_loss:.4f} | "
                                    f"LR: {lr:.6f} | Grad: {grad_norm:.2f} | "
                                    f"Tok/s: {tok_per_sec:,.0f}")
                            if at is not None:
                                line += f" | AT: {at.state} LR×{at.lr_mult:.2f}"
                                if self.current_control:
                                    line += f" | Phase: {self.current_control.phase}"
                            print(line)
                        self.total_loss = 0.0

                    # Evaluation
                    if eval_dataloader and self.global_step % args.eval_steps == 0:
                        eval_loss = self.evaluate(eval_dataloader)
                        if at is not None:
                            at.capture_eval(self.global_step, eval_loss)
                        if eval_loss < self.best_eval_loss:
                            self.best_eval_loss = eval_loss
                            self._save_checkpoint(best=True)
                        self.model.train()

                    # Checkpoint
                    if self.global_step % args.save_steps == 0:
                        self._save_checkpoint()

                if args.max_steps > 0 and self.global_step >= args.max_steps:
                    break

            if args.max_steps > 0 and self.global_step >= args.max_steps:
                break

        self._save_checkpoint(final=True)
        print(f"Training complete. Final step: {self.global_step}")

    @torch.no_grad()
    def evaluate(self, eval_dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for i, batch in enumerate(eval_dataloader):
            if self.args.max_eval_samples > 0 and i >= self.args.max_eval_samples:
                break
            batch = {k: v.to(self.device) for k, v in batch.items()}
            output = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                output_mtp_loss=False,
            )
            total_loss += output["loss"].item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  Eval loss: {avg_loss:.4f} ({n_batches} batches)")
        return avg_loss

    def apply_training_control(self, control: TrainingControl):
        """Apply TrainingControl to trainer state."""
        self.current_control = control

        # Apply LR multiplier to optimizer
        base_lr = self.args.learning_rate
        effective_lr = base_lr * control.hyperparams.lr_mult
        for group in self.optimizer.param_groups:
            group["lr"] = effective_lr

        # Apply weight decay multiplier
        base_wd = self.args.weight_decay
        effective_wd = base_wd * control.hyperparams.wd_mult
        for i, group in enumerate(self.optimizer.param_groups):
            if i == 0:  # params with weight decay
                group["weight_decay"] = effective_wd

        # Apply gradient norm multiplier
        self._effective_grad_norm = self.args.max_grad_norm * control.hyperparams.gn_mult

        # Apply momentum boost
        if control.hyperparams.mom_boost != 0.0:
            for group in self.optimizer.param_groups:
                if "betas" in group:
                    beta1 = group["betas"][0]
                    group["betas"] = (min(0.99, beta1 + control.hyperparams.mom_boost), group["betas"][1])

        # Apply optimizer betas
        for group in self.optimizer.param_groups:
            if "betas" in group:
                group["betas"] = (control.optimizer.beta1, control.optimizer.beta2)
            if "eps" in group:
                group["eps"] = control.optimizer.eps

        # Apply gradient accumulation
        if control.batch.grad_accum != self.args.gradient_accumulation_steps:
            self.args.gradient_accumulation_steps = control.batch.grad_accum

        # Apply MTP settings
        self.config.mtp.enabled = control.mtp.enabled
        self.config.mtp.depth = control.mtp.depth

        # Apply MoE settings
        self.config.moe.routing_temperature = control.moe.routing_temperature
        self.config.moe.load_balance_weight = control.moe.load_balance_weight
        self.config.moe.expert_capacity_factor = control.moe.expert_capacity_factor

        # Apply dataset mode to curriculum router
        if self.curriculum_router:
            phase = control.phase
            self.curriculum_router.set_phase(phase, control.step)
            if control.dataset.category_weights:
                self.curriculum_router.set_manual_weights(control.dataset.category_weights)

    def _get_grad_norm_limit(self):
        """Get current gradient norm limit (may be modified by TrainingControl)."""
        return getattr(self, '_effective_grad_norm', self.args.max_grad_norm)

    def _get_grad_norm(self) -> float:
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def _compute_tok_per_sec(self, batch: Dict, elapsed: float) -> float:
        if elapsed <= 0:
            return 0.0
        n_tokens = batch["input_ids"].numel() * self.args.gradient_accumulation_steps
        return n_tokens * self.global_step / max(elapsed, 1.0)

    def _save_checkpoint(self, final: bool = False, best: bool = False):
        if final:
            save_dir = os.path.join(self.args.output_dir, "final")
        elif best:
            save_dir = os.path.join(self.args.output_dir, "best")
        else:
            save_dir = os.path.join(self.args.output_dir, f"checkpoint-{self.global_step}")

        os.makedirs(save_dir, exist_ok=True)

        torch.save(self.model.state_dict(), os.path.join(save_dir, "model.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(save_dir, "optimizer.pt"))

        with open(os.path.join(save_dir, "training_state.json"), "w") as f:
            state = {
                "global_step": self.global_step,
                "epoch": self.epoch,
                "best_eval_loss": self.best_eval_loss,
            }
            if self.auto_tuner is not None:
                state["auto_tuner_state"] = self.auto_tuner.save_state()
            if self.curriculum_router is not None:
                state["router_state"] = {
                    "phase": self.curriculum_router.get_phase(),
                    "weights": self.curriculum_router.get_weights(),
                    "active_categories": list(self.curriculum_router.get_active_categories()),
                }
            json.dump(state, f, indent=2)

        config_dict = {
            "model_name": self.config.model_name,
            "version": self.config.version,
            "vocab_size": self.config.vocab_size,
            "hidden_size": self.config.architecture.hidden_size,
            "num_layers": self.config.architecture.num_layers,
        }
        with open(os.path.join(save_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=2)

        print(f"  ✓ Checkpoint saved: {save_dir}")

    def _load_checkpoint(self, checkpoint_dir: str, scheduler=None):
        model_path = os.path.join(checkpoint_dir, "model.pt")
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        state_path = os.path.join(checkpoint_dir, "training_state.json")

        if os.path.exists(model_path):
            model_sd = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(model_sd, strict=False)
            del model_sd
            print(f"  Loaded model from {model_path}")

        if os.path.exists(optimizer_path):
            try:
                opt_sd = torch.load(optimizer_path, map_location=self.device, weights_only=True)
            except Exception:
                opt_sd = torch.load(optimizer_path, map_location=self.device, weights_only=False)
            self.optimizer.load_state_dict(opt_sd)
            del opt_sd
            print(f"  Loaded optimizer from {optimizer_path}")

        if os.path.exists(state_path):
            with open(state_path, "r") as f:
                state = json.load(f)
                self.global_step = state.get("global_step", 0)
                self.epoch = state.get("epoch", 0)
                self.best_eval_loss = state.get("best_eval_loss", float("inf"))
            print(f"  Resumed from step {self.global_step}")
